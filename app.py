from flask import Flask, request, jsonify, render_template
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import json
import logging
import datetime
import time
import threading
import collections
import math
import statistics
from groq import Groq
from dotenv import load_dotenv

# Optional: Dramatically speeds up JSON payload transfers
try:
    from flask_compress import Compress
    HAS_COMPRESS = True
except ImportError:
    HAS_COMPRESS = False

load_dotenv()

app = Flask(__name__)
if HAS_COMPRESS:
    Compress(app)

# ==========================================================
# LOGGING
# ==========================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("skysense")

# ==========================================================
# CONFIGURATION
# ==========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY is not configured.")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set; AI features will report as unconfigured.")

app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB

# ----------------------------------------------------------
# HTTP CONNECTION POOLING (Massive Speedup for API calls)
# ----------------------------------------------------------
http_session = requests.Session()
retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[408, 429, 500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retries)
http_session.mount('http://', adapter)
http_session.mount('https://', adapter)
HTTP_TIMEOUT = 8

# ----------------------------------------------------------
# SERVER-SIDE CACHE
# ----------------------------------------------------------
CACHE_LOCK = threading.Lock()
WEATHER_CACHE = {}
LOCATION_CACHE = {}

WEATHER_CACHE_TTL = 300          
WEATHER_STALE_TTL = 1800         
LOCATION_CACHE_TTL = 3600        

# ----------------------------------------------------------
# RATE LIMITER
# ----------------------------------------------------------
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_HITS = collections.defaultdict(list)
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "500"))
CHAT_RATE_LIMIT = 15        
CHAT_RATE_WINDOW = 60       

def get_client_id():
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.remote_addr or "unknown"

def is_rate_limited(client_id, limit=CHAT_RATE_LIMIT, window=CHAT_RATE_WINDOW):
    now = time.monotonic()
    with RATE_LIMIT_LOCK:
        hits = RATE_LIMIT_HITS[client_id]
        cutoff = now - window
        while hits and hits[0] < cutoff:
            hits.pop(0)
        if len(hits) >= limit:
            return True
        hits.append(now)
        return False

def cache_get(store, key, ttl):
    now = time.monotonic()
    with CACHE_LOCK:
        item = store.get(key)
        if not item or (now - item["time"] > ttl):
            return None
        return item["value"]

cache_get_stale = cache_get

def cache_set(store, key, value):
    with CACHE_LOCK:
        store[key] = {"time": time.monotonic(), "value": value}
        if len(store) > CACHE_MAX_ITEMS:
            oldest_keys = sorted(store, key=lambda k: store[k].get("time", 0))[:max(1, len(store) - CACHE_MAX_ITEMS)]
            for old_key in oldest_keys:
                store.pop(old_key, None)

def request_with_pool(method, url, *, params=None, headers=None, timeout=HTTP_TIMEOUT):
    """Uses persistent TCP connections to eliminate SSL handshake overhead."""
    try:
        response = http_session.request(method, url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise RuntimeError("HTTP request failed.") from exc

def weather_cache_key(lat, lon):
    return f"{float(lat):.3f},{float(lon):.3f}"

def json_safe_copy(value):
    return json.loads(json.dumps(value, default=str))

# ==========================================================
# GROQ HELPER
# ==========================================================
DEFAULT_SYSTEM_PROMPT = "You are SkySense AI, a helpful, accurate and friendly AI assistant. Answer the user's question naturally."

def generate_ai_text(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=0.4):
    if client is None:
        raise RuntimeError("Groq client is not configured.")
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500
    )
    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("Groq returned an empty response.")
    return response.choices[0].message.content.strip()

# ==========================================================
# ROUTES
# ==========================================================
@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "geolocation=(self), microphone=(), camera=()")
    response.headers.setdefault("X-XSS-Protection", "0")
    if request.is_secure:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response

@app.errorhandler(413)
def handle_payload_too_large(_error):
    return jsonify({"error": "Request body is too large.", "error_type": "payload_too_large"}), 413

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", "weather": bool(OPENWEATHER_API_KEY),
        "chat": bool(GROQ_API_KEY and client), "model": GROQ_MODEL if GROQ_API_KEY else None
    })

MAX_CHAT_MESSAGE_LENGTH = 2000

@app.route('/api/chat', methods=['POST'])
def chat():
    client_id = get_client_id()
    if is_rate_limited(client_id):
        return jsonify({"reply": "You're sending messages a little too fast. Please wait a moment.", "error_type": "rate_limit"}), 429

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"reply": "I couldn't read that message. Please try again."}), 400

    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"reply": "Please enter a question first."}), 400

    if len(message) > MAX_CHAT_MESSAGE_LENGTH:
        return jsonify({"reply": f"That message is too long (max {MAX_CHAT_MESSAGE_LENGTH} characters).", "error_type": "message_too_long"}), 400

    context = data.get("context", {})
    if isinstance(context, str):
        try: context = json.loads(context)
        except ValueError: context = {"raw_context": context}
    if not isinstance(context, dict):
        context = {}

    weather_context = {
        "city": context.get("city"), "temperature": context.get("temperature"), "feels_like": context.get("feels_like"),
        "description": context.get("description"), "humidity": context.get("humidity"), "wind_speed": context.get("wind_speed"),
        "wind_dir": context.get("wind_dir"), "pressure": context.get("pressure"), "aqi": context.get("aqi"),
        "chance_of_rain": context.get("chance_of_rain"), "hourly_rain": context.get("hourly_rain", []),
        "forecast": context.get("forecast", []), "activities": context.get("activities", {})
    }

    system_prompt = """
You are SkySense AI, a smart and friendly assistant. Answer general questions or use the supplied weather context.
Never invent weather numbers, forecast times, or rain probabilities. Keep simple answers concise and practical.
"""
    prompt = f"SKYSENSE WEATHER DATA:\n{json.dumps(weather_context, ensure_ascii=False, default=str)}\n\nUSER:\n{message}"

    if client is None:
        return jsonify({"reply": "AI chatbot is not configured yet. Add GROQ_API_KEY.", "error_type": "configuration"}), 503

    try:
        reply = generate_ai_text(prompt, system_prompt=system_prompt, temperature=0.2)
        return jsonify({"reply": reply})
    except Exception as e:
        logger.error("SkySense chat failed: %s: %s", type(e).__name__, e)
        error = str(e).lower()
        if "429" in error: return jsonify({"reply": "SkySense AI has reached its current request limit.", "error_type": "rate_limit"}), 429
        if "401" in error: return jsonify({"reply": "SkySense AI authentication failed.", "error_type": "authentication"}), 502
        return jsonify({"reply": "SkySense AI could not process the request right now.", "error_type": "ai_service"}), 502

# ==========================================================
# SKYSENSE INTELLIGENCE ENGINE
# ==========================================================
def create_weather_insights(city, temp, feels_like, description, humidity, wind_speed, aqi_text, rain_chance):
    rain = int(rain_chance or 0)
    wind_kmh = round(float(wind_speed) * 3.6)

    if rain >= 70:
        insight = f"{city} has a high chance of rain ({rain}%). Conditions are {description}."
        travel = "Carry rain protection and allow extra travel time."
    elif rain >= 40:
        insight = f"{city} has a moderate rain signal ({rain}%). Current conditions: {description}."
        travel = "A light rain layer or umbrella is sensible."
    elif temp >= 34:
        insight = f"{city} is {round(temp)}°C. Heat is the main factor, better during cooler hours."
        travel = "Carry water, use sun protection, and avoid peak heat."
    elif temp <= 12:
        insight = f"{city} is cool at {round(temp)}°C. A warm layer may improve comfort."
        travel = "Carry a light warm layer."
    else:
        insight = f"Conditions in {city} are {description} at {round(temp)}°C. Broadly manageable."
        travel = "Normal travel plans should be fine."

    def score(base):
        value = base
        if rain >= 70: value -= 4
        elif rain >= 40: value -= 2
        elif rain >= 20: value -= 1
        if wind_kmh >= 40: value -= 2
        elif wind_kmh >= 30: value -= 1
        if temp >= 38 or temp <= 5: value -= 3
        elif temp >= 34 or temp <= 10: value -= 1
        if aqi_text in ["Poor", "Very Poor"]: value -= (2 if aqi_text == "Poor" else 4)
        return max(1, min(10, value))

    scores = {act: f"{score(8)}/10" for act in ["cycling", "running", "hiking", "photography", "motorcycling"]}
    return insight, travel, scores

def _clamp(value, low=0, high=100): return max(low, min(high, value))
def _num(value, default=0.0):
    try: return float(value)
    except (TypeError, ValueError): return default

def calculate_skyscore(temp, feels_like, humidity, wind_kmh, rain_chance, aqi_text, uv_index=None, visibility_km=None, cloud_pct=None):
    score = 100.0
    t = _num(feels_like, _num(temp, 25))
    if t < 12: score -= min(22, (12 - t) * 1.6)
    elif t > 30: score -= min(28, (t - 30) * 1.8)
    elif t < 18: score -= (18 - t) * 0.8
    elif t > 28: score -= (t - 28) * 1.0

    score -= _clamp(_num(rain_chance), 0, 100) * 0.20
    wind = max(0, _num(wind_kmh))
    if wind > 45: score -= min(18, (wind - 45) * 0.45)
    elif wind > 30: score -= (wind - 30) * 0.25

    humidity_n = _num(humidity)
    if humidity_n > 85: score -= min(10, (humidity_n - 85) * 0.4)
    elif humidity_n < 25: score -= min(5, (25 - humidity_n) * 0.2)

    score -= {"Good": 0, "Fair": 3, "Moderate": 9, "Poor": 20, "Very Poor": 35, "Unavailable": 0}.get(aqi_text, 0)
    
    if uv_index is not None:
        uv = _num(uv_index)
        score -= (8 if uv >= 11 else 5 if uv >= 8 else 2 if uv >= 6 else 0)
    if visibility_km is not None:
        vis = _num(visibility_km)
        score -= (12 if vis < 2 else 6 if vis < 5 else 0)
    if cloud_pct is not None and _num(cloud_pct) >= 95:
        score -= 3

    score = int(round(_clamp(score)))
    label = "Excellent" if score >= 85 else "Very Good" if score >= 75 else "Good" if score >= 60 else "Fair" if score >= 45 else "Challenging"
    return {"score": score, "label": label}

def calculate_forecast_confidence(forecast_list):
    if not forecast_list: return {"score": 0, "label": "Unavailable", "reason": "No forecast data"}
    temps, pops = [], []
    for item in forecast_list[:16]:
        try:
            temps.append(float(item.get("main", {}).get("temp")))
            pops.append(float(item.get("pop", 0)) * 100)
        except (TypeError, ValueError): continue
    if len(temps) < 2: return {"score": 50, "label": "Moderate", "reason": "Limited forecast samples"}

    temp_instability = min(1.0, statistics.mean([abs(b - a) for a, b in zip(temps, temps[1:])]) / 8.0)
    pop_instability = min(1.0, statistics.mean([abs(b - a) for a, b in zip(pops, pops[1:])]) / 60.0)
    score = int(round(_clamp(96 - 45 * temp_instability - 35 * pop_instability)))
    label = "High" if score >= 80 else "Moderate" if score >= 60 else "Low"
    reason = "Forecast trend is consistent." if score >= 80 else "Changes are noticeable." if score >= 60 else "Forecast changes are volatile."
    return {"score": score, "label": label, "reason": reason}

def build_weather_story(temp, feels_like, description, rain_chance, wind_kmh, aqi_text):
    rain, t, feels, wind = int(_num(rain_chance)), round(_num(temp)), round(_num(feels_like)), round(_num(wind_kmh))
    return {
        "headline": f"{description.title()} with a current temperature of {t}°C.",
        "morning": "Comfortable" if 16 <= t <= 29 else ("Cool" if t < 16 else "Warm"),
        "afternoon": "Heat may be the main factor" if feels >= 33 else "Generally manageable",
        "evening": "Rain risk is elevated" if rain >= 60 else ("Keep an umbrella nearby" if rain >= 35 else "Rain risk is relatively low"),
        "summary": f"Feels like {feels}°C, wind around {wind} km/h, rain probability {rain}%, AQI {aqi_text}."
    }

def calculate_activity_scores(temp, feels_like, humidity, wind_kmh, rain_chance, aqi_text, uv_index=None):
    base = calculate_skyscore(temp, feels_like, humidity, wind_kmh, rain_chance, aqi_text, uv_index)["score"]
    rain, wind, feels = _num(rain_chance), _num(wind_kmh), _num(feels_like, _num(temp, 25))
    aqi_penalty = {"Good": 0, "Fair": 2, "Moderate": 7, "Poor": 18, "Very Poor": 30}.get(aqi_text, 0)

    def activity(extra=0, heat_limit=None, wind_limit=None):
        s = base + extra
        if heat_limit and feels > heat_limit: s -= min(20, (feels - heat_limit) * 1.5)
        if wind_limit and wind > wind_limit: s -= min(18, (wind - wind_limit) * 0.6)
        s -= (18 if rain >= 60 else 8 if rain >= 35 else 0)
        s -= aqi_penalty * 0.25
        return int(round(_clamp(s)))

    return {
        "motorcycling": activity(-1, heat_limit=35, wind_limit=35), "cycling": activity(0, heat_limit=32, wind_limit=30),
        "running": activity(-2, heat_limit=30, wind_limit=25), "hiking": activity(1, heat_limit=33, wind_limit=40),
        "photography": activity(2, heat_limit=36, wind_limit=45), "outdoor_dining": activity(0, heat_limit=32, wind_limit=25),
        "beach": activity(2, heat_limit=36, wind_limit=35)
    }

def build_risk_alerts(temp, feels_like, wind_kmh, rain_chance, aqi_text, visibility_km=None):
    alerts = []
    feels, wind, rain = _num(feels_like, _num(temp)), _num(wind_kmh), _num(rain_chance)
    
    if feels >= 40: alerts.append({"level": "critical", "icon": "🔥", "title": "Extreme heat", "message": "Limit prolonged exposure."})
    elif feels >= 35: alerts.append({"level": "warning", "icon": "🌡️", "title": "High heat", "message": "Heat stress possible."})
    
    if rain >= 80: alerts.append({"level": "critical", "icon": "🌧️", "title": "Very high rain risk", "message": "Heavy rain possible."})
    elif rain >= 60: alerts.append({"level": "warning", "icon": "☔", "title": "High rain risk", "message": "Rain is likely."})
    
    if wind >= 45: alerts.append({"level": "warning", "icon": "💨", "title": "Strong wind", "message": "Strong winds may affect plans."})
    if aqi_text in ("Poor", "Very Poor"): alerts.append({"level": "warning", "icon": "😷", "title": "Poor air quality", "message": "Consider reducing strenuous activity."})
    if visibility_km is not None and _num(visibility_km) < 2: alerts.append({"level": "warning", "icon": "🌫️", "title": "Low visibility", "message": "Reduced visibility outdoors."})
    return alerts

def build_intelligence_payload(temp, feels_like, humidity, wind_speed, rain_chance, aqi_text, description, forecast_list, daily_forecasts, uv_index=None, visibility_km=None, cloud_pct=None):
    wind_kmh = round(_num(wind_speed) * 3.6)
    return {
        "skyscore": calculate_skyscore(temp, feels_like, humidity, wind_kmh, rain_chance, aqi_text, uv_index, visibility_km, cloud_pct),
        "forecast_confidence": calculate_forecast_confidence(forecast_list),
        "activities": calculate_activity_scores(temp, feels_like, humidity, wind_kmh, rain_chance, aqi_text, uv_index),
        "alerts": build_risk_alerts(temp, feels_like, wind_kmh, rain_chance, aqi_text, visibility_km),
        "weather_story": build_weather_story(temp, feels_like, description, rain_chance, wind_kmh, aqi_text),
        "wind_kmh": wind_kmh
    }

class ApiError(Exception):
    def __init__(self, message, error_type, status_code):
        super().__init__(message)
        self.payload = {"error": message, "error_type": error_type}
        self.status_code = status_code
    def response(self): return jsonify(self.payload), self.status_code

def resolve_city_location(query):
    location_key = query.lower()
    cached = cache_get(LOCATION_CACHE, location_key, LOCATION_CACHE_TTL)
    if cached: return cached["lat"], cached["lon"], cached["name"]

    try:
        geo_res = request_with_pool("GET", "https://nominatim.openstreetmap.org/search", params={"q": query, "format": "json", "limit": 1}, headers={"User-Agent": "SkySenseAI/1.0"})
        places = geo_res.json()
    except (RuntimeError, ValueError):
        raise ApiError("Location service is temporarily unavailable.", "location_service", 503)

    if not places:
        raise ApiError("Location not found. Check the spelling.", "location_not_found", 404)

    place = places[0]
    parts = place.get("display_name", query).split(",")
    resolved_name = f"{parts[0].strip()}, {parts[1].strip()}" if len(parts) > 1 else parts[0].strip()
    cache_set(LOCATION_CACHE, location_key, {"lat": place["lat"], "lon": place["lon"], "name": resolved_name})
    return place["lat"], place["lon"], resolved_name

def validate_coordinates(lat, lon):
    try:
        lat_f, lon_f = float(lat), float(lon)
        if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180): raise ValueError
        return lat_f, lon_f
    except (TypeError, ValueError):
        raise ApiError("Invalid location coordinates.", "invalid_coordinates", 400)

def fetch_air_quality(lat_float, lon_float):
    try:
        res = request_with_pool("GET", "https://api.openweathermap.org/data/2.5/air_pollution", params={"lat": lat_float, "lon": lon_float, "appid": OPENWEATHER_API_KEY})
        data = res.json()
        if data.get("list"): return {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}.get(data["list"][0]["main"]["aqi"], "Unavailable")
    except Exception: pass
    return "Unavailable"

def fetch_forecast(lat_float, lon_float):
    daily_forecasts, hourly_rain, chance_of_rain = [], [], 0
    try:
        res = request_with_pool("GET", "https://api.openweathermap.org/data/2.5/forecast", params={"lat": lat_float, "lon": lon_float, "appid": OPENWEATHER_API_KEY, "units": "metric"})
        forecast_list = res.json().get("list", [])
    except Exception:
        return [], [], [], 0, "unavailable"

    if not forecast_list: return [], [], [], 0, "unavailable"
    chance_of_rain = int(forecast_list[0].get("pop", 0) * 100)
    daily_buckets = collections.defaultdict(lambda: {"temps": [], "rain": [], "descriptions": []})

    for i, item in enumerate(forecast_list):
        try:
            date_key = item["dt_txt"].split(" ")[0]
            temp, pop, desc = float(item["main"]["temp"]), int(item.get("pop", 0) * 100), item["weather"][0]["main"]
            
            if i < 8:
                t_obj = datetime.datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")
                hourly_rain.append({"time": t_obj.strftime("%I %p"), "pop": pop, "desc": desc})

            daily_buckets[date_key]["temps"].append(temp)
            daily_buckets[date_key]["rain"].append(pop)
            daily_buckets[date_key]["descriptions"].append(desc)
        except Exception: continue

    for date_key, bucket in list(daily_buckets.items())[:5]:
        d_obj = datetime.datetime.strptime(date_key, "%Y-%m-%d")
        daily_forecasts.append({
            "date": d_obj.strftime("%d %b"), "day": d_obj.strftime("%a"),
            "temp": round(max(bucket["temps"])), "min_temp": round(min(bucket["temps"])),
            "description": max(set(bucket["descriptions"]), key=bucket["descriptions"].count),
            "pop": max(bucket["rain"]) if bucket["rain"] else 0
        })

    return forecast_list, daily_forecasts, hourly_rain, chance_of_rain, "live"

@app.route('/api/weather', methods=['GET'])
def weather():
    query = request.args.get('city', 'Pune').strip()
    try:
        lat, lon = request.args.get('lat'), request.args.get('lon')
        if not lat or not lon:
            lat, lon, query = resolve_city_location(query)
        lat_f, lon_f = validate_coordinates(lat, lon)
    except ApiError as e: return e.response()

    cache_key = weather_cache_key(lat_f, lon_f)
    cached = cache_get(WEATHER_CACHE, cache_key, WEATHER_CACHE_TTL)
    if cached:
        res = json_safe_copy(cached)
        res.update({"cached": True, "cache_age_seconds": 0})
        return jsonify(res)

    try:
        curr_res = request_with_pool("GET", "https://api.openweathermap.org/data/2.5/weather", params={"lat": lat_f, "lon": lon_f, "appid": OPENWEATHER_API_KEY, "units": "metric"})
        data = curr_res.json()
    except Exception:
        stale = cache_get_stale(WEATHER_CACHE, cache_key, WEATHER_STALE_TTL)
        if stale:
            res = json_safe_copy(stale)
            res.update({"cached": True, "stale": True, "warning": "Live service unavailable. Showing cached data."})
            return jsonify(res)
        return jsonify({"error": "Weather service temporarily unavailable.", "error_type": "weather_service"}), 503

    try:
        temp, feels_like, desc, humidity = data["main"]["temp"], data["main"]["feels_like"], data["weather"][0]["description"], data["main"]["humidity"]
        wind_speed, pressure, wind_deg = data["wind"]["speed"], data["main"]["pressure"], data["wind"].get("deg", 0)
        timezone_shift = data.get("timezone", 0)
    except KeyError:
        return jsonify({"error": "Weather service returned incomplete data.", "error_type": "weather_data"}), 502

    wind_dir = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((wind_deg / 45) + 0.5) % 8]
    try:
        sunrise = datetime.datetime.fromtimestamp(data["sys"]["sunrise"] + timezone_shift, datetime.timezone.utc).strftime("%I:%M %p")
        sunset = datetime.datetime.fromtimestamp(data["sys"]["sunset"] + timezone_shift, datetime.timezone.utc).strftime("%I:%M %p")
    except Exception: sunrise, sunset = "N/A", "N/A"

    aqi_text = fetch_air_quality(lat_f, lon_f)
    forecast_list, daily_forecasts, hourly_rain, chance_of_rain, forecast_status = fetch_forecast(lat_f, lon_f)

    insight_text, travel_text, _ = create_weather_insights(query, temp, feels_like, desc, humidity, wind_speed, aqi_text, chance_of_rain)
    intel = build_intelligence_payload(temp, feels_like, humidity, wind_speed, chance_of_rain, aqi_text, desc, forecast_list, daily_forecasts, data.get("uvi"), data.get("visibility", 0) / 1000 if data.get("visibility") is not None else None, data.get("clouds", {}).get("all"))

    activities = {k: f"{v}/10" for k, v in intel["activities"].items()}
    chat_context = f"Current in {query}: {temp}°C, {desc}. Feels like {feels_like}°C. Humidity: {humidity}%. Wind: {round(wind_speed * 3.6)} km/h {wind_dir}. AQI: {aqi_text}."
    for item in forecast_list[:16]:
        chat_context += f" [{item.get('dt_txt', '')} -> {round(item.get('main', {}).get('temp', temp))}°C, {item.get('weather', [{}])[0].get('main', 'Unknown')}, {round(item.get('pop', 0) * 100)}% rain]"

    result = {
        "city": query, "latitude": lat_f, "longitude": lon_f, "temperature": temp, "feels_like": feels_like,
        "description": desc, "humidity": humidity, "wind_speed": wind_speed, "pressure": pressure, "wind_dir": wind_dir,
        "sunrise": sunrise, "sunset": sunset, "aqi": aqi_text, "chance_of_rain": f"{chance_of_rain}%",
        "hourly_rain": hourly_rain, "ai_summary": insight_text, "travel_advice": travel_text, "activities": activities,
        "forecast": daily_forecasts, "forecast_status": forecast_status, "skyscore": intel["skyscore"],
        "forecast_confidence": intel["forecast_confidence"], "weather_story": intel["weather_story"],
        "risk_alerts": intel["alerts"], "activity_scores": intel["activities"], "wind_kmh": intel["wind_kmh"],
        "chatbot_context": chat_context, "cached": False, "stale": False
    }

    cache_set(WEATHER_CACHE, cache_key, json_safe_copy(result))
    return jsonify(result)

# ==========================================================
# RENDER STARTUP
# ==========================================================
if __name__ == '__main__':
    # For Production: Use Gunicorn to run this app for async concurrency.
    # Command: gunicorn -w 4 -k gthread --threads 4 app:app
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
