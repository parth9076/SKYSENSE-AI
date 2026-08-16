from flask import Flask, request, jsonify, render_template
import requests
import os
import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ==========================================================
# CONFIGURATION
# ==========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured.")

if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY is not configured.")

# Groq is used ONLY by the chatbot.
# /api/weather does NOT call Groq.
client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "llama-3.3-70b-versatile"


# ==========================================================
# GEMINI HELPER
# ==========================================================

def generate_ai_text(prompt):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are SkySense AI, a helpful, accurate and friendly "
                    "AI assistant. Answer the user's question naturally."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.4,
        max_tokens=500
    )

    reply = response.choices[0].message.content

    if not reply:
        raise RuntimeError("Groq returned an empty response.")

    return reply.strip()


# ==========================================================
# HOME
# ==========================================================

@app.route('/')
def home():
    return render_template('index.html')


# ==========================================================
# GENERAL AI CHATBOT
# ==========================================================

@app.route('/api/chat', methods=['POST'])
def chat():

    data = request.get_json(silent=True) or {}

    user_message = (data.get('message') or '').strip()
    weather_context = data.get('context') or (
        "No current weather information is available."
    )

    if not user_message:
        return jsonify({
            "reply": "Please enter a question."
        }), 400

    prompt = f"""
You are SkySense AI, an intelligent, friendly and helpful AI assistant.

The user can ask ANY question. Do not restrict the user to predefined
weather questions.

If the question is related to weather, use the current weather context
below.

Weather-related topics include:
- Current weather
- Forecasts
- Rain
- Temperature
- Humidity
- Wind
- Air quality
- Travel
- Clothing
- Cycling
- Running
- Hiking
- Photography
- Motorcycling
- Weather science
- Outdoor activities

For questions unrelated to weather, answer normally using your general
knowledge.

Do not mention that you are restricted to weather questions.

Keep simple questions concise. For questions requiring explanation,
provide enough detail to make the answer useful.

If the user asks for programming, computer science, AI, cloud computing,
technology, study help, or general knowledge, answer normally.

CURRENT WEATHER CONTEXT:
{weather_context}

USER QUESTION:
{user_message}
"""

    try:

        reply = generate_ai_text(prompt)

        return jsonify({
            "reply": reply
        })

    except Exception as e:

        print(
            f"Groq Chat Error: {type(e).__name__}: {e}",
            flush=True
        )

        # Specific response for quota exhaustion
        if "429" in str(e) or "rate_limit" in str(e).lower():

            return jsonify({
                "reply": (
                    "SkySense AI has temporarily reached its AI request "
                    "limit. Please try again later."
                )
            }), 429

        return jsonify({
            "reply": (
                "I'm having trouble connecting to my AI network "
                "right now. Please try again."
            )
        }), 500


# ==========================================================
# RULE-BASED WEATHER INSIGHTS
#
# IMPORTANT:
# This function intentionally does NOT call Gemini.
# It prevents every page refresh from consuming Gemini quota.
# ==========================================================

def create_weather_insights(
    city,
    temp,
    feels_like,
    description,
    humidity,
    wind_speed,
    aqi_text,
    chance_of_rain
):

    wind_kmh = round(wind_speed * 3.6)

    # ------------------------------------------------------
    # Weather insight
    # ------------------------------------------------------

    if chance_of_rain >= 70:
        rain_advice = (
            f"There is a high chance of rain ({chance_of_rain}%), "
            "so keep an umbrella or rain protection nearby."
        )
    elif chance_of_rain >= 40:
        rain_advice = (
            f"There is a moderate chance of rain ({chance_of_rain}%), "
            "so conditions may change later in the day."
        )
    else:
        rain_advice = (
            f"The rain chance is relatively low at {chance_of_rain}%, "
            "so outdoor plans are less likely to be disrupted by rain."
        )

    if temp >= 35:
        temperature_advice = (
            "Temperatures are very high, so limit prolonged exposure "
            "to direct sunlight and stay hydrated."
        )
    elif temp >= 30:
        temperature_advice = (
            "It is warm outside, so lighter clothing and regular hydration "
            "are recommended."
        )
    elif temp <= 10:
        temperature_advice = (
            "Temperatures are cool, so a warm outer layer may be useful."
        )
    else:
        temperature_advice = (
            "Temperatures are in a generally comfortable range for many "
            "outdoor activities."
        )

    if humidity >= 80:
        humidity_advice = (
            "Humidity is high, which can make it feel warmer and less "
            "comfortable during exercise."
        )
    elif humidity <= 35:
        humidity_advice = (
            "Humidity is relatively low, so hydration can be important "
            "during longer outdoor activities."
        )
    else:
        humidity_advice = (
            "Humidity is at a moderate level."
        )

    insight_text = (
        f"Conditions in {city} are currently {description} at "
        f"{round(temp)}°C, with a feels-like temperature of "
        f"{round(feels_like)}°C. {rain_advice} {temperature_advice}"
    )

    # ------------------------------------------------------
    # Travel advice
    # ------------------------------------------------------

    if chance_of_rain >= 60:
        travel_text = (
            "Carry an umbrella or rain jacket, protect electronics, "
            "and allow extra travel time if roads become wet."
        )
    elif temp >= 33:
        travel_text = (
            "Wear light clothing, carry water, and consider avoiding "
            "long periods outdoors during the hottest part of the day."
        )
    elif aqi_text in ("Poor", "Very Poor"):
        travel_text = (
            "Consider reducing prolonged outdoor exposure and carry "
            "appropriate air-quality protection if needed."
        )
    else:
        travel_text = (
            "Conditions look generally suitable for travel; "
            "carry water and dress according to the current temperature."
        )

    # ------------------------------------------------------
    # Activity scoring
    # ------------------------------------------------------

    def base_score():
        score = 10

        if chance_of_rain >= 80:
            score -= 4
        elif chance_of_rain >= 60:
            score -= 3
        elif chance_of_rain >= 40:
            score -= 1

        if temp >= 38 or temp <= 5:
            score -= 3
        elif temp >= 34 or temp <= 10:
            score -= 2

        if humidity >= 85:
            score -= 2
        elif humidity >= 75:
            score -= 1

        if wind_kmh >= 45:
            score -= 2
        elif wind_kmh >= 30:
            score -= 1

        if aqi_text == "Poor":
            score -= 2
        elif aqi_text == "Very Poor":
            score -= 4
        elif aqi_text == "Moderate":
            score -= 1

        return max(1, min(10, score))

    general_score = base_score()

    cycling = general_score
    running = max(1, general_score - (1 if humidity >= 75 else 0))
    hiking = general_score
    photography = min(
        10,
        general_score + (
            1 if chance_of_rain < 30 and temp < 34 else 0
        )
    )
    motorcycling = general_score

    return (
        insight_text,
        travel_text,
        {
            "cycling": f"{cycling}/10",
            "running": f"{running}/10",
            "hiking": f"{hiking}/10",
            "photography": f"{photography}/10",
            "motorcycling": f"{motorcycling}/10"
        }
    )


# ==========================================================
# WEATHER API
# ==========================================================


# ==========================================================
# FORECAST INTELLIGENCE / MULTI-MODEL CONSENSUS
# ==========================================================
#
# Compares three global numerical weather prediction systems:
# ECMWF IFS, NOAA GFS, and DWD ICON.
#
# Open-Meteo exposes forecasts from multiple national weather
# services and allows individual model selection. This endpoint
# calculates a simple consensus/dispersion signal from the next
# 24 forecast hours.
#
# This is guidance, NOT an official probability of correctness.
# ==========================================================

@app.route('/api/forecast-intelligence', methods=['GET'])
def forecast_intelligence():

    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if not lat or not lon:
        return jsonify({
            "error": "Latitude and longitude are required."
        }), 400

    try:
        latitude = float(lat)
        longitude = float(lon)
    except ValueError:
        return jsonify({
            "error": "Invalid coordinates."
        }), 400

    base_url = "https://api.open-meteo.com/v1/forecast"

    models = {
        "ECMWF IFS": "ecmwf_ifs",
        "NOAA GFS": "gfs_seamless",
        "DWD ICON": "icon_seamless"
    }

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,"
            "precipitation_probability,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "forecast_days": 3,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    model_results = {}

    for display_name, model_id in models.items():

        request_params = dict(params)
        request_params["models"] = model_id

        try:
            response = requests.get(
                base_url,
                params=request_params,
                timeout=20
            )
            response.raise_for_status()
            payload = response.json()

            hourly = payload.get("hourly", {})

            if hourly.get("time") and hourly.get("temperature_2m"):
                model_results[display_name] = hourly

        except requests.RequestException as e:
            print(
                f"{display_name} forecast error: "
                f"{type(e).__name__}: {e}",
                flush=True
            )

    if len(model_results) < 2:
        available = ", ".join(model_results.keys()) or "none"

        return jsonify({
            "error": (
                "Not enough weather models responded for consensus. "
                f"Available: {available}"
            )
        }), 502

    # Use the first 24 hours for the near-term consensus.
    first_hourly = next(iter(model_results.values()))
    times = first_hourly.get("time", [])
    horizon = min(24, len(times))

    def safe_float(value):
        try:
            number = float(value)
            if number == number and abs(number) != float("inf"):
                return number
        except (TypeError, ValueError):
            pass
        return None

    hourly_consensus = []

    for hour in range(horizon):

        temperatures = []
        rain_probabilities = []
        precipitation = []
        winds = []

        for hourly in model_results.values():

            if hour < len(hourly.get("temperature_2m", [])):
                value = safe_float(
                    hourly["temperature_2m"][hour]
                )
                if value is not None:
                    temperatures.append(value)

            if hour < len(hourly.get("precipitation_probability", [])):
                value = safe_float(
                    hourly["precipitation_probability"][hour]
                )
                if value is not None:
                    rain_probabilities.append(value)

            if hour < len(hourly.get("precipitation", [])):
                value = safe_float(
                    hourly["precipitation"][hour]
                )
                if value is not None:
                    precipitation.append(value)

            if hour < len(hourly.get("wind_speed_10m", [])):
                value = safe_float(
                    hourly["wind_speed_10m"][hour]
                )
                if value is not None:
                    winds.append(value)

        if not temperatures:
            continue

        temp_mean = sum(temperatures) / len(temperatures)
        temp_spread = max(temperatures) - min(temperatures)

        rain_mean = (
            sum(rain_probabilities) / len(rain_probabilities)
            if rain_probabilities else None
        )

        rain_amount = (
            sum(precipitation) / len(precipitation)
            if precipitation else None
        )

        wind_mean = (
            sum(winds) / len(winds)
            if winds else None
        )

        hourly_consensus.append({
            "time": times[hour],
            "temperature_mean": round(temp_mean, 1),
            "temperature_spread": round(temp_spread, 1),
            "rain_probability_mean": (
                round(rain_mean)
                if rain_mean is not None else None
            ),
            "precipitation_mean": (
                round(rain_amount, 2)
                if rain_amount is not None else None
            ),
            "wind_mean": (
                round(wind_mean, 1)
                if wind_mean is not None else None
            )
        })

    if not hourly_consensus:
        return jsonify({
            "error": "Unable to calculate model consensus."
        }), 502

    temp_spreads = [
        item["temperature_spread"]
        for item in hourly_consensus
    ]

    average_temp_spread = (
        sum(temp_spreads) / len(temp_spreads)
    )

    max_temp_spread = max(temp_spreads)

    # A simple model-agreement score:
    # <1.5°C average spread = strong agreement
    # <3°C = moderate agreement
    # otherwise = weaker agreement.
    if average_temp_spread <= 1.5:
        agreement = "Strong"
        confidence = "High"
    elif average_temp_spread <= 3.0:
        agreement = "Moderate"
        confidence = "Moderate"
    else:
        agreement = "Mixed"
        confidence = "Lower"

    # Find the largest model disagreement period.
    peak_disagreement = max(
        hourly_consensus,
        key=lambda item: item["temperature_spread"]
    )

    # 24-hour consensus rain signal.
    rain_values = [
        item["rain_probability_mean"]
        for item in hourly_consensus
        if item["rain_probability_mean"] is not None
    ]

    peak_rain = max(rain_values) if rain_values else None

    # Approximate next-24h temperature range from model means.
    mean_temperatures = [
        item["temperature_mean"]
        for item in hourly_consensus
    ]

    next_24_low = min(mean_temperatures)
    next_24_high = max(mean_temperatures)

    model_summary = {}

    for name, hourly in model_results.items():

        values = [
            safe_float(value)
            for value in hourly.get("temperature_2m", [])[:horizon]
        ]

        values = [
            value for value in values
            if value is not None
        ]

        if values:
            model_summary[name] = {
                "mean_temperature": round(
                    sum(values) / len(values),
                    1
                ),
                "low": round(min(values), 1),
                "high": round(max(values), 1)
            }

    return jsonify({

        "source": "ECMWF IFS + NOAA GFS + DWD ICON via Open-Meteo",

        "models_available": list(model_results.keys()),

        "model_count": len(model_results),

        "confidence": confidence,

        "confidence_note": (
            "All selected models are closely grouped."
            if agreement == "Strong"
            else
            "The models show some disagreement."
            if agreement == "Moderate"
            else
            "The models disagree noticeably; forecast uncertainty is higher."
        ),

        "agreement": agreement,

        "average_temperature_spread": round(
            average_temp_spread,
            1
        ),

        "max_temperature_spread": round(
            max_temp_spread,
            1
        ),

        "peak_disagreement_time": peak_disagreement["time"],

        "peak_rain_probability": peak_rain,

        "next_24h_low": round(next_24_low, 1),

        "next_24h_high": round(next_24_high, 1),

        "models": model_summary,

        "hourly": hourly_consensus[:12],

        "message": (
            "Model agreement is strong."
            if agreement == "Strong"
            else
            "Models show some differences; confidence is moderate."
            if agreement == "Moderate"
            else
            "Models disagree noticeably; treat the forecast with extra caution."
        )
    })



@app.route('/api/weather', methods=['GET'])
def weather():

    query = request.args.get('city', 'Pune').strip()

    lat = request.args.get('lat')
    lon = request.args.get('lon')

    resolved_city_name = query

    # ------------------------------------------------------
    # LOCATION SEARCH
    # ------------------------------------------------------

    if not lat or not lon:

        geo_url = (
            "https://nominatim.openstreetmap.org/search"
            f"?q={requests.utils.quote(query)}"
            "&format=json"
            "&limit=1"
        )

        headers = {
            'User-Agent': 'SkySenseAI-WeatherApp'
        }

        try:

            geo_res = requests.get(
                geo_url,
                headers=headers,
                timeout=15
            )

        except requests.RequestException:

            return jsonify({
                "error": "Unable to connect to the location service."
            }), 500

        if (
            geo_res.status_code == 200
            and len(geo_res.json()) > 0
        ):

            place = geo_res.json()[0]

            lat = place['lat']
            lon = place['lon']

            parts = place['display_name'].split(',')

            if len(parts) > 1:
                resolved_city_name = (
                    f"{parts[0].strip()}, {parts[1].strip()}"
                )
            else:
                resolved_city_name = parts[0].strip()

        else:

            return jsonify({
                "error": (
                    "Location not found. Please check your spelling "
                    "or try a nearby major city/PIN code."
                )
            }), 400

    # ------------------------------------------------------
    # CURRENT WEATHER
    # ------------------------------------------------------

    current_url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={OPENWEATHER_API_KEY}"
        "&units=metric"
    )

    try:

        current_response = requests.get(
            current_url,
            timeout=15
        )

    except requests.RequestException:

        return jsonify({
            "error": "Unable to connect to the weather service."
        }), 500

    if current_response.status_code != 200:

        return jsonify({
            "error": (
                "Could not retrieve weather telemetry "
                "for this coordinate."
            )
        }), 400

    data = current_response.json()

    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    description = data['weather'][0]['description']
    humidity = data['main']['humidity']
    wind_speed = data['wind']['speed']
    pressure = data['main']['pressure']

    # ------------------------------------------------------
    # WIND DIRECTION
    # ------------------------------------------------------

    wind_deg = data['wind'].get('deg', 0)

    dirs = [
        'N', 'NE', 'E', 'SE',
        'S', 'SW', 'W', 'NW'
    ]

    wind_dir = dirs[
        int((wind_deg / 42.5) + 0.5) % 8
    ]

    # ------------------------------------------------------
    # SUNRISE / SUNSET
    # ------------------------------------------------------

    timezone_shift = data.get('timezone', 0)

    sunrise_time = datetime.datetime.fromtimestamp(
        data['sys']['sunrise'] + timezone_shift,
        datetime.timezone.utc
    ).strftime('%I:%M %p')

    sunset_time = datetime.datetime.fromtimestamp(
        data['sys']['sunset'] + timezone_shift,
        datetime.timezone.utc
    ).strftime('%I:%M %p')

    # ------------------------------------------------------
    # AIR QUALITY
    # ------------------------------------------------------

    aqi_text = "Good"

    air_url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    try:

        air_response = requests.get(
            air_url,
            timeout=15
        )

        if air_response.status_code == 200:

            air_data = air_response.json()

            if (
                'list' in air_data
                and len(air_data['list']) > 0
            ):

                aqi_number = air_data['list'][0]['main']['aqi']

                aqi_map = {
                    1: "Good",
                    2: "Fair",
                    3: "Moderate",
                    4: "Poor",
                    5: "Very Poor"
                }

                aqi_text = aqi_map.get(
                    aqi_number,
                    "Good"
                )

    except requests.RequestException:

        aqi_text = "Unavailable"

    # ------------------------------------------------------
    # FORECAST
    # ------------------------------------------------------

    forecast_url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={OPENWEATHER_API_KEY}"
        "&units=metric"
    )

    try:

        forecast_response = requests.get(
            forecast_url,
            timeout=15
        )

    except requests.RequestException:

        forecast_response = None

    daily_forecasts = []
    hourly_rain_timeline = []
    chance_of_rain = 0

    # ------------------------------------------------------
    # CHATBOT WEATHER CONTEXT
    # ------------------------------------------------------

    chatbot_context = (
        f"Current Weather in {resolved_city_name}: "
        f"{temp}°C, {description}. "
        f"Feels like {feels_like}°C. "
        f"Humidity: {humidity}%. "
        f"Wind: {round(wind_speed * 3.6)} km/h "
        f"from {wind_dir}. "
        f"Pressure: {pressure} hPa. "
        f"Air Quality: {aqi_text}.\n"
        f"Hourly/Upcoming forecast timeline:"
    )

    if (
        forecast_response
        and forecast_response.status_code == 200
    ):

        forecast_data = forecast_response.json()

        forecast_list = forecast_data.get(
            'list',
            []
        )

        if forecast_list:

            chance_of_rain = int(
                forecast_list[0].get(
                    'pop',
                    0
                ) * 100
            )

        for i, item in enumerate(forecast_list):

            dt_txt = item['dt_txt']

            t = round(
                item['main']['temp']
            )

            desc = item['weather'][0]['main']

            pop = int(
                item.get('pop', 0) * 100
            )

            chatbot_context += (
                f" [{dt_txt} -> "
                f"Temp: {t}°C, "
                f"Condition: {desc}, "
                f"Rain Probability: {pop}%]"
            )

            # Hourly timeline
            if i < 8:

                time_obj = datetime.datetime.strptime(
                    dt_txt,
                    '%Y-%m-%d %H:%M:%S'
                )

                hourly_rain_timeline.append({
                    "time": time_obj.strftime('%I %p'),
                    "pop": pop,
                    "desc": desc
                })

            # Daily forecast
            if '12:00:00' in dt_txt:

                date_obj = datetime.datetime.strptime(
                    dt_txt.split(' ')[0],
                    '%Y-%m-%d'
                )

                daily_forecasts.append({

                    "date": date_obj.strftime('%d %b'),

                    "day": date_obj.strftime('%a'),

                    "temp": t,

                    "min_temp": round(
                        item['main']['temp'] - 3
                    ),

                    "description": desc,

                    # Probability of precipitation for this forecast point
                    "pop": pop
                })

    # ======================================================
    # IMPORTANT:
    # NO GEMINI CALL HERE.
    #
    # This means opening/reloading the weather dashboard
    # does NOT consume a Gemini API request.
    # ======================================================

    (
        insight_text,
        travel_text,
        activities
    ) = create_weather_insights(
        resolved_city_name,
        temp,
        feels_like,
        description,
        humidity,
        wind_speed,
        aqi_text,
        chance_of_rain
    )

    # ------------------------------------------------------
    # RETURN WEATHER DATA
    # ------------------------------------------------------

    return jsonify({

        "city": resolved_city_name,

        "latitude": float(lat),

        "longitude": float(lon),

        "temperature": temp,

        "feels_like": feels_like,

        "description": description,

        "humidity": humidity,

        "wind_speed": wind_speed,

        "pressure": pressure,

        "wind_dir": wind_dir,

        "sunrise": sunrise_time,

        "sunset": sunset_time,

        "aqi": aqi_text,

        "chance_of_rain": f"{chance_of_rain}%",

        "hourly_rain": hourly_rain_timeline,

        "ai_summary": insight_text,

        "travel_advice": travel_text,

        "activities": activities,

        "forecast": daily_forecasts,

        "chatbot_context": chatbot_context
    })


# ==========================================================
# RENDER STARTUP
# ==========================================================

if __name__ == '__main__':

    port = int(
        os.getenv(
            "PORT",
            5000
        )
    )

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
