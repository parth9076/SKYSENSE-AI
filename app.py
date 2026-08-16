from flask import Flask, request, jsonify, render_template
import requests
import os
import datetime
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ==========================================================
# CONFIGURATION
# ==========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY is not configured.")

# Gemini is now used ONLY by the chatbot.
# /api/weather does NOT call Gemini anymore.
client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-3.5-flash"


# ==========================================================
# GEMINI HELPER
# ==========================================================

def generate_ai_text(prompt):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response or not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()


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
            f"Gemini Chat Error: {type(e).__name__}: {e}",
            flush=True
        )

        # Specific response for quota exhaustion
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):

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

                    "description": desc
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
