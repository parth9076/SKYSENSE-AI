from flask import Flask, request, jsonify, render_template
import requests
import os
import datetime
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ==========================================================
# GEMINI AI CONFIGURATION
# ==========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-3.5-flash"


def generate_ai_text(prompt):
    """
    Send a prompt to Gemini and return the generated text.
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response or not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route('/')
def home():
    return render_template('index.html')


# ==========================================================
# AI CHATBOT
# ==========================================================

@app.route('/api/chat', methods=['POST'])
def chat():

    data = request.get_json(silent=True) or {}

    user_message = (data.get('message') or '').strip()

    weather_context = data.get('context') or (
        "No current weather information is available."
    )

    # Empty message protection
    if not user_message:
        return jsonify({
            "reply": "Please enter a question."
        }), 400

    # ======================================================
    # UNIVERSAL AI CHATBOT PROMPT
    # ======================================================

    prompt = f"""
You are SkySense AI, an intelligent, friendly and helpful AI assistant.

You are NOT restricted to predefined questions.

The user can ask you ANY question.

You can answer:

- Weather questions
- Weather forecast questions
- Temperature questions
- Rain questions
- Humidity questions
- Wind questions
- Air quality questions
- Travel questions
- Outdoor activity questions
- Clothing recommendations
- Cycling questions
- Running questions
- Hiking questions
- Motorcycle riding questions
- Photography questions
- Weather science questions
- General knowledge questions
- Programming questions
- Computer science questions
- Cloud computing questions
- AI and machine learning questions
- Study-related questions
- Technology questions
- Everyday questions
- General conversation

IMPORTANT:

If the question is related to the weather, use the live weather
context provided below.

If the question is NOT related to weather, answer it normally using
your general knowledge.

Do NOT say that the user can only ask weather questions.

Do NOT restrict yourself to the predefined suggestion buttons
shown in the user interface.

Answer naturally like a helpful AI assistant.

For simple questions, give a concise answer.

For questions that require explanation, provide enough detail to
make the answer easy to understand.

If the user asks for a comparison, explain the important differences.

If the user asks for instructions, provide clear step-by-step guidance.

If the user asks about programming, provide useful code when appropriate.

If the user asks about weather conditions, make your answer specific
to the current location and weather context when possible.

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

        return jsonify({
            "reply": (
                "I'm having trouble connecting to my AI network "
                "right now. Please try again."
            )
        }), 500


# ==========================================================
# WEATHER API
# ==========================================================

@app.route('/api/weather', methods=['GET'])
def weather():

    query = request.args.get('city', 'Pune').strip()

    lat = request.args.get('lat')
    lon = request.args.get('lon')

    weather_api_key = os.getenv("OPENWEATHER_API_KEY")

    if not weather_api_key:
        return jsonify({
            "error": "OPENWEATHER_API_KEY is not configured on the server."
        }), 500

    resolved_city_name = query

    # ======================================================
    # LOCATION SEARCH
    # ======================================================

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
                    f"{parts[0].strip()}, "
                    f"{parts[1].strip()}"
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

    # ======================================================
    # CURRENT WEATHER
    # ======================================================

    current_url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={weather_api_key}"
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

    # ======================================================
    # CURRENT WEATHER VALUES
    # ======================================================

    temp = data['main']['temp']

    feels_like = data['main']['feels_like']

    description = data['weather'][0]['description']

    humidity = data['main']['humidity']

    wind_speed = data['wind']['speed']

    pressure = data['main']['pressure']

    # ======================================================
    # WIND DIRECTION
    # ======================================================

    wind_deg = data['wind'].get('deg', 0)

    dirs = [
        'N',
        'NE',
        'E',
        'SE',
        'S',
        'SW',
        'W',
        'NW'
    ]

    wind_dir = dirs[
        int((wind_deg / 42.5) + 0.5) % 8
    ]

    # ======================================================
    # SUNRISE / SUNSET
    # ======================================================

    timezone_shift = data.get('timezone', 0)

    sunrise_time = datetime.datetime.fromtimestamp(
        data['sys']['sunrise'] + timezone_shift,
        datetime.timezone.utc
    ).strftime('%I:%M %p')

    sunset_time = datetime.datetime.fromtimestamp(
        data['sys']['sunset'] + timezone_shift,
        datetime.timezone.utc
    ).strftime('%I:%M %p')

    # ======================================================
    # AIR QUALITY
    # ======================================================

    aqi_text = "Good"

    air_url = (
        "http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={weather_api_key}"
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

                aqi_number = air_data[
                    'list'
                ][0]['main']['aqi']

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

    # ======================================================
    # FORECAST
    # ======================================================

    forecast_url = (
        "http://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&appid={weather_api_key}"
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

    # ======================================================
    # CHATBOT WEATHER CONTEXT
    # ======================================================

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

    # ======================================================
    # PROCESS FORECAST
    # ======================================================

    if (
        forecast_response
        and forecast_response.status_code == 200
    ):

        forecast_data = forecast_response.json()

        forecast_list = forecast_data.get(
            'list',
            []
        )

        if len(forecast_list) > 0:

            chance_of_rain = int(
                forecast_list[0].get(
                    'pop',
                    0
                ) * 100
            )

        for i, item in enumerate(
            forecast_list
        ):

            dt_txt = item['dt_txt']

            t = round(
                item['main']['temp']
            )

            desc = item[
                'weather'
            ][0]['main']

            pop = int(
                item.get(
                    'pop',
                    0
                ) * 100
            )

            chatbot_context += (
                f" [{dt_txt} -> "
                f"Temp: {t}°C, "
                f"Condition: {desc}, "
                f"Rain Probability: {pop}%]"
            )

            # ==================================================
            # HOURLY RAIN TIMELINE
            # ==================================================

            if i < 8:

                time_obj = datetime.datetime.strptime(
                    dt_txt,
                    '%Y-%m-%d %H:%M:%S'
                )

                hourly_rain_timeline.append({
                    "time": time_obj.strftime(
                        '%I %p'
                    ),
                    "pop": pop,
                    "desc": desc
                })

            # ==================================================
            # DAILY FORECAST
            # ==================================================

            if '12:00:00' in dt_txt:

                date_obj = datetime.datetime.strptime(
                    dt_txt.split(' ')[0],
                    '%Y-%m-%d'
                )

                daily_forecasts.append({

                    "date": date_obj.strftime(
                        '%d %b'
                    ),

                    "day": date_obj.strftime(
                        '%a'
                    ),

                    "temp": t,

                    "min_temp": round(
                        item['main']['temp'] - 3
                    ),

                    "description": desc
                })

    # ======================================================
    # AI WEATHER INSIGHTS
    # ======================================================

    insight_prompt = f"""
You are an expert AI meteorologist.

The current weather in {resolved_city_name} is:

Temperature: {temp}°C
Feels Like: {feels_like}°C
Condition: {description}
Humidity: {humidity}%
Wind: {round(wind_speed * 3.6)} km/h
Wind Direction: {wind_dir}
Pressure: {pressure} hPa
Air Quality: {aqi_text}
Chance of Rain: {chance_of_rain}%

Respond EXACTLY in this format:

INSIGHT: [Write a 2-sentence insight predicting conditions for the evening]

TRAVEL: [Write 1 sentence of practical travel and packing advice for someone visiting today]

CYCLING: [Score out of 10, e.g. 9/10]

RUNNING: [Score out of 10, e.g. 7/10]

HIKING: [Score out of 10, e.g. 5/10]

PHOTOGRAPHY: [Score out of 10, e.g. 8/10]

MOTORCYCLING: [Score out of 10, e.g. 8/10]
"""

    # ======================================================
    # DEFAULT VALUES
    # ======================================================

    insight_text = (
        f"Conditions in {resolved_city_name} "
        f"are currently {description}."
    )

    travel_text = (
        "Standard weather conditions apply for travel."
    )

    activities = {

        "cycling": "--/10",

        "running": "--/10",

        "hiking": "--/10",

        "photography": "--/10",

        "motorcycling": "--/10"
    }

    # ======================================================
    # GENERATE WEATHER AI INSIGHTS
    # ======================================================

    try:

        response_text = generate_ai_text(
            insight_prompt
        )

        for line in response_text.split('\n'):

            line = line.strip().replace(
                '**',
                ''
            )

            if line.startswith(
                'INSIGHT:'
            ):

                insight_text = (
                    line.replace(
                        'INSIGHT:',
                        ''
                    ).strip()
                )

            elif line.startswith(
                'TRAVEL:'
            ):

                travel_text = (
                    line.replace(
                        'TRAVEL:',
                        ''
                    ).strip()
                )

            elif line.startswith(
                'CYCLING:'
            ):

                activities[
                    'cycling'
                ] = line.replace(
                    'CYCLING:',
                    ''
                ).strip()

            elif line.startswith(
                'RUNNING:'
            ):

                activities[
                    'running'
                ] = line.replace(
                    'RUNNING:',
                    ''
                ).strip()

            elif line.startswith(
                'HIKING:'
            ):

                activities[
                    'hiking'
                ] = line.replace(
                    'HIKING:',
                    ''
                ).strip()

            elif line.startswith(
                'PHOTOGRAPHY:'
            ):

                activities[
                    'photography'
                ] = line.replace(
                    'PHOTOGRAPHY:',
                    ''
                ).strip()

            elif line.startswith(
                'MOTORCYCLING:'
            ):

                activities[
                    'motorcycling'
                ] = line.replace(
                    'MOTORCYCLING:',
                    ''
                ).strip()

    except Exception as e:

        print(
            f"Gemini Weather Error: "
            f"{type(e).__name__}: {e}",
            flush=True
        )

    # ======================================================
    # RETURN WEATHER DATA
    # ======================================================

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
# START SERVER
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
