from flask import Flask, request, jsonify, render_template
import requests
import os
import datetime
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Google GenAI SDK (current production SDK)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-3.5-flash"


def generate_ai_text(prompt):
    """Generate text with Gemini and return plain text."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    if not response or not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text.strip()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    weather_context = data.get('context') or 'No weather context is currently available.'

    if not user_message:
        return jsonify({"reply": "Please enter a weather question."}), 400

    prompt = f"""You are SkySense AI, an expert meteorological chatbot.
Use the following weather data context to answer the user's question accurately.
If the user asks "What should I do today?", provide personalized, highly practical
outdoor and indoor recommendations based on the current weather conditions,
air quality, and rain chances.

Keep your answer conversational, friendly, and practical (1 to 3 sentences maximum).

Weather Context:
{weather_context}

User Question:
{user_message}
"""

    try:
        reply = generate_ai_text(prompt)
        return jsonify({"reply": reply})
    except Exception as e:
        # Keep the user-facing message friendly, but log the real error in Render.
        print(f"Gemini chat error: {type(e).__name__}: {e}", flush=True)
        return jsonify({
            "reply": "I'm having trouble connecting to my neural network right now. Please try again!"
        }), 500


@app.route('/api/weather', methods=['GET'])
def weather():
    query = request.args.get('city', 'Pune').strip()
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    weather_api_key = os.getenv("OPENWEATHER_API_KEY")

    if not weather_api_key:
        return jsonify({"error": "OPENWEATHER_API_KEY is not configured on the server."}), 500

    resolved_city_name = query

    if not lat or not lon:
        geo_url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(query)}&format=json&limit=1"
        headers = {'User-Agent': 'SkySenseAI-WeatherApp'}
        geo_res = requests.get(geo_url, headers=headers, timeout=15)

        if geo_res.status_code == 200 and len(geo_res.json()) > 0:
            place = geo_res.json()[0]
            lat = place['lat']
            lon = place['lon']
            parts = place['display_name'].split(',')
            resolved_city_name = f"{parts[0].strip()}, {parts[1].strip()}" if len(parts) > 1 else parts[0].strip()
        else:
            return jsonify({"error": "Location not found. Please check your spelling or try a nearby major city/PIN code."}), 400

    current_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={weather_api_key}&units=metric"
    current_response = requests.get(current_url, timeout=15)

    if current_response.status_code != 200:
        return jsonify({"error": "Could not retrieve weather telemetry for this coordinate."}), 400

    data = current_response.json()
    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    description = data['weather'][0]['description']
    humidity = data['main']['humidity']
    wind_speed = data['wind']['speed']
    pressure = data['main']['pressure']

    wind_deg = data['wind'].get('deg', 0)
    dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    wind_dir = dirs[int((wind_deg / 42.5) + 0.5) % 8]

    timezone_shift = data.get('timezone', 0)
    sunrise_time = datetime.datetime.fromtimestamp(
        data['sys']['sunrise'] + timezone_shift,
        datetime.timezone.utc
    ).strftime('%I:%M %p')
    sunset_time = datetime.datetime.fromtimestamp(
        data['sys']['sunset'] + timezone_shift,
        datetime.timezone.utc
    ).strftime('%I:%M %p')

    aqi_text = "Good"
    air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={weather_api_key}"
    air_response = requests.get(air_url, timeout=15)
    if air_response.status_code == 200:
        air_data = air_response.json()
        if 'list' in air_data and len(air_data['list']) > 0:
            aqi_number = air_data['list'][0]['main']['aqi']
            aqi_map = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
            aqi_text = aqi_map.get(aqi_number, "Good")

    forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={weather_api_key}&units=metric"
    forecast_response = requests.get(forecast_url, timeout=15)
    daily_forecasts = []
    hourly_rain_timeline = []
    chance_of_rain = 0

    chatbot_context = f"Current Weather in {resolved_city_name}: {temp}°C, {description}. Humidity: {humidity}%. AQI: {aqi_text}.\nHourly/Upcoming forecast timeline:"

    if forecast_response.status_code == 200:
        forecast_data = forecast_response.json()

        if 'list' in forecast_data and len(forecast_data['list']) > 0:
            chance_of_rain = int(forecast_data['list'][0].get('pop', 0) * 100)

        for i, item in enumerate(forecast_data.get('list', [])):
            dt_txt = item['dt_txt']
            t = round(item['main']['temp'])
            desc = item['weather'][0]['main']
            pop = int(item.get('pop', 0) * 100)

            chatbot_context += f" [{dt_txt} -> Temp: {t}°C, Condition: {desc}, Rain Probability: {pop}%]"

            if i < 8:
                time_obj = datetime.datetime.strptime(dt_txt, '%Y-%m-%d %H:%M:%S')
                hourly_rain_timeline.append({
                    "time": time_obj.strftime('%I %p'),
                    "pop": pop,
                    "desc": desc
                })

            if '12:00:00' in dt_txt:
                date_obj = datetime.datetime.strptime(dt_txt.split(' ')[0], '%Y-%m-%d')
                daily_forecasts.append({
                    "date": date_obj.strftime('%d %b'),
                    "day": date_obj.strftime('%a'),
                    "temp": t,
                    "min_temp": round(item['main']['temp'] - 3),
                    "description": desc
                })

    prompt = f"""You are an expert AI meteorologist. The current weather in {resolved_city_name} is {temp}°C with {description}, humidity at {humidity}%, and chance of rain at {chance_of_rain}%.
Respond EXACTLY in this format line by line:
INSIGHT: [Write a 2-sentence insight predicting conditions for the evening]
TRAVEL: [Write 1 sentence of practical travel and packing advice for someone visiting today]
CYCLING: [Score out of 10, e.g. 9/10]
RUNNING: [Score out of 10, e.g. 7/10]
HIKING: [Score out of 10, e.g. 5/10]
PHOTOGRAPHY: [Score out of 10, e.g. 8/10]
MOTORCYCLING: [Score out of 10, e.g. 8/10]"""

    insight_text = f"Conditions in {resolved_city_name} are currently {description}."
    travel_text = "Standard weather conditions apply for travel."
    activities = {
        "cycling": "--/10",
        "running": "--/10",
        "hiking": "--/10",
        "photography": "--/10",
        "motorcycling": "--/10"
    }

    try:
        response_text = generate_ai_text(prompt)

        for line in response_text.split('\n'):
            line = line.strip().replace('**', '')
            if line.startswith('INSIGHT:'):
                insight_text = line.replace('INSIGHT:', '').strip()
            elif line.startswith('TRAVEL:'):
                travel_text = line.replace('TRAVEL:', '').strip()
            elif line.startswith('CYCLING:'):
                activities['cycling'] = line.replace('CYCLING:', '').strip()
            elif line.startswith('RUNNING:'):
                activities['running'] = line.replace('RUNNING:', '').strip()
            elif line.startswith('HIKING:'):
                activities['hiking'] = line.replace('HIKING:', '').strip()
            elif line.startswith('PHOTOGRAPHY:'):
                activities['photography'] = line.replace('PHOTOGRAPHY:', '').strip()
            elif line.startswith('MOTORCYCLING:'):
                activities['motorcycling'] = line.replace('MOTORCYCLING:', '').strip()

    except Exception as e:
        print(f"Gemini weather insight error: {type(e).__name__}: {e}", flush=True)

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
