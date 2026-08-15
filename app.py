from flask import Flask, render_template, request, jsonify
import os
import requests

app = Flask(__name__)

# Retrieve API keys from environment variables
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city')
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    
    if not OPENWEATHER_API_KEY:
        return jsonify({"error": "OpenWeather API key not configured on server."}), 500

    # Build OpenWeatherMap URL based on coordinates or city name
    if lat and lon:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    elif city:
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    else:
        # Default fallback to Pune
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?q=Pune&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q=Pune&appid={OPENWEATHER_API_KEY}&units=metric"

    try:
        # Fetch current weather
        w_res = requests.get(weather_url)
        w_data = w_res.json()
        if w_res.status_code != 200:
            return jsonify({"error": w_data.get("message", "Failed to fetch weather data.")}), 400

        # Fetch forecast data
        f_res = requests.get(forecast_url)
        f_data = f_res.json()
        forecast_list = f_data.get('list', [])

        # Format hourly/forecast slots safely
        formatted_forecast = []
        for item in forecast_list[:4]:
            # Extract time slot (e.g., "15:00") from dt_txt
            dt_txt = item.get('dt_txt', '00:00:00')
            time_str = dt_txt.split(' ')[1][:5] if ' ' in dt_txt else 'Now'
            pop_val = int(item.get('pop', 0) * 100)  # Convert decimal to percentage
            formatted_forecast.append({
                "time": time_str,
                "pop": pop_val
            })

        # Generate mock or basic AI insight if Gemini is integrated
        ai_insight = f"Conditions in {w_data.get('name', 'location')} are currently {w_data['weather'][0]['description']}. Temperature is comfortable at {round(w_data['main']['temp'])}°C."

        return jsonify({
            "city": str(w_data.get('name', 'Unknown')),
            "temp": float(w_data.get('main', {}).get('temp', 0)),
            "description": str(w_data.get('weather', [{}])[0].get('description', 'N/A')),
            "humidity": int(w_data.get('main', {}).get('humidity', 0)),
            "wind_speed": float(w_data.get('wind', {}).get('speed', 0)),
            "ai_insight": ai_insight,
            "forecast": formatted_forecast
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    context = data.get('context', '')
    
    # Simple response handler or Gemini integration hook
    reply = f"Based on the current conditions ({context}), I recommend dressing comfortably and staying hydrated!"
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
