# SkySense AI  ☀️

Production-ready AI weather intelligence dashboard featuring real-time telemetry, advanced SkyScore™ analytics, interactive weather timelines, pinned locations, and an embedded LLM assistant powered by Groq.

![SkySense AI Banner](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Python Flask](https://img.shields.io/badge/Backend-Python%2C%20Flask-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Key Features

* **High-Precision Live Telemetry:** Fetches real-time weather stats, air quality (AQI), pressure, wind direction, and sunrise/sunset timings based on precise GPS coordinates or search queries.
* **Pinned Locations (Quick-Access Chips):** Save up to 4 favorite cities directly to your dashboard via `localStorage` for instant one-tap switching.
* **SkyScore™ Intelligence Engine:** Dynamically calculates an overall weather operating score alongside custom sub-scores for cycling, running, hiking, motorcycling, photography, and outdoor dining.
* **Interactive AI Weather Assistant:** Embedded chat interface that answers contextual questions about your local forecast using live environmental data.
* **Robust Offline & Network Handling:** Gracefully falls back to cached weather data and alerts users when connection is lost.
* **GPU-Accelerated Motion System:** Custom 60fps CSS transitions, staggered cascading card reveals, and dynamic trend charts using Chart.js.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, Modern CSS3 (Grid/Flexbox, Custom Properties), Vanilla JavaScript, Chart.js.
* **Backend:** Python, Flask, Groq API (LLM integration).
* **APIs & Geocoding:** OpenWeather API, OpenStreetMap Nominatim.
* **Deployment:** Render (`Procfile`).

---

## 📁 Project Structure

```text
SKYSENSE-AI/
├── templates/
│   └── index.html      # Complete frontend UI, CSS, and client-side logic
├── app.py              # Flask backend server & AI intelligence routing
├── requirements.txt    # Python dependencies
├── Procfile            # Deployment configuration for Render
└── README.md           # Project documentation



⚙️ Local Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/parth9076/SKYSENSE-AI.git](https://github.com/parth9076/SKYSENSE-AI.git)
cd SKYSENSE-AI
Install dependencies:

Bash
pip install -r requirements.txt
Set up environment variables:
Create a .env file or export your API keys in your environment:

Bash
export OPENWEATHER_API_KEY="your_openweather_api_key"
export GROQ_API_KEY="your_groq_api_key"
Run the Flask server:

Bash
python app.py
Open in your browser:
Navigate to http://127.0.0.1:5000

📄 License
Distributed under the MIT License. See LICENSE for more information.
