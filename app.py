from flask import Flask, render_template, request
import joblib
import numpy as np
import requests

app = Flask(__name__)

# ---------------- LOAD ML MODEL ----------------
model = joblib.load("model/crop_model.pkl")

# ⚠️ In production, store API key as environment variable
WEATHER_API_KEY = "72d6c804ff6841bddd44e4f1eecc2e90"

# Store crop history (in-memory)
crop_history = {}

# ---------------- WEATHER (CURRENT) ----------------
def get_weather(city):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        # Safety check
        if response.status_code != 200 or "main" not in data:
            return None

        return {
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"]
        }

    except Exception:
        return None


# ---------------- WEATHER (5-DAY FORECAST) ----------------
def get_forecast(city):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code != 200 or "list" not in data:
            return [], []

        days, temps = [], []
        for item in data["list"]:
            if "12:00:00" in item["dt_txt"]:
                days.append(item["dt_txt"].split(" ")[0])
                temps.append(item["main"]["temp"])

        return days[:5], temps[:5]

    except Exception:
        return [], []


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Form inputs
        city = request.form["city"]
        area = float(request.form["area"])

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Weather data
        weather = get_weather(city)
        if weather is None:
            return render_template(
                "index.html",
                error="❌ City not found or Weather API error"
            )

        temperature = weather["temp"]
        humidity = weather["humidity"]
        lat = weather["lat"]
        lon = weather["lon"]

        # Forecast
        days, forecast_temps = get_forecast(city)

        # ML prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        crop = model.predict(features)[0]

        # Extra insights
        fertilizer = "NPK 20-20-20"
        fert_amount = round(area * 50, 2)  # kg
        irrigation = "Moderate irrigation recommended"
        yield_estimate = round(area * 2.5, 2)  # tons

        # Crop history
        crop_history[crop] = crop_history.get(crop, 0) + 1

        return render_template(
            "index.html",
            crop=crop,
            fertilizer=fertilizer,
            fert_amount=fert_amount,
            irrigation=irrigation,
            yield_estimate=yield_estimate,
            temperature=temperature,
            humidity=humidity,
            N=N, P=P, K=K,
            days=days,
            forecast_temps=forecast_temps,
            lat=lat,
            lon=lon,
            crop_labels=list(crop_history.keys()),
            crop_counts=list(crop_history.values())
        )

    except Exception as e:
        return render_template(
            "index.html",
            error="❌ Invalid input or server error"
        )


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
