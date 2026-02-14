from flask import Flask, render_template, request
import joblib
import numpy as np
import requests

app = Flask(__name__)

# Load ML model
model = joblib.load("model/crop_model.pkl")

WEATHER_API_KEY = "72d6c804ff6841bddd44e4f1eecc2e90"

# ---------------- GET USER LOCATION (IP BASED) ----------------
def get_location():
    try:
        res = requests.get("https://ipapi.co/json/", timeout=5).json()
        return res["latitude"], res["longitude"]
    except:
        return None, None


# ---------------- GET LIVE WEATHER ----------------
def get_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }

    try:
        res = requests.get(url, params=params, timeout=5).json()
        return res["main"]["temp"], res["main"]["humidity"]
    except:
        return None, None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Location
        lat, lon = get_location()
        if lat is None:
            raise Exception("Location error")

        # Weather
        temperature, humidity = get_weather(lat, lon)
        if temperature is None:
            raise Exception("Weather error")

        # ML Prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        crop = model.predict(features)[0]

        # Extra insights
        fertilizer = "Nitrogen (Urea)"
        fert_amount = round(area * 50, 2)
        irrigation = "Low irrigation required"
        yield_estimate = round(area * 2.5, 2)

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
            lat=lat,
            lon=lon
        )

    except:
        return render_template(
            "index.html",
            error="❌ Unable to fetch live weather. Please try again."
        )


if __name__ == "__main__":
    app.run(debug=True)
