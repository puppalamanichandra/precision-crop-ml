from flask import Flask, render_template, request
import joblib
import numpy as np
import requests

app = Flask(__name__)

# Load ML model
model = joblib.load("model/crop_model.pkl")

WEATHER_API_KEY = "72d6c804ff6841bddd44e4f1eecc2e90"

# ---------------- LOCATION (SAFE) ----------------
def get_location():
    try:
        res = requests.get("https://ipapi.co/json/", timeout=3).json()
        return res.get("latitude"), res.get("longitude")
    except:
        return None, None


# ---------------- WEATHER (SAFE + FALLBACK) ----------------
def get_weather():
    lat, lon = get_location()

    # If location fails → fallback
    if lat is None or lon is None:
        return 25.0, 65, False   # temp, humidity, live=False

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        data = requests.get(url, params=params, timeout=3).json()
        return data["main"]["temp"], data["main"]["humidity"], True
    except:
        return 25.0, 65, False


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Inputs
    area = float(request.form["area"])
    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # Weather
    temperature, humidity, live = get_weather()

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
        weather_mode="Live Weather" if live else "Estimated Weather",
        N=N, P=P, K=K
    )


if __name__ == "__main__":
    app.run(debug=True)
