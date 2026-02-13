from flask import Flask, render_template, request
import joblib
import numpy as np
import requests

app = Flask(__name__)

# Load ML model
model = joblib.load("model/crop_model.pkl")

# 🔑 WEATHER API KEY (PASTE YOUR KEY HERE)
WEATHER_API_KEY = "72d6c804ff6841bddd44e4f1eecc2e90"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # 🔹 User inputs
    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])
    area = float(request.form["area"])  # farm area in hectares

    city = request.form["city"]

    # 🌦 Fetch weather data
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    weather_data = requests.get(weather_url).json()

    temperature = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]

    # 🌾 Crop prediction
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop = model.predict(features)[0]

    # 🌿 Fertilizer recommendation & amount
    if N < 50:
        fertilizer = "Nitrogen (Urea)"
        fert_amount = area * 50   # kg per hectare
    elif P < 30:
        fertilizer = "Phosphorus (DAP)"
        fert_amount = area * 40
    elif K < 30:
        fertilizer = "Potassium (MOP)"
        fert_amount = area * 30
    else:
        fertilizer = "No extra fertilizer needed"
        fert_amount = 0

    # 🌾 Yield estimation
    yield_estimate = round((rainfall * 0.02 + temperature * 0.15) * area, 2)

    # 💧 Irrigation advice
    if rainfall < 50:
        irrigation = "High irrigation required"
    elif rainfall < 100:
        irrigation = "Moderate irrigation required"
    else:
        irrigation = "Low irrigation required"

    return render_template(
        "index.html",
        crop=crop,
        fertilizer=fertilizer,
        fert_amount=fert_amount,
        irrigation=irrigation,
        yield_estimate=yield_estimate,
        temperature=temperature,
        humidity=humidity,
        N=N, P=P, K=K
    )

if __name__ == "__main__":
    app.run()
