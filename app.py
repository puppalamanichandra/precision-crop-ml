from flask import Flask, render_template, request
import joblib
import numpy as np
import requests

app = Flask(__name__)

model = joblib.load("model/crop_model.pkl")

WEATHER_API_KEY = "72d6c804ff6841bddd44e4f1eecc2e90"

crop_history = {}

def get_weather(city):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
    data = requests.get(url).json()
    return (
        data["main"]["temp"],
        data["main"]["humidity"],
        data["coord"]["lat"],
        data["coord"]["lon"]
    )

def get_forecast(city):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
    data = requests.get(url).json()

    days, temps = [], []
    for item in data["list"]:
        if "12:00:00" in item["dt_txt"]:
            days.append(item["dt_txt"].split(" ")[0])
            temps.append(item["main"]["temp"])

    return days[:5], temps[:5]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    city = request.form["city"]
    area = float(request.form["area"])

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    temperature, humidity, lat, lon = get_weather(city)
    days, forecast_temps = get_forecast(city)

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop = model.predict(features)[0]

    fertilizer = "NPK 20-20-20"
    fert_amount = round(area * 50, 2)
    irrigation = "Moderate irrigation recommended"
    yield_estimate = round(area * 2.5, 2)

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
        lat=lat, lon=lon,
        crop_labels=list(crop_history.keys()),
        crop_counts=list(crop_history.values())
    )

if __name__ == "__main__":
    app.run(debug=True)
