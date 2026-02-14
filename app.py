from flask import Flask, render_template, request
import joblib
import numpy as np
import requests

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/crop_model.pkl")

WEATHER_API_KEY = "72d6c804ff6841bddd44e4f1eecc2e90"


# ---------------- LOCATION (IP BASED) ----------------
def get_location_from_ip():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5)
        data = res.json()
        lat, lon = data["loc"].split(",")
        return float(lat), float(lon)
    except:
        return None, None


# ---------------- LIVE WEATHER ----------------
def get_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }

    try:
        res = requests.get(url, timeout=5)
        data = res.json()

        if res.status_code != 200 or "main" not in data:
            return None, None

        return data["main"]["temp"], data["main"]["humidity"]

    except:
        return None, None


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------- FORM DATA --------
        area = float(request.form["area"])
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # -------- LOCATION --------
        lat, lon = get_location_from_ip()
        if lat is None:
            return render_template("index.html", error="❌ Location detection failed")

        # -------- WEATHER --------
        temperature, humidity = get_weather(lat, lon)
        if temperature is None:
            return render_template("index.html", error="❌ Weather API error")

        # -------- ML PREDICTION --------
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        crop = model.predict(features)[0]

        # -------- INSIGHTS --------
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
            temperature=round(temperature, 2),
            humidity=round(humidity, 2),
            N=N, P=P, K=K,
            lat=lat,
            lon=lon,

            # 🔽 placeholders to avoid JS errors
            days=[],
            forecast_temps=[],
            crop_labels=[],
            crop_counts=[]
        )

    except Exception:
        return render_template("index.html", error="❌ Invalid input")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
