from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model/crop_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get inputs
    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # Crop Prediction
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop = model.predict(features)[0]

    # 🌿 Fertilizer Recommendation
    if N < 50:
        fertilizer = "Apply Nitrogen fertilizer (Urea)"
    elif P < 30:
        fertilizer = "Apply Phosphorus fertilizer (DAP)"
    elif K < 30:
        fertilizer = "Apply Potassium fertilizer (MOP)"
    else:
        fertilizer = "Soil nutrients are sufficient"

    # 🌾 Yield Estimation (simple logic)
    yield_estimate = round((rainfall * 0.02 + temperature * 0.1), 2)

    # 💧 Irrigation Recommendation
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
        yield_estimate=yield_estimate,
        irrigation=irrigation
    )

if __name__ == "__main__":
    app.run()
