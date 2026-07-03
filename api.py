"""
api.py — Flask REST API for the Medical Insurance Charges ANN.

Run locally:
    python api.py
    (serves on http://localhost:5000)

Test:
    curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"age":30,"sex":"male","bmi":25.0,"children":0,"smoker":"no","region":"northeast"}'

Requires these 3 files in the same folder:
    preprocessor.pkl
    y_scaler.pkl
    insurance_ann_model.keras
"""

import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from tensorflow import keras

app = Flask(__name__)

preprocessor = joblib.load("preprocessor.pkl")
y_scaler = joblib.load("y_scaler.pkl")
model = keras.models.load_model("insurance_ann_model.keras")

REQUIRED_FIELDS = ["age", "sex", "bmi", "children", "smoker", "region"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        input_df = pd.DataFrame([{
            "age": float(data["age"]),
            "sex": str(data["sex"]).lower(),
            "bmi": float(data["bmi"]),
            "children": int(data["children"]),
            "smoker": str(data["smoker"]).lower(),
            "region": str(data["region"]).lower(),
        }])
        X_proc = preprocessor.transform(input_df)
        pred_scaled = model.predict(X_proc)
        pred = float(y_scaler.inverse_transform(pred_scaled)[0][0])
        return jsonify({"predicted_charges": round(pred, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
