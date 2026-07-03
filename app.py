"""
app.py — Streamlit web app for the Medical Insurance Charges ANN.

Run locally:
    streamlit run app.py

Requires these 3 files in the same folder (produced by train.py or your notebooks):
    preprocessor.pkl
    y_scaler.pkl
    insurance_ann_model.keras
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow import keras

st.set_page_config(page_title="Insurance Charge Predictor", page_icon="💰", layout="centered")


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
    model = keras.models.load_model("insurance_ann_model.keras")
    return preprocessor, y_scaler, model


st.title("💰 Medical Insurance Charge Predictor")
st.write("Enter patient details to estimate annual medical insurance charges using a trained ANN.")

try:
    preprocessor, y_scaler, model = load_artifacts()
except Exception as e:
    st.error(
        "Could not load model artifacts. Make sure `preprocessor.pkl`, `y_scaler.pkl`, "
        "and `insurance_ann_model.keras` are in the same folder as this app.\n\n"
        f"Details: {e}"
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Charges", type="primary"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])

    X_proc = preprocessor.transform(input_df)
    pred_scaled = model.predict(X_proc)
    pred = y_scaler.inverse_transform(pred_scaled)[0][0]

    st.success(f"### Estimated annual charges: **${pred:,.2f}**")

    if smoker == "yes":
        st.info("Smoking status is the strongest driver of predicted cost for this model.")

st.caption("Model: Keras ANN (64→32→16→1) trained on the medical insurance dataset.")
