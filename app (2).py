# ==============================================================================
#                      app.py - Streamlit Accident Severity App
# ==============================================================================

import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide"
)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    """Load pre-trained XGBoost model."""
    try:
        model = joblib.load("xgb_accident_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file missing! Upload 'xgb_accident_model.pkl'.")
        return None

model = load_model()

# --- 3. USER INTERFACE ---
st.title("🚦 Road Accident Severity Prediction")
st.markdown("Predict accident severity based on vehicle count, light, and weather conditions.")

if model is not None:
    # Sidebar Inputs
    st.sidebar.header("🔮 Accident Scenario")
    num_vehicles = st.sidebar.number_input("Number of Vehicles", 1, 15, 2)
    light_conditions = st.sidebar.selectbox(
        "Light Conditions", 
        options=["Daylight", "Darkness – lights lit", "Darkness – no lighting", "Darkness – lights unlit"]
    )
    weather_conditions = st.sidebar.selectbox(
        "Weather Conditions", 
        options=["Fine", "Rain", "Snow", "Fog", "Other"]
    )

    # Prediction Button
    if st.sidebar.button("Predict Severity"):
        # Get feature names from model
        feature_columns_in_order = model.get_booster().feature_names

        # Build input dictionary with default 0s
        input_data = {col: [0] for col in feature_columns_in_order}

        # Map our 3 inputs if they exist in model features
        mapping = {
            "number_of_vehicles": num_vehicles,
            "light_conditions": light_conditions,
            "weather_conditions": weather_conditions
        }
        for key, value in mapping.items():
            if key in input_data:
                input_data[key] = [value]

        # Build DataFrame
        input_df = pd.DataFrame(input_data)[feature_columns_in_order]

        # Encode categoricals
        for col in input_df.select_dtypes(include=["object"]).columns:
            input_df[col] = input_df[col].astype("category").cat.codes

        # Ensure numeric
        input_df = input_df.astype(float)

        # Predict
        prediction_index = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        severity_labels = {0: "Slight", 1: "Serious", 2: "Fatal"}
        predicted_severity = severity_labels[prediction_index]

        # Display result
        st.subheader("Prediction Result")
        if predicted_severity == "Fatal":
            st.error(f"🚨 Predicted Severity: **{predicted_severity}** (Confidence: {prediction_proba[prediction_index]:.2%})")
        elif predicted_severity == "Serious":
            st.warning(f"⚠️ Predicted Severity: **{predicted_severity}** (Confidence: {prediction_proba[prediction_index]:.2%})")
        else:
            st.success(f"✅ Predicted Severity: **{predicted_severity}** (Confidence: {prediction_proba[prediction_index]:.2%})")
else:
    st.error("⚠️ Model not loaded. Please check 'xgb_accident_model.pkl'.")
