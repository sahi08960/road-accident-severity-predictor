import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(page_title="Road Accident Severity Prediction", page_icon="🚗", layout="wide")

# --- Load Model and Encoders ---
@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load("xgb_accident_model.pkl")
        le_light = joblib.load("le_light.pkl")
        le_weather = joblib.load("le_weather.pkl")
        return model, le_light, le_weather
    except FileNotFoundError:
        st.error("❌ Model or encoders missing!")
        return None, None, None

model, le_light, le_weather = load_model_and_encoders()

# --- UI ---
st.title("🚦 Road Accident Severity Prediction")
st.markdown("Predict accident severity based on vehicles, light, and weather.")

if model is not None:
    scenario = st.sidebar.selectbox(
        "Select Test Scenario",
        ["Custom", "Low Risk", "Medium Risk", "High Risk"]
    )

    if scenario == "Custom":
        num_vehicles = st.sidebar.number_input("Number of Vehicles", 1, 15, 2)
        light_conditions = st.sidebar.selectbox("Light Conditions", options=le_light.classes_)
        weather_conditions = st.sidebar.selectbox("Weather Conditions", options=le_weather.classes_)
    elif scenario == "Low Risk":
        num_vehicles = 1
        light_conditions = "Daylight"
        weather_conditions = "Fine"
    elif scenario == "Medium Risk":
        num_vehicles = 3
        light_conditions = "Darkness – lights lit"
        weather_conditions = "Rain"
    else:  # High Risk
        num_vehicles = 4
        light_conditions = "Darkness – no lighting"
        weather_conditions = "Fog"

    if st.sidebar.button("Predict Severity"):
        feature_columns = model.get_booster().feature_names
        input_data = {col: [0] for col in feature_columns}

        # Map inputs
        if "number_of_vehicles" in input_data:
            input_data["number_of_vehicles"] = [num_vehicles]
        if "light_conditions" in input_data:
            input_data["light_conditions"] = [le_light.transform([light_conditions])[0]]
        if "weather_conditions" in input_data:
            input_data["weather_conditions"] = [le_weather.transform([weather_conditions])[0]]

        input_df = pd.DataFrame(input_data)[feature_columns].astype(float)

        # Predict
        pred_index = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        severity_labels = {0: "Slight", 1: "Serious", 2: "Fatal"}
        predicted_severity = severity_labels[pred_index]

        # Display
        st.subheader("Prediction Result")
        if predicted_severity == "Fatal":
            st.error(f"🚨 Predicted Severity: **{predicted_severity}** ({pred_proba[pred_index]:.2%})")
        elif predicted_severity == "Serious":
            st.warning(f"⚠️ Predicted Severity: **{predicted_severity}** ({pred_proba[pred_index]:.2%})")
        else:
            st.success(f"✅ Predicted Severity: **{predicted_severity}** ({pred_proba[pred_index]:.2%})")
