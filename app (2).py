# ==============================================================================
#                      app.py - Streamlit Accident Severity App
# ==============================================================================

import streamlit as st
import pandas as pd
import joblib
import zipfile
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CACHED FUNCTIONS ---
@st.cache_resource
def load_model():
    """Load pre-trained XGBoost model."""
    try:
        model = joblib.load("xgb_accident_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file missing! Upload 'xgb_accident_model.pkl' to repo.")
        return None

@st.cache_data
def load_and_prep_data():
    """Load and prepare dataset (used only for defaults)."""
    zip_path = "archive (4).zip"
    extract_path = "dataset"
    csv_filename = "AccidentsBig.csv"
    csv_filepath = os.path.join(extract_path, csv_filename)

    try:
        if not os.path.exists(csv_filepath):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

        df = pd.read_csv(csv_filepath, low_memory=False)
        df.dropna(subset=["Accident_Severity", "latitude", "longitude"], inplace=True)

        # Only keep useful columns for our app
        keep_cols = ["Light_Conditions", "Weather_Conditions",
                     "Number_of_Vehicles", "Number_of_Casualties", "Time"]
        df = df[keep_cols]

        # Extract hour from Time
        df["Time_dt"] = pd.to_datetime(df["Time"], errors="coerce", format="%H:%M")
        df["Hour"] = df["Time_dt"].dt.hour.fillna(12).astype(int)

        return df
    except FileNotFoundError:
        st.error("❌ Dataset file missing! Upload 'archive (4).zip' to repo.")
        return pd.DataFrame()

# --- 3. LOAD MODEL + DATA ---
model = load_model()
df = load_and_prep_data()

# --- 4. USER INTERFACE ---
st.title("🚦 Road Accident Severity Prediction")
st.markdown("Adjust the inputs to simulate an accident and predict severity.")

if model is not None and not df.empty:
    # Sidebar Inputs
    st.sidebar.header("🔮 Accident Scenario")
    hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 12)
    light_conditions = st.sidebar.selectbox("Light Conditions", options=sorted(df["Light_Conditions"].dropna().unique()))
    weather_conditions = st.sidebar.selectbox("Weather Conditions", options=sorted(df["Weather_Conditions"].dropna().unique()))
    num_vehicles = st.sidebar.number_input("Number of Vehicles", 1, 15, 2)
    num_casualties = st.sidebar.number_input("Number of Casualties", 1, 20, 1)

    # Prediction Button
    if st.sidebar.button("Predict Severity", type="primary", use_container_width=True):
        feature_columns_in_order = model.get_booster().feature_names

        # Build input data with defaults
        input_data = {col: [0] for col in feature_columns_in_order}
        if "hour" in feature_columns_in_order: input_data["hour"] = [hour]
        if "light_conditions" in feature_columns_in_order: input_data["light_conditions"] = [light_conditions]
        if "weather_conditions" in feature_columns_in_order: input_data["weather_conditions"] = [weather_conditions]
        if "number_of_vehicles" in feature_columns_in_order: input_data["number_of_vehicles"] = [num_vehicles]
        if "number_of_casualties" in feature_columns_in_order: input_data["number_of_casualties"] = [num_casualties]

        # DataFrame in correct order
        input_df = pd.DataFrame(input_data)[feature_columns_in_order]

        # 🔹 Encode categorical columns
        for col in input_df.select_dtypes(include=["object"]).columns:
            input_df[col] = input_df[col].astype("category").cat.codes

        # 🔹 Ensure numeric
        input_df = input_df.astype(float)

        # Predict
        prediction_index = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        severity_labels = {0: "Slight", 1: "Serious", 2: "Fatal"}
        predicted_severity = severity_labels[prediction_index]

        # Show result
        st.subheader("Prediction Result")
        if predicted_severity == "Fatal":
            st.error(f"🚨 Predicted Severity: **{predicted_severity}** (Confidence: {prediction_proba[prediction_index]:.2%})")
        elif predicted_severity == "Serious":
            st.warning(f"⚠️ Predicted Severity: **{predicted_severity}** (Confidence: {prediction_proba[prediction_index]:.2%})")
        else:
            st.success(f"✅ Predicted Severity: **{predicted_severity}** (Confidence: {prediction_proba[prediction_index]:.2%})")
else:
    st.error("⚠️ App could not load model or dataset. Please check files in repo.")
