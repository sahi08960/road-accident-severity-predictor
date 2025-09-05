# ==============================================================================
#                      app.py - Simple Accident Severity Predictor
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

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgb_accident_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'xgb_accident_model.pkl' is in the repo.")
        return None

# --- 3. LOAD DATA ---
@st.cache_data
def load_and_prep_data():
    zip_path = "archive (4).zip"
    extract_path = "dataset"
    csv_filename = "AccidentsBig.csv"
    csv_filepath = os.path.join(extract_path, csv_filename)

    try:
        if not os.path.exists(csv_filepath):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

        df = pd.read_csv(csv_filepath, low_memory=False)
        df.dropna(subset=['Accident_Severity'], inplace=True)

        return df
    except FileNotFoundError:
        st.error(f"Dataset file '{zip_path}' not found.")
        return pd.DataFrame()

# --- 4. LOAD ASSETS ---
model = load_model()
df = load_and_prep_data()

# --- 5. MAIN USER INTERFACE ---
st.title("🚦 Road Accident Severity Predictor")
st.markdown("A simple ML-powered tool to predict accident severity based on key factors.")

if model is not None and not df.empty:
    st.sidebar.header("🔮 Accident Scenario Input")

    # Input fields
    hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 17)
    weather_conditions = st.sidebar.selectbox("Weather Conditions", options=sorted(df['Weather_Conditions'].dropna().unique()))
    num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 15, 2)
    daylight = st.sidebar.radio("Daylight/Darkness", ["Daylight", "Darkness"])

    # Prediction button
    if st.sidebar.button("Predict Severity", type="primary", use_container_width=True):
        feature_columns_in_order = model.get_booster().feature_names

        # Default input with median/mode values
        input_data = {}
        for col in feature_columns_in_order:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    input_data[col] = [df[col].median()]
                else:
                    input_data[col] = [df[col].mode()[0]]
            else:
                input_data[col] = [0]

        # Update with user inputs
        user_inputs = {
            'Hour': [hour],
            'Weather_Conditions': [weather_conditions],
            'Number_of_Vehicles': [num_vehicles],
            'Light_Conditions': [1 if daylight == "Daylight" else 0]  # encoding
        }
        input_data.update(user_inputs)

        input_df = pd.DataFrame(input_data)[feature_columns_in_order]

        # Predict
        prediction_index = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
        predicted_severity = severity_labels[prediction_index]

        st.subheader("Prediction Result")
        if predicted_severity == 'Fatal':
            st.error(f"Predicted Severity: *{predicted_severity}* (Confidence: {prediction_proba[prediction_index]:.2%})")
        elif predicted_severity == 'Serious':
            st.warning(f"Predicted Severity: *{predicted_severity}* (Confidence: {prediction_proba[prediction_index]:.2%})")
        else:
            st.success(f"Predicted Severity: *{predicted_severity}* (Confidence: {prediction_proba[prediction_index]:.2%})")

else:
    st.error("Could not load model or dataset.")
