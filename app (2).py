
# ==============================================================================
#                      app.py - Main Streamlit Application
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import zipfile
import os

# --- 1. PAGE CONFIGURATION ---
# Set the page title, icon, and layout
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CACHED FUNCTIONS FOR PERFORMANCE ---
# Using caching prevents the app from reloading the model and data on every interaction.
@st.cache_resource
def load_model():
    """Load the pre-trained XGBoost model."""
    try:
        model = joblib.load('xgb_accident_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'xgb_accident_model.pkl' is in the repository.")
        return None

@st.cache_data
def load_and_prep_data():
    """Load, clean, and prepare the raw accident data for visualization."""
    zip_path = "archive (4).zip"
    extract_path = "dataset"
    csv_filename = "AccidentsBig.csv"
    csv_filepath = os.path.join(extract_path, csv_filename)

    try:
        if not os.path.exists(csv_filepath):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

        df = pd.read_csv(csv_filepath, low_memory=False)
        df.dropna(subset=['Accident_Severity', 'latitude', 'longitude'], inplace=True)

        # Create user-friendly labels for plotting and filtering
        severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
        df['Severity Label'] = df['Accident_Severity'].map(severity_map)

        df['Time_dt'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M')
        df['Hour'] = df['Time_dt'].dt.hour

        return df
    except FileNotFoundError:
        st.error(f"Dataset file '{zip_path}' not found. Please ensure it is in the repository.")
        return pd.DataFrame()

# --- 3. LOAD ASSETS ---
model = load_model()
df = load_and_prep_data()

# --- 4. MAIN USER INTERFACE ---
st.title("Indian Road Accident Severity: Prediction & Analysis 🚦")
st.markdown("This interactive dashboard uses a machine learning model to predict accident severity and visualize high-risk locations.")

if model is not None and not df.empty:
    # --- Sidebar for User Input ---
    st.sidebar.header("🔮 Simulate an Accident Scenario")
    st.sidebar.markdown("Adjust the features below to predict the severity of an accident.")

    # Create input widgets
    hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 17)
    day_of_week = st.sidebar.selectbox("Day of Week", options=range(1, 8), format_func=lambda x: ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'][x-1], index=4)
    light_conditions = st.sidebar.selectbox("Light Conditions", options=sorted(df['Light_Conditions'].unique()))
    weather_conditions = st.sidebar.selectbox("Weather Conditions", options=sorted(df['Weather_Conditions'].unique()))
    road_surface = st.sidebar.selectbox("Road Surface Conditions", options=sorted(df['Road_Surface_Conditions'].unique()))
    num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 15, 2)
    num_casualties = st.sidebar.number_input("Number of Casualties", 1, 20, 1)

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Severity", type="primary", use_container_width=True):
        # The order of columns MUST match the order used during model training
        # We use median values from the original dataset for features not in the UI
        feature_columns_in_order = model.get_booster().feature_names

        input_data = {col: [df[col].median()] for col in feature_columns_in_order}

        # Update with user inputs
        user_inputs = {
            'Hour': [hour], 'Day_of_Week': [day_of_week], 'Light_Conditions': [light_conditions],
            'Weather_Conditions': [weather_conditions], 'Road_Surface_Conditions': [road_surface],
            'Number_of_Vehicles': [num_vehicles], 'Number_of_Casualties': [num_casualties]
        }
        input_data.update(user_inputs)

        input_df = pd.DataFrame(input_data)[feature_columns_in_order] # Ensure correct order

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

    # --- Interactive Map Visualization ---
    st.subheader("🗺 Interactive Map of Accident Hotspots")
    selected_severity_map = st.selectbox(
        "Select Severity to Visualize on the Map:",
        options=sorted(df['Severity Label'].unique())
    )

    map_data = df[df['Severity Label'] == selected_severity_map]

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/dark-v9',
        initial_view_state=pdk.ViewState(
            latitude=20.5937, longitude=78.9629, zoom=4, pitch=50,
        ),
        layers=[
            pdk.Layer(
               'HeatmapLayer',
               data=map_data,
               get_position='[longitude, latitude]',
               radius=100,
               elevation_scale=4,
               elevation_range=[0, 1000],
               pickable=True,
               extruded=True,
            ),
        ],
        tooltip={"html": "<b>Severity:</b> {Severity Label}<br/><b>Hour:</b> {Hour}:00"}
    ))
else:
    st.error("Could not load necessary files. Please check the GitHub repository and reboot the app.")
