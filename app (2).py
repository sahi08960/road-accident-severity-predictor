import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import zipfile
import os

# --- Page Configuration ---
st.set_page_config(page_title="Road Accident Severity Prediction", page_icon="🚗", layout="wide")

# --- Caching Functions ---
@st.cache_resource
def load_model():
    return joblib.load('xgb_accident_model.pkl')

@st.cache_data
def load_and_prep_data():
    zip_path = "archive (4).zip"
    extract_path = "dataset"
    csv_filename = "AccidentsBig.csv"
    csv_filepath = os.path.join(extract_path, csv_filename)

    if not os.path.exists(csv_filepath):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    df = pd.read_csv(csv_filepath, low_memory=False)
    df.dropna(subset=['Accident_Severity', 'latitude', 'longitude'], inplace=True)
    
    # --- THIS IS THE CRITICAL FIX: Perform Feature Engineering Here ---
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df['Time_dt'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M')
    
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df['Hour'] = df['Time_dt'].dt.hour
    
    # Create user-friendly labels for plotting
    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    df['Severity Label'] = df['Accident_Severity'].map(severity_map)
    
    # Encode categorical columns that might be objects
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['Severity Label']: # Don't encode our new friendly label
            df[col] = df[col].astype('category').cat.codes

    return df

# --- Load Assets ---
model = load_model()
df = load_and_prep_data()

# --- User Interface ---
st.title("Road Accident Severity: Prediction & Analysis 🚦")
st.markdown("This interactive dashboard uses an XGBoost model to predict accident severity and visualize high-risk locations across India.")

if model is not None and not df.empty:
    # --- Sidebar for User Input ---
    st.sidebar.header("🔮 Simulate an Accident Scenario")
    
    hour = st.sidebar.slider("Hour of Day", 0, 23, 17)
    day_of_week = st.sidebar.selectbox("Day of Week", options=df['Day_of_Week'].unique(), index=4)
    light_conditions = st.sidebar.selectbox("Light Conditions", options=sorted(df['Light_Conditions'].unique()))
    weather_conditions = st.sidebar.selectbox("Weather Conditions", options=sorted(df['Weather_Conditions'].unique()))
    road_surface = st.sidebar.selectbox("Road Surface Conditions", options=sorted(df['Road_Surface_Conditions'].dropna().unique()))
    num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 20, 2)
    num_casualties = st.sidebar.number_input("Number of Casualties", 1, 25, 1)

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Severity", type="primary", use_container_width=True):
        
        feature_columns_in_order = model.get_booster().feature_names
        
        input_data = {}
        # Create a dictionary of default values (median for numeric, mode for categorical)
        for col in feature_columns_in_order:
            if pd.api.types.is_numeric_dtype(df[col]):
                input_data[col] = [df[col].median()]
            else:
                input_data[col] = [df[col].mode()[0]]

        # Update with user inputs
        user_inputs = {
            'Hour': [hour], 'Day_of_Week': [day_of_week], 'Light_Conditions': [light_conditions],
            'Weather_Conditions': [weather_conditions], 'Road_Surface_Conditions': [road_surface],
            'Number_of_Vehicles': [num_vehicles], 'Number_of_Casualties': [num_casualties]
        }
        input_data.update(user_inputs)

        input_df = pd.DataFrame(input_data)[feature_columns_in_order]
        
        # Make prediction
        prediction_index = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
        predicted_severity = severity_labels[prediction_index]
        
        # Display prediction
        st.subheader("Prediction Result")
        if predicted_severity == 'Fatal':
            st.error(f"Predicted Severity: *{predicted_severity}* (Confidence: {prediction_proba[prediction_index]:.2%})")
        elif predicted_severity == 'Serious':
            st.warning(f"Predicted Severity: *{predicted_severity}* (Confidence: {prediction_proba[prediction_index]:.2%})")
        else:
            st.success(f"Predicted Severity: *{predicted_severity}* (Confidence: {prediction_proba[prediction_index]:.2%})")
    
    # --- Map Visualization (remains the same) ---
    st.subheader("🗺 Interactive Map of Accident Hotspots")
    # ... (the rest of the map code goes here) ...
