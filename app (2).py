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
    
    # --- Feature Engineering ---
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df['Time_dt'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M')
    
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df['Hour'] = df['Time_dt'].dt.hour
    
    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    df['Severity Label'] = df['Accident_Severity'].map(severity_map)
    
    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['Severity Label']:
            df[col] = df[col].astype('category').cat.codes

    # Normalize column names (for safe matching with model features)
    df.columns = df.columns.str.strip().str.lower()
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
    day_of_week = st.sidebar.selectbox("Day of Week", options=df['weekday'].unique(), index=4)
    light_conditions = st.sidebar.selectbox("Light Conditions", options=sorted(df['light_conditions'].unique()))
    weather_conditions = st.sidebar.selectbox("Weather Conditions", options=sorted(df['weather_conditions'].unique()))
    road_surface = st.sidebar.selectbox("Road Surface Conditions", options=sorted(df['road_surface_conditions'].dropna().unique()))
    num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 20, 2)
    num_casualties = st.sidebar.number_input("Number of Casualties", 1, 25, 1)

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Severity", type="primary", use_container_width=True):
        
        feature_columns_in_order = [col.strip().lower() for col in model.get_booster().feature_names]
        
        input_data = {}
        for col in feature_columns_in_order:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    input_data[col] = [df[col].median()]
                else:
                    input_data[col] = [df[col].mode()[0]]
            else:
                input_data[col] = [0]  # default fallback

        # Update with user inputs
        user_inputs = {
            'hour': [hour],
            'weekday': [day_of_week],
            'light_conditions': [light_conditions],
            'weather_conditions': [weather_conditions],
            'road_surface_conditions': [road_surface],
            'number_of_vehicles': [num_vehicles],
            'number_of_casualties': [num_casualties]
        }
        input_data.update(user_inputs)

        # --- Encode categorical inputs using training mapping ---
        for col in ['weekday', 'light_conditions', 'weather_conditions', 'road_surface_conditions']:
            if col in df.columns and col in input_data:
                try:
                    categories = dict(enumerate(df[col].astype('category').cat.categories))
                    reverse_map = {v: k for k, v in categories.items()}
                    input_data[col] = [reverse_map.get(input_data[col][0], 0)]
                except Exception:
                    input_data[col] = [0]

        # Create input DataFrame with correct order
        input_df = pd.DataFrame(input_data)
        input_df = input_df.reindex(columns=feature_columns_in_order, fill_value=0)

        # Make prediction
        prediction_index = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
        predicted_severity = severity_labels[prediction_index]
        
        # Display prediction
        st.subheader("Prediction Result")
        if predicted_severity == 'Fatal':
            st.error(f"Predicted Severity: {predicted_severity} (Confidence: {prediction_proba[prediction_index]:.2%})")
        elif predicted_severity == 'Serious':
            st.warning(f"Predicted Severity: {predicted_severity} (Confidence: {prediction_proba[prediction_index]:.2%})")
        else:
            st.success(f"Predicted Severity: {predicted_severity} (Confidence: {prediction_proba[prediction_index]:.2%})")
    
    # --- Map Visualization ---
    st.subheader("🗺 Interactive Map of Accident Hotspots")
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=df['latitude'].mean(), longitude=df['longitude'].mean(), zoom=5, pitch=50),
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=df[['latitude', 'longitude']],
                get_position='[longitude, latitude]',
                radius=500,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=df[['latitude', 'longitude']],
                get_position='[longitude, latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))
