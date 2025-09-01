import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import zipfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    return joblib.load('xgb_accident_model.pkl')

@st.cache_data
def load_and_prep_data():
    """Load, clean, and prepare the raw accident data."""
    zip_path = "archive (4).zip"
    extract_path = "dataset"
    csv_filename = "AccidentsBig.csv"
    csv_filepath = os.path.join(extract_path, csv_filename)

    if not os.path.exists(csv_filepath):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    df = pd.read_csv(csv_filepath, low_memory=False)
    df.dropna(subset=['Accident_Severity', 'latitude', 'longitude'], inplace=True)
    
    # Create user-friendly labels for plotting
    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    df['Severity Label'] = df['Accident_Severity'].map(severity_map)
    
    df['Time_dt'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M')
    df['Hour'] = df['Time_dt'].dt.hour
    
    return df

# --- Load Assets ---
model = load_model()
df = load_and_prep_data()

# --- User Interface ---
st.title("Road Accident Severity: Prediction & Analysis 🚦")
st.markdown("This interactive dashboard uses an XGBoost model to predict accident severity and visualize high-risk locations across India.")

# --- Sidebar for User Input ---
st.sidebar.header(" Simulate an Accident Scenario")
st.sidebar.markdown("Use the controls below to predict the severity of an accident.")

# Create input widgets
hour = st.sidebar.slider("Hour of Day", 0, 23, 17) # Default to 5 PM
day_of_week = st.sidebar.selectbox("Day of Week", options=range(1, 8), format_func=lambda x: ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'][x-1], index=4)
light_conditions = st.sidebar.selectbox("Light Conditions", options=df['Light_Conditions'].unique(), index=0)
weather_conditions = st.sidebar.selectbox("Weather Conditions", options=df['Weather_Conditions'].unique(), index=0)
road_surface = st.sidebar.selectbox("Road Surface Conditions", options=df['Road_Surface_Conditions'].unique(), index=0)
num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 10, 2)
num_casualties = st.sidebar.number_input("Number of Casualties", 1, 15, 1)

# --- Prediction Logic ---
if st.sidebar.button("Predict Severity", type="primary"):
    # Create a DataFrame from user input
    # Note: The order of columns MUST match the order used during model training
    input_features = pd.DataFrame({
        'longitude': [78.9629], 'latitude': [20.5937], # Using center of India as placeholder
        'Police_Force': [df['Police_Force'].median()],
        'Number_of_Vehicles': [num_vehicles],
        'Number_of_Casualties': [num_casualties],
        'Day_of_Week': [day_of_week],
        'Local_Authority_(District)': [df['Local_Authority_(District)'].median()],
        'Local_Authority_(Highway)': [df['Local_Authority_(Highway)'].median()],
        '1st_Road_Class': [df['1st_Road_Class'].median()],
        '1st_Road_Number': [df['1st_Road_Number'].median()],
        'Road_Type': [df['Road_Type'].median()],
        'Speed_limit': [df['Speed_limit'].median()],
        'Junction_Detail': [df['Junction_Detail'].median()],
        'Junction_Control': [df['Junction_Control'].median()],
        '2nd_Road_Class': [df['2nd_Road_Class'].median()],
        '2nd_Road_Number': [df['2nd_Road_Number'].median()],
        'Pedestrian_Crossing-Human_Control': [df['Pedestrian_Crossing-Human_Control'].median()],
        'Pedestrian_Crossing-Physical_Facilities': [df['Pedestrian_Crossing-Physical_Facilities'].median()],
        'Light_Conditions': [light_conditions],
        'Weather_Conditions': [weather_conditions],
        'Road_Surface_Conditions': [road_surface],
        'Special_Conditions_at_Site': [df['Special_Conditions_at_Site'].median()],
        'Carriageway_Hazards': [df['Carriageway_Hazards'].median()],
        'Urban_or_Rural_Area': [df['Urban_or_Rural_Area'].median()],
        'Did_Police_Officer_Attend_Scene_of_Accident': [df['Did_Police_Officer_Attend_Scene_of_Accident'].median()],
        'Month': [df['Month'].median()],
        'Weekday': [df['Weekday'].median()],
        'Hour': [hour]
    })
    
    # Make prediction
    prediction_index = model.predict(input_features)[0]
    prediction_proba = model.predict_proba(input_features)[0]
    
    severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
    predicted_severity = severity_labels[prediction_index]
    
    # Display prediction
    st.subheader("Prediction Result")
    if predicted_severity == 'Fatal':
        st.error(f"Predicted Severity: *{predicted_severity}* (Probability: {prediction_proba[prediction_index]:.2%})")
    elif predicted_severity == 'Serious':
        st.warning(f"Predicted Severity: *{predicted_severity}* (Probability: {prediction_proba[prediction_index]:.2%})")
    else:
        st.success(f"Predicted Severity: *{predicted_severity}* (Probability: {prediction_proba[prediction_index]:.2%})")

# --- Interactive Map Visualization ---
st.subheader("Interactive Map of Accident Hotspots")
selected_severity_map = st.selectbox(
    "Select Severity to Visualize:",
    options=df['Severity Label'].unique()
)

map_data = df[df['Severity Label'] == selected_severity_map]

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v9',
    initial_view_state=pdk.ViewState(
        latitude=20.5937,
        longitude=78.9629,
        zoom=4,
        pitch=50,
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
))

