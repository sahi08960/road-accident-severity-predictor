import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
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

# --- 2. DATA MAPPINGS ---
# These dictionaries will make our app user-friendly.
WEATHER_MAP = {
    1: 'Fine', 2: 'Raining', 3: 'Snowing', 4: 'Fine + High winds',
    5: 'Raining + High winds', 6: 'Snowing + High winds', 7: 'Fog or mist',
    8: 'Other', 9: 'Unknown', -1: 'Data missing or out of range'
}

LIGHT_MAP = {
    1: 'Daylight', 4: 'Darkness - lights lit', 5: 'Darkness - lights unlit',
    6: 'Darkness - no lighting', 7: 'Darkness - lighting unknown', -1: 'Data missing or out of range'
}

# --- 3. CACHED FUNCTIONS FOR PERFORMANCE ---
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
    
    # --- Create user-friendly labels from the maps ---
    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    df['Severity Label'] = df['Accident_Severity'].map(severity_map)
    df['Weather Label'] = df['Weather_Conditions'].map(WEATHER_MAP)
    df['Light Label'] = df['Light_Conditions'].map(LIGHT_MAP)
    
    return df

# --- 4. LOAD ASSETS ---
model = load_model()
df = load_and_prep_data()

# --- 5. MAIN USER INTERFACE ---
st.title("🚦 Road Accident Severity: Prediction & Analysis")
st.markdown("This dashboard uses a machine learning model to predict accident severity and visualize high-risk locations across India.")

# Create a container for the main content
main_container = st.container()

with main_container:
    # --- Interactive Map Visualization (CORRECTED) ---
    st.subheader("🗺 Interactive Map of Accident Hotspots")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Filter by Severity (using the friendly label)
        selected_severity_map = st.selectbox(
            "Filter by Severity:",
            options=sorted(df['Severity Label'].dropna().unique())
        )
    with col2:
        # Filter by Weather (using the friendly label)
        selected_weather_map = st.selectbox(
            "Filter by Weather:",
            options=sorted(df['Weather Label'].dropna().unique())
        )
    with col3:
        # Filter by Light Condition (using the friendly label)
        selected_light_map = st.selectbox(
            "Filter by Light Condition:",
            options=sorted(df['Light Label'].dropna().unique())
        )

    # --- Filter Data Based on User Selection ---
    map_data = df[
        (df['Severity Label'] == selected_severity_map) &
        (df['Weather Label'] == selected_weather_map) &
        (df['Light Label'] == selected_light_map)
    ]

    st.write(f"Visualizing *{len(map_data)}* accidents matching your criteria.")

    # --- Create and display the Folium Map ---
    if not map_data.empty:
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/dark-v9',
            initial_view_state=pdk.ViewState(
                latitude=20.5937, longitude=78.9629, zoom=4, pitch=50
            ),
            layers=[
                pdk.Layer(
                   'HeatmapLayer',
                   data=map_data,
                   get_position='[longitude, latitude]',
                   opacity=0.8,
                   get_weight=1
                ),
            ],
            tooltip={"html": "<b>Severity:</b> {Severity Label}<br/><b>Weather:</b> {Weather Label}"}
        ))
    else:
        st.warning("No accident data available for the selected filters.")

# --- Sidebar for Prediction ---
st.sidebar.header("🔮 Simulate an Accident Scenario")
# ... (The prediction sidebar code remains the same as the previous correct version) ...
# ... It will use its own separate set of friendly dropdowns ...
