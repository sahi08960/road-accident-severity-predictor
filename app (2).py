# ==============================================================================
#                      app.py - Simplified Streamlit Application
# ==============================================================================
import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib
import zipfile
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
)

# --- 2. CACHED FUNCTIONS ---
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        return joblib.load("xgb_accident_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please add 'xgb_accident_model.pkl'.")
        return None

@st.cache_data
def load_and_prep_data():
    """Load and prepare dataset (only for visualization + defaults)."""
    zip_path = "archive (4).zip"
    extract_path = "dataset"
    csv_filename = "AccidentsBig.csv"
    csv_filepath = os.path.join(extract_path, csv_filename)

    if not os.path.exists(csv_filepath):
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
        else:
            st.error("❌ Dataset not found. Please add 'archive (4).zip'.")
            return pd.DataFrame()

    df = pd.read_csv(csv_filepath, low_memory=False)
    df.dropna(subset=["Accident_Severity", "latitude", "longitude"], inplace=True)

    # Labels
    severity_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
    df["Severity Label"] = df["Accident_Severity"].map(severity_map)

    # Extract hour
    df["Time_dt"] = pd.to_datetime(df["Time"], errors="coerce", format="%H:%M")
    df["Hour"] = df["Time_dt"].dt.hour

    return df

# --- 3. LOAD MODEL + DATA ---
model = load_model()
df = load_and_prep_data()

# --- 4. USER INTERFACE ---
st.title("🚦 Road Accident Severity Prediction (Simplified)")
st.markdown("Enter details below to simulate accident severity prediction.")

if model is not None and not df.empty:
    # Sidebar Inputs
    st.sidebar.header("📝 Accident Details")

    hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
    daylight = st.sidebar.selectbox("Daylight / Darkness", ["Daylight", "Darkness"])
    weather = st.sidebar.selectbox("Weather Conditions", sorted(df["Weather_Conditions"].dropna().unique()))
    num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 15, 2)

    # --- Prediction ---
    if st.sidebar.button("🔮 Predict Severity"):
        feature_columns_in_order = model.get_booster().feature_names

        # Default values (median/mode)
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
        input_data.update({
            "Hour": [hour],
            "Weather_Conditions": [weather],
            "Number_of_Vehicles": [num_vehicles],
            "Light_Conditions": [daylight]  # using Daylight/Darkness
        })

        # Make dataframe in correct order
        input_df = pd.DataFrame(input_data)[feature_columns_in_order]

        # Predict
        prediction_index = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        severity_labels = {0: "Slight", 1: "Serious", 2: "Fatal"}
        predicted_severity = severity_labels[prediction_index]

        # Display result
        st.subheader("📊 Prediction Result")
        if predicted_severity == "Fatal":
            st.error(f"Predicted Severity: **{predicted_severity}** (Confidence: {prediction_proba[prediction_index]:.2%})")
        elif predicted_severity == "Serious":
            st.warning(f"Predicted Severity: **{predicted_severity}** (Confidence: {prediction_proba[prediction_index]:.2%})")
        else:
            st.success(f"Predicted Severity: **{predicted_severity}** (Confidence: {prediction_proba[prediction_index]:.2%})")

    # --- Map Visualization ---
    st.subheader("🗺 Accident Hotspots Map")
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=20.5937, longitude=78.9629, zoom=4, pitch=50,
        ),
        layers=[
            pdk.Layer(
                "HeatmapLayer",
                data=df,
                get_position="[longitude, latitude]",
                radius=100,
            )
        ],
    ))
else:
    st.error("⚠️ Could not load model or dataset. Please check your repository.")
