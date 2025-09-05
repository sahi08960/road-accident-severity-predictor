import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Accident Severity Predictor", page_icon="🚦", layout="centered")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    """Load all necessary assets: model, columns, and explainers."""
    model = joblib.load('xgb_accident_model.pkl')
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
    
    explainer_shap = shap.TreeExplainer(model)
    return model, model_columns, explainer_shap

# --- Load Assets ---
model, model_columns, explainer_shap = load_assets()

# --- 3. USER INTERFACE ---
st.title("🚦 Explainable Accident Severity Predictor")

if model is None or model_columns is None:
    st.error("Could not load necessary model files. Please ensure all files are in the repository.")
else:
    st.markdown("Select the conditions of an accident to get a prediction and an explanation.")

    # --- Sidebar for User Input ---
    st.sidebar.header("Accident Scenario")
    hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 17)
    day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 4)
    weather_conditions = st.sidebar.selectbox("Weather Condition (Code)", options=range(1, 10), index=0)
    num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 20, 2)
    num_casualties = st.sidebar.number_input("Number of Casualties", 1, 25, 1)

    # --- 4. PREDICTION AND EXPLANATION LOGIC ---
    if st.sidebar.button("Predict & Explain", type="primary", use_container_width=True):
        
        # Create a dictionary with default value 0 for all model columns
        input_data = {col: [0] for col in model_columns}

        # Update with the user's specific inputs
        user_inputs = {
            'Hour': [hour], 'Weekday': [day_of_week], # Assuming your feature is named Weekday
            'Weather_Conditions': [weather_conditions],
            'Number_of_Vehicles': [num_vehicles], 
            'Number_of_Casualties': [num_casualties]
        }
        # Update only the keys that exist in the input_data dictionary
        for key, value in user_inputs.items():
            if key in input_data:
                input_data[key] = value
        
        input_df = pd.DataFrame(input_data)[model_columns]
        
        # Make Prediction
        prediction_index = model.predict(input_df)[0]
        severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
        predicted_severity = severity_labels.get(prediction_index, "Unknown")

        # Display Prediction
        st.subheader("Model Prediction")
        # ... (rest of the prediction display code) ...

        # --- LIME and SHAP Explanations ---
        # ... (rest of the LIME and SHAP code, it should work now) ...
