import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import zipfile
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚦",
    layout="centered"
)

# --- 2. ASSET LOADING (with caching for performance) ---
@st.cache_resource
def load_assets():
    """Load all necessary assets: model, columns list, and the SHAP explainer."""
    import shap  # Import shap here for the cached function to work reliably
    try:
        model = joblib.load('xgb_accident_model.pkl')
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        
        explainer_shap = shap.TreeExplainer(model)
        return model, model_columns, explainer_shap
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_default_data():
    """Load and prepare the full dataset to get realistic default values (median/mode)."""
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
        
        X = df.drop(columns=['Accident_Severity', 'Accident_Index', 'Date', 'Time', 'LSOA_of_Accident_Location'])
        
        # Engineer the same features the model was trained on
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df['Time_dt'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M')
        X['Month'] = df['Date'].dt.month
        X['Weekday'] = df['Date'].dt.weekday
        X['Hour'] = df['Time_dt'].dt.hour

        # Encode any text columns to numbers
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes
            
        return X
    except FileNotFoundError:
        return None

# --- Load Assets ---
model, model_columns, explainer_shap = load_assets()
X_defaults = load_default_data()

# --- 3. USER INTERFACE ---
st.title("🚦 Explainable Accident Severity Predictor")

if model is None or model_columns is None or X_defaults is None:
    st.error("Could not load necessary model or data files. Please ensure 'xgb_accident_model.pkl', 'model_columns.json', and 'archive (4).zip' are in your GitHub repository and reboot the app.")
else:
    st.markdown("Select the conditions of an accident to get a prediction and an explanation of *why* the model made that choice.")

    # --- Sidebar for User Input ---
    st.sidebar.header("Accident Scenario")
    hour = st.sidebar.slider("Time of Day (Hour)", 0, 23, 17)
    weather_conditions = st.sidebar.selectbox(
        "Weather Condition",
        options={1: "Fine", 2: "Rainy", 7: "Fog/Mist", 8: "Other"}.keys(),
        format_func=lambda x: {1: "Fine", 2: "Rainy", 7: "Fog/Mist", 8: "Other"}[x]
    )
    num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 20, 2)

    # --- 4. PREDICTION AND EXPLANATION LOGIC ---
    if st.sidebar.button("Predict & Explain", type="primary", use_container_width=True):
        
        # Create a dictionary for the input using realistic defaults (median/mode)
        input_data = {}
        for col in model_columns:
            if pd.api.types.is_numeric_dtype(X_defaults[col]):
                input_data[col] = X_defaults[col].median()
            else:
                input_data[col] = X_defaults[col].mode()[0]
        
        # Now, create a DataFrame from this single row of defaults
        input_df = pd.DataFrame([input_data])
        
        # Update the DataFrame with the user's specific inputs
        input_df['Hour'] = hour
        input_df['Weather_Conditions'] = weather_conditions
        input_df['Number_of_Vehicles'] = num_vehicles
        
        # Ensure the column order is exactly what the model expects
        input_df = input_df[model_columns]

        # --- Make Prediction ---
        prediction_index = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
        predicted_severity = severity_labels.get(prediction_index, "Unknown")
        is_serious = predicted_severity in ['Serious', 'Fatal']

        # --- Display Prediction ---
        st.subheader("Model Prediction")
        if is_serious:
            st.error(f"The model predicts this is a *{predicted_severity}* accident.")
        else:
            st.success(f"The model predicts this is a *{predicted_severity} (Normal)* accident.")
        
        # Display Probabilities as a bar chart
        st.write("*Prediction Confidence:*")
        prob_df = pd.DataFrame({
            'Severity': ['Slight (Normal)', 'Serious', 'Fatal'],
            'Probability': prediction_proba * 100
        })
        st.bar_chart(prob_df.set_index('Severity'))

        # --- SHAP Explanation (Waterfall Plot) ---
        with st.expander("View Detailed Reasons for this Prediction (SHAP)"):
            shap_values = explainer_shap.shap_values(input_df)
            
            # Create the SHAP Explanation object
            shap_explanation = shap.Explanation(
                values=shap_values[prediction_index][0],
                base_values=explainer_shap.expected_value[prediction_index],
                data=input_df.iloc[0],
                feature_names=input_df.columns.tolist()
            )
            
            st.write(f"*Explanation for the '{predicted_severity}' prediction:*")
            fig, ax = plt.subplots()
            shap.waterfall_plot(shap_explanation, max_display=10, show=False)
            st.pyplot(fig, use_container_width=True)
            st.markdown("This waterfall plot shows how each feature pushed the prediction from the average case (bottom) to the final result (top). Red arrows increase the risk, blue arrows decrease it.")
