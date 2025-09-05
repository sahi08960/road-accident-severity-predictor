import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚦",
    layout="centered"
)

# --- 2. ASSET LOADING (with caching) ---
# This function loads the trained model and the SHAP explainer.
# It's cached so it only runs once.
@st.cache_resource
def load_model_and_explainer():
    """Load the model and create the SHAP explainer."""
    import shap  # <-- THIS IS THE FIX for the NameError
    try:
        model = joblib.load('xgb_accident_model.pkl')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except FileNotFoundError:
        return None, None

# This function loads and prepares the data needed for LIME and default values.
@st.cache_data
def load_training_data():
    """Load and prep the data needed for LIME and default values."""
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
        
        # This minimal dataframe is just for getting feature names and defaults
        X = df.drop(columns=['Accident_Severity', 'Accident_Index', 'Date', 'Time', 'LSOA_of_Accident_Location'])
        
        # Encode categorical columns so we can get the mode correctly
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].astype('category').cat.codes
            
        return X
    except FileNotFoundError:
        return None

# --- Load Assets ---
model, explainer_shap = load_model_and_explainer()
X_train_for_lime = load_training_data()

# --- 3. MAIN USER INTERFACE ---
st.title("🚦 Explainable Accident Severity Predictor")

if model is None or X_train_for_lime is None:
    st.error("Could not load necessary model or data files. Please ensure 'xgb_accident_model.pkl' and 'archive (4).zip' are in the GitHub repository and reboot the app.")
else:
    st.markdown("Select the conditions of an accident scenario below to get a prediction and an explanation of *why* the model made that choice.")

    # --- Sidebar for User Input ---
    st.sidebar.header("Accident Scenario")

    hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 17)
    day_of_week = st.sidebar.selectbox("Day of Week", options=sorted(X_train_for_lime['Day_of_Week'].unique()), index=4)
    weather_conditions = st.sidebar.selectbox("Weather Condition", options=sorted(X_train_for_lime['Weather_Conditions'].unique()))
    num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 20, 2)
    num_casualties = st.sidebar.number_input("Number of Casualties", 1, 25, 1)

    # --- 4. PREDICTION AND EXPLANATION LOGIC ---
    if st.sidebar.button("Predict & Explain", type="primary", use_container_width=True):
        
        feature_columns_in_order = model.get_booster().feature_names
        
        input_data = {}
        for col in feature_columns_in_order:
            if pd.api.types.is_numeric_dtype(X_train_for_lime[col]):
                input_data[col] = [X_train_for_lime[col].median()]
            else:
                input_data[col] = [X_train_for_lime[col].mode()[0]]

        user_inputs = {
            'Hour': [hour], 'Day_of_Week': [day_of_week], 
            'Weather_Conditions': [weather_conditions],
            'Number_of_Vehicles': [num_vehicles], 
            'Number_of_Casualties': [num_casualties]
        }
        input_data.update(user_inputs)
        
        input_df = pd.DataFrame(input_data)[feature_columns_in_order]
        
        # --- Make Prediction ---
        prediction_index = model.predict(input_df)[0]
        severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
        predicted_severity = severity_labels[prediction_index]

        # --- Display Prediction ---
        st.subheader("Model Prediction")
        if predicted_severity == 'Fatal':
            st.error(f"The model predicts this scenario is most likely to be a *Fatal* accident.")
        elif predicted_severity == 'Serious':
            st.warning(f"The model predicts this scenario is most likely to be a *Serious* accident.")
        else:
            st.success(f"The model predicts this scenario is most likely to be a *Slight* accident.")

        # --- LIME Explanation ---
        with st.expander("View the Top Reasons for this Prediction (LIME)"):
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_for_lime.values,
                feature_names=X_train_for_lime.columns.tolist(),
                class_names=list(severity_labels.values()),
                mode='classification'
            )
            
            lime_explanation = lime_explainer.explain_instance(
                input_df.iloc[0].values,
                model.predict_proba,
                num_features=5,
                top_labels=1
            )
            
            fig = lime_explanation.as_pyplot_figure(label=prediction_index)
            st.pyplot(fig, use_container_width=True)
            st.markdown("This chart shows the top 5 factors for this single prediction. *Green bars* support the prediction, while *red bars* oppose it.")

        # --- SHAP Explanation (Waterfall Plot) ---
        with st.expander("View a Detailed Breakdown of Feature Contributions (SHAP)"):
            shap_values = explainer_shap.shap_values(input_df)
            
            fig, ax = plt.subplots()
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[prediction_index][0],
                base_values=explainer_shap.expected_value[prediction_index],
                data=input_df.iloc[0],
                feature_names=input_df.columns.tolist()
            ), show=False)
            st.pyplot(fig, use_container_width=True)
            st.markdown("This waterfall plot shows how each feature's value pushes the model's output from the baseline (E[f(x)]) to the final prediction for this specific scenario.")
