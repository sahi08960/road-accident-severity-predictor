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
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚦",
    layout="centered"
)

# --- 2. ASSET LOADING (with caching) ---
@st.cache_resource
def load_assets():
    """Load all necessary assets: model, columns, and explainers."""
    import shap # Import here for the cached function
    try:
        model = joblib.load('xgb_accident_model.pkl')
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        
        explainer_shap = shap.TreeExplainer(model)
        return model, model_columns, explainer_shap
    except FileNotFoundError:
        return None, None, None

# This minimal DataFrame is needed for LIME's explainer
@st.cache_data
def load_training_data_for_lime():
    # Load your full dataset and preprocess it just enough to get the training data structure
    # This is a simplified version; in a real app, you might save a preprocessed sample
    # For this project, we can derive it from the full data you already have.
    df = pd.read_csv('AccidentsBig.csv', low_memory=False) # Assumes the CSV is inside the zip
    df.dropna(subset=['Accident_Severity'], inplace=True)
    X = df.drop(columns=['Accident_Severity', 'Accident_Index', 'Date', 'Time', 'LSOA_of_Accident_Location'])
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype('category').cat.codes
    return X


# --- Load Assets ---
model, model_columns, explainer_shap = load_assets()
# We need to make sure the dataset file AccidentsBig.csv is extracted from your archive.zip
# This will happen automatically if you use the full app structure, but for this simplified version,
# we assume AccidentsBig.csv is available. A simple way is to unzip it and add it to your repo.
# X_train_for_lime = load_training_data_for_lime() # Optional for now to simplify deployment

# --- 3. USER INTERFACE ---
st.title("🚦 Explainable Accident Severity Predictor")

if model is None or model_columns is None:
    st.error("Could not load necessary model files. Please ensure all required files are in the GitHub repository and reboot the app.")
else:
    st.markdown("Select the conditions of an accident to get a prediction and an explanation.")

    # --- Sidebar for User Input (Simplified to 3 inputs) ---
    st.sidebar.header("Accident Scenario")
    hour = st.sidebar.slider("Time of Day (Hour)", 0, 23, 17) # Default to 5 PM
    # Assuming weather codes 1=Fine, 2=Rainy, 7=Fog
    weather_conditions = st.sidebar.selectbox(
        "Weather Condition",
        options={1: "Fine", 2: "Rainy", 7: "Fog/Mist"}.keys(),
        format_func=lambda x: {1: "Fine", 2: "Rainy", 7: "Fog/Mist"}[x]
    )
    num_vehicles = st.sidebar.number_input("Number of Vehicles Involved", 1, 20, 2)

    # --- 4. PREDICTION AND EXPLANATION LOGIC ---
    if st.sidebar.button("Predict & Explain", type="primary", use_container_width=True):
        
        # Create a dictionary with a default value (0) for all model columns
        input_data = {col: [0] for col in model_columns}

        # Update with the user's specific inputs
        user_inputs = {
            'Hour': [hour],
            'Weather_Conditions': [weather_conditions],
            'Number_of_Vehicles': [num_vehicles]
        }
        
        # Update only the keys that exist in the input_data dictionary
        for key, value in user_inputs.items():
            if key in input_data:
                input_data[key] = value
        
        # Create the final DataFrame in the correct order for the model
        input_df = pd.DataFrame(input_data)[model_columns]
        
        # --- Make Prediction ---
        prediction_index = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
        predicted_severity = severity_labels.get(prediction_index, "Unknown")
        is_serious = predicted_severity in ['Serious', 'Fatal']

        # --- Display Prediction ---
        st.subheader("Model Prediction")
        if is_serious:
            st.error(f"The model predicts this scenario is likely to be a *{predicted_severity}* accident.")
        else:
            st.success(f"The model predicts this scenario is likely to be a *{predicted_severity}* (Normal) accident.")

        # --- Display Probabilities as a simple bar chart ---
        st.write("*Prediction Confidence:*")
        prob_df = pd.DataFrame({
            'Severity': ['Slight (Normal)', 'Serious', 'Fatal'],
            'Probability': prediction_proba * 100 # As percentage
        })
        st.bar_chart(prob_df.set_index('Severity'))

        # --- SHAP Explanation (Waterfall Plot) ---
        with st.expander("View Detailed Reasons for this Prediction (SHAP)"):
            shap_values = explainer_shap.shap_values(input_df)
            
            st.write(f"*Explanation for the '{predicted_severity}' prediction:*")
            fig, ax = plt.subplots()
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[prediction_index][0],
                base_values=explainer_shap.expected_value[prediction_index],
                data=input_df.iloc[0],
                feature_names=input_df.columns.tolist()
            ), max_display=10, show=False) # Show top 10 features
            st.pyplot(fig, use_container_width=True)
            st.markdown("This plot shows how each feature pushed the prediction from the average case to the final result.")
