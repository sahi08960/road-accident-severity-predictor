import streamlit as st
import pandas as pd
import numpy as np
import joblib
import zipfile
import os
import shap # <-- Main import is still here
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
def load_model_and_explainer():
    """Load the model and create the SHAP explainer."""
    import shap # <--- THIS IS THE FIX. Import the library inside the cached function.
    model = joblib.load('xgb_accident_model.pkl')
    explainer = shap.TreeExplainer(model)
    return model, explainer

@st.cache_data
def load_training_data():
    """Load and prep the data needed for LIME and default values."""
    zip_path = "archive (4).zip"
    extract_path = "dataset"
    # ... (rest of the function is the same)
    # ... (rest of your app code is the same)
