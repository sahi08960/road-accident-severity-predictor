import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# ================= Load Model & Data =================
@st.cache_resource
def load_model():
    try:
        # Try loading saved model + training data
        model = joblib.load("xgb_accident_model.pkl")
        X_train = joblib.load("X_train.pkl")
        class_names = ["Slight", "Serious", "Fatal"]
    except FileNotFoundError:
        # Fallback: create dummy training data with SAME columns as your UI
        st.warning("⚠️ Model file not found. Using DummyClassifier for demo.")
        cols = ["Speed_limit", "Number_of_Vehicles", "Number_of_Casualties",
                "Weather_Conditions", "Light_Conditions"]
        
        # Fake training data (so shapes & columns match)
        X_train = pd.DataFrame(np.random.randint(0, 5, size=(100, len(cols))), columns=cols)
        y_train = np.random.randint(0, 3, size=100)  # 3 classes (Slight, Serious, Fatal)
        
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_train, y_train)
        class_names = ["Slight", "Serious", "Fatal"]
    return model, X_train, class_names


model, X_train, class_names = load_model()

# Initialize LIME
explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=class_names,
    mode="classification"
)

# ================= Streamlit UI =================
st.title("🚦 Road Accident Severity Predictor")

st.sidebar.header("Enter Accident Details")

# Example inputs (adjust according to dataset features)
speed = st.sidebar.number_input("Speed Limit", 20, 120, 50)
vehicles = st.sidebar.number_input("Number of Vehicles", 1, 10, 2)
casualties = st.sidebar.number_input("Number of Casualties", 1, 20, 1)
weather = st.sidebar.selectbox("Weather Condition", ["Fine", "Rain", "Snow", "Fog"])
light = st.sidebar.selectbox("Light Condition", ["Daylight", "Darkness - lights lit", "Darkness - no lights"])

# Create input dataframe
input_dict = {
    "Speed_limit": speed,
    "Number_of_Vehicles": vehicles,
    "Number_of_Casualties": casualties,
    "Weather_Conditions": weather,
    "Light_Conditions": light
}
input_df = pd.DataFrame([input_dict])

# Encode categorical features
for col in input_df.select_dtypes(include="object").columns:
    input_df[col] = input_df[col].astype("category").cat.codes

# ================= Prediction =================
if st.sidebar.button("Predict Severity"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"**Severity:** {class_names[prediction]}")
    st.write("**Probabilities:**")
    for i, c in enumerate(class_names):
        st.write(f"{c}: {proba[i]:.2f}")

    # ================= LIME Explanation =================
    exp = explainer.explain_instance(
        data_row=input_df.iloc[0],
        predict_fn=model.predict_proba,
        num_features=5
    )
    st.subheader("🔎 LIME Explanation")
    st.write(exp.as_list())
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)

st.markdown("---")
st.info("This is a demo Accident Severity Predictor using XGBoost + LIME")

