# --- Prediction Logic ---
if st.sidebar.button("Predict Severity", type="primary", use_container_width=True):

    # Get feature names from model
    feature_columns_in_order = [col.strip().lower() for col in model.get_booster().feature_names]

    # Build default input data
    input_data = {}
    for col in feature_columns_in_order:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                input_data[col] = [df[col].median()]
            else:
                input_data[col] = [df[col].mode()[0]]
        else:
            input_data[col] = [0]  # fallback if model expects a missing feature

    # User inputs (make sure names are lowercased to match)
    user_inputs = {
        'hour': [hour],
        'day_of_week': [day_of_week],
        'light_conditions': [light_conditions],
        'weather_conditions': [weather_conditions],
        'road_surface_conditions': [road_surface],
        'number_of_vehicles': [num_vehicles],
        'number_of_casualties': [num_casualties]
    }

    # Overwrite defaults with user inputs
    for k, v in user_inputs.items():
        if k in input_data:
            input_data[k] = v

    # Convert to DataFrame with *exact model feature order*
    input_df = pd.DataFrame(input_data)[feature_columns_in_order]

    # Debug (show feature mismatch if any)
    st.write("🔍 Model expects:", feature_columns_in_order)
    st.write("📊 Input DataFrame:", input_df.columns.tolist())

    # Make prediction
    prediction_index = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    severity_labels = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
    predicted_severity = severity_labels[prediction_index]

    # Display prediction
    st.subheader("Prediction Result")
    if predicted_severity == 'Fatal':
        st.error(f"Predicted Severity: *{predicted_severity}* (Confidence: {prediction_proba[prediction_index]:.2%})")
    elif predicted_severity == 'Serious':
        st.warning(f"Predicted Severity: *{predicted_severity}* (Confidence: {prediction_proba[prediction_index]:.2%})")
    else:
        st.success(f"Predicted Severity: *{predicted_severity}* (Confidence: {prediction_proba[prediction_index]:.2%})")
