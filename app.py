import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Diabetes Risk Advisor", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource # This keeps the model in memory so the app stays fast
def load_assets():
    model = joblib.load('diabetes_ensemble_model.pkl')
    features = joblib.load('selected_features.pkl')
    return model, features

try:
    pipeline, selected_features = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- HEADER ---
st.title("🩺 Diabetes Health Risk Advisor")
st.markdown("""
This tool uses an **Ensemble Machine Learning model (XGBoost + LightGBM)** to estimate your risk of diabetes based on CDC health indicators.
""")

st.sidebar.header("Your Health Metrics")

# --- DYNAMIC INPUT GENERATION ---
# This dictionary maps internal feature names to user-friendly labels/options

# 1. Keep the mapping dictionary outside or inside the function
age_mapping = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
    "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
    "70-74": 11, "75-79": 12, "80 or older": 13
}

def get_user_inputs(features):
    inputs = {}
    
    for col in features:
        if col == 'BMI':
            inputs[col] = st.sidebar.slider("Your BMI", 10.0, 60.0, 25.0)
            
        # --- INSERTED FIX START ---
        elif col == 'Age':
            selected_age_label = st.sidebar.selectbox(
                "Your Age Group", 
                options=list(age_mapping.keys())
            )
            # Map the string label back to the numeric value (1-13) for the model
            inputs['Age'] = age_mapping[selected_age_label]
        # --- INSERTED FIX END ---
            
        elif col == 'GenHlth':
            inputs[col] = st.sidebar.select_slider(
                "General Health Rating", 
                options=[1, 2, 3, 4, 5],
                help="1: Excellent, 2: Very Good, 3: Good, 4: Fair, 5: Poor"
            )
        elif col in ['HighBP', 'HighChol', 'Smoker', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'CholCheck']:
            # Clean up the display labels for a better UI
            display_label = col.replace('HeartDiseaseorAttack', 'Heart Disease History').replace('HighBP', 'High Blood Pressure')
            inputs[col] = st.sidebar.selectbox(
                f"Status: {display_label}", 
                options=[0, 1], 
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        elif col in ['MentHlth', 'PhysHlth']:
            inputs[col] = st.sidebar.number_input(f"Days of poor {col} (Last 30 days)", 0, 30, 0)
        else:
            inputs[col] = st.sidebar.number_input(f"Enter value for {col}", value=0)
            
    return pd.DataFrame(inputs, index=[0])

# Collect data and ensure column order matches the model
user_input_df = get_user_inputs(selected_features)
user_input_df = user_input_df[selected_features] # CRITICAL: Maintains feature order

# --- PREDICTION LOGIC ---
st.divider()
if st.button("Analyze My Risk Profile", type="primary"):
    # Get probabilities
    probs = pipeline.predict_proba(user_input_df)[0]
    
    # Extract specific values
    # 0: Healthy, 1: Pre-diabetic, 2: Diabetic
    p_healthy = probs[0] * 100
    p_pre = probs[1] * 100
    p_diabetic = probs[2] * 100

    # Display Results
    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Healthy", f"{p_healthy:.1f}%")
    c2.metric("Pre-Diabetic", f"{p_pre:.1f}%")
    c3.metric("Diabetes Risk", f"{p_diabetic:.1f}%")

    # Final Advisory
    if p_diabetic > 50:
        st.error("⚠️ **High Risk Detected:** Your indicators strongly align with diabetic profiles. Please consult a healthcare professional.")
    elif (p_diabetic + p_pre) > 40:
        st.warning("🟠 **Moderate Risk:** Consider lifestyle adjustments and routine blood sugar checkups.")
    else:
        st.success("🟢 **Low Risk:** Your indicators suggest a healthy profile. Keep maintaining a balanced lifestyle!")

st.info("Note: This is an ML-based estimation for educational purposes and not a medical diagnosis.")