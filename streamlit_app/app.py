# streamlit_app/app.py
"""Simple Streamlit UI for the Insurance Prediction API.
Features:
- Light‑mode gradient background
- Token‑based login (hard‑coded token for demo)
- Form matching `HealthMetricsRequest` fields
- Instant synchronous prediction via httpx
- Displays premium, insurance tier, probabilities, and latency.
"""

import os
import json
from pathlib import Path
import httpx
import streamlit as st

# ---- Configuration ----
API_URL = os.getenv("API_URL", "http://localhost:8001/predict")
# Simple token for demo – in production replace with proper auth.
TOKEN = os.getenv("STREAMLIT_TOKEN", "demo-token")

# ---- Styling ----
st.set_page_config(page_title="Insurance Advisor", page_icon="🏥", layout="wide")
# Light gradient background via CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #f0f4ff, #e0eaff);
        color: #222;
    }
    .login-card {
        background: white;
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        max-width: 350px;
        margin: auto;
    }
    .prediction-card {
        background: #ffffff;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-top: 1rem;
        width: 100%;
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Authentication ----
# Authentication removed – users can access directly
st.session_state["authenticated"] = True

# ---- Main UI ----
# Logo placeholder – file not found; you may add an image at assets/insurance_logo.png
st.header("🏥 Health Insurance Prediction")

with st.form("prediction_form"):
    # Arrange inputs in compact three‑item rows using columns
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "M", "F"], index=0)
    with col3:
        bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=22.5)

    col4, col5, col6 = st.columns(3)
    with col4:
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
    with col5:
        smoker = st.selectbox("Smoker", ["Yes", "No"], index=1)
    with col6:
        region = st.selectbox("Region", ["Northeast", "Northwest", "Central", "Southeast", "Southwest"], index=0)

    col7, col8, col9 = st.columns(3)
    with col7:
        occupation = st.text_input("Occupation", value="Engineer")
    with col8:
        annual_income_usd = st.number_input("Annual income (USD)", min_value=0, max_value=1_000_000, value=50000)
    with col9:
        exercise_level = st.selectbox("Exercise level", ["Low", "Moderate", "High"], index=1)

    col10, col11, col12 = st.columns(3)
    with col10:
        chronic_diseases = st.number_input("Chronic disease count", min_value=0, max_value=10, value=0)
    with col11:
        doctor_visits = st.number_input("Doctor visits/yr", min_value=0, max_value=50, value=2)
    with col12:
        hospitalizations = st.number_input("Hospitalizations last yr", min_value=0, max_value=20, value=0)

    col13, col14, col15 = st.columns(3)
    with col13:
        alcohol = st.number_input("Alcohol drinks/week", min_value=0, max_value=50, value=2)
    with col14:
        blood_pressure = st.text_input("Blood pressure (systolic/diastolic)", value="120/80")
    with col15:
        diabetes = st.selectbox("Diabetes", ["Yes", "No"], index=1)

    col16, col17, col18 = st.columns(3)
    with col16:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=190)
    with col17:
        sleep_hours = st.slider("Sleep hours/night", min_value=3.0, max_value=12.0, value=7.0)
    with col18:
        stress_level = st.slider("Stress level (1‑10)", min_value=1.0, max_value=10.0, value=5.0)

    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Widowed"], index=0)
    submit = st.form_submit_button("Predict")

if submit:
    payload = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "occupation": occupation,
        "annual_income_usd": annual_income_usd,
        "exercise_level": exercise_level,
        "chronic_diseases": chronic_diseases,
        "doctor_visits_per_year": doctor_visits,
        "hospitalizations_last_year": hospitalizations,
        "alcohol_consumption_per_week": alcohol,
        "blood_pressure": blood_pressure,
        "diabetes": diabetes,
        "cholesterol": cholesterol,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "marital_status": marital_status,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Prediction successful!")
                # Styled container for results
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.subheader("📈 Prediction Summary")
                # Estimated cost and insurance tier
                st.write(f"**{data.get('predicted_insurance_type')} Insurance:** ${data.get('predicted_annual_premium'):,.2f}")
                # Show recommended insurance tier with its likelihood
                probs = data.get('classification_probabilities', {})
                if probs:
                    recommended = max(probs, key=probs.get)
                    st.write(f"**Suggested: {recommended} Insurance ({probs[recommended]*100:.1f}% likelihood)**")
                st.write("**Likelihood of each insurance tier**")
                # Convert to percentage strings for display
                probs_percent = {k: f"{v*100:.1f}%" for k, v in probs.items()}
                st.table(probs_percent)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        st.error(f"Failed to call API: {e}")
