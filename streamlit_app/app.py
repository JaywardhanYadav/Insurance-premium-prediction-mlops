# streamlit_app/app.py
"""Unified Streamlit UI for the Insurance Prediction API & Metrics.
Features:
- Highly stylized, bold, and attractive sidebar navigation
- Single-page, tab-less input form for all user fields
- Custom styled long red Predict button
- Instant synchronous prediction via httpx
"""

import json
import os
from pathlib import Path

import httpx
import streamlit as st

import sys

# ---- Configuration ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

API_URL = os.getenv("API_URL", "http://localhost:8001/predict")
METRICS_PATH = Path(
    os.getenv("METRICS_PATH", PROJECT_ROOT / "metrics" / "training_metrics.json")
)


@st.cache_data
def load_training_metrics(metrics_path: str) -> dict | None:
    """Load persisted training metrics written by src.train."""
    path = Path(metrics_path)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def predict_local(payload: dict) -> dict:
    """Fallback local prediction using serialized models when FastAPI is offline."""
    import joblib
    import pandas as pd
    import numpy as np
    from src.features import engineer_raw_features, transform_features, derive_tier_from_premium
    from src.ensemble import EnsembleRegressor  # Required to deserialize model

    preprocessor_path = PROJECT_ROOT / "models" / "preprocessor.joblib"
    regressor_path = PROJECT_ROOT / "models" / "regressor.joblib"

    if not preprocessor_path.is_file() or not regressor_path.is_file():
        raise FileNotFoundError("Local model files not found. Run python src/train.py to generate models.")

    artifacts = joblib.load(preprocessor_path)
    regressor = joblib.load(regressor_path)

    # Transform inputs
    input_df = pd.DataFrame([payload])
    engineered = engineer_raw_features(input_df)
    feature_matrix = transform_features(engineered, artifacts)

    # Predict & invert log scaling
    log_pred = regressor.predict(feature_matrix)[0]
    premium = float(np.expm1(log_pred))

    # Derive tier
    tier_config = {"thresholds": [15000, 30000], "labels": ["Basic", "Standard", "Premium"]}
    predicted_label = derive_tier_from_premium(premium, config=tier_config)

    # Compute proximity confidence
    min_dist = min(abs(premium - t) for t in tier_config["thresholds"])
    confidence = float(min(min_dist / 5000.0, 1.0))

    return {
        "predicted_annual_premium": premium,
        "predicted_insurance_type": predicted_label,
        "tier_confidence": confidence,
        "tier_boundaries": {
            "Basic -> Standard": 15000.0,
            "Standard -> Premium": 30000.0
        }
    }


def _format_r2(value: float) -> str:
    return f"{value:.3f}"


def _format_usd(value: float) -> str:
    return f"${value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_f1(value: float) -> str:
    return f"{value:.3f}"

# ---- Styling ----
st.set_page_config(page_title="Insurance Advisor Dashboard", page_icon="🏥", layout="wide")

st.markdown(
    """
    <style>
    /* Main Layout Styling */
    .stApp {
        background: linear-gradient(to bottom right, #f0f4ff, #e0eaff);
        color: #222;
    }
    
    /* --- ENHANCED PREDICTION CONTAINER CARD --- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-radius: 0.75rem !important;
        padding: 2rem !important;
        box-shadow: 0 10px 25px rgba(30, 41, 59, 0.1) !important;
        border: 1px solid #e2e8f0 !important;
        margin-top: 1.5rem !important;
    }
    
    .result-premium-title {
        color: #1e293b !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .result-premium-price {
        color: #10b981 !important; 
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        background: #ecfdf5;
        padding: 0.4rem 1.2rem;
        border-radius: 0.5rem;
        display: inline-block;
        margin-bottom: 1.2rem !important;
        border: 1px solid #a7f3d0;
    }
    
    .result-suggestion {
        font-size: 1.25rem !important;
        color: #475569 !important;
        margin-bottom: 1.5rem !important;
    }
    
    div[data-testid="stTable"] table {
        border-radius: 0.5rem !important;
        overflow: hidden !important;
        border-collapse: collapse !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
    }
    div[data-testid="stTable"] th {
        background-color: #1e293b !important;
        color: white !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* --- SIDEBAR BASE STYLE --- */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    
    /* Combined Sidebar Header Layout */
    .sidebar-title {
        color: #f8fafc !important;
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-subtitle {
        color: #94a3b8 !important; 
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem;
        border-bottom: 2px solid #38bdf8;
        padding-bottom: 0.75rem;
    }

    /* CRITICAL SPACING FIX: Completely eliminate any residual radio label placeholder spacing */
    [data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
        display: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* --- RADIO LABELS & CONTAINERS --- */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background-color: #334155 !important;
        padding: 0.8rem 1rem !important;
        border-radius: 0.6rem !important;
        margin-bottom: 0.75rem !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        border-left: 5px solid transparent;
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        color: #ffffff !important;       
        font-size: 1.25rem !important;   
        font-weight: 700 !important;     
        margin: 0 !important;
        padding-left: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: #475569 !important;
        border-left: 5px solid #38bdf8 !important; 
        transform: translateX(4px);
    }
    
    /* --- RADIO ICON BUTTON SCALING --- */
    [data-testid="stSidebar"] div[data-testid="stRadio"] label div:first-child div {
        border-color: #38bdf8 !important; 
        transform: scale(1.25); 
    }

    [data-testid="stSidebar"] div[data-testid="stRadio"] label input:checked + div div div {
        background-color: #38bdf8 !important;
    }
    
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.1rem !important;
        padding-top: 0px !important; /* Forces the choices up against the blue accent line border */
    }

    /* --- Long Red Predict Button Styling --- */
    div.stButton > button {
        background-color: #e53935 !important;
        color: white !important;
        border-color: #e53935 !important;
        font-weight: bold !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        border-radius: 0.5rem !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #d32f2f !important;
        border-color: #d32f2f !important;
        box-shadow: 0 4px 12px rgba(229,57,53,0.3) !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar Navigation ----
st.sidebar.markdown(
    """
    <p class="sidebar-title">🧭 Navigation Dashboard</p>
    <p class="sidebar-subtitle">Select Interface View:</p>
    """, 
    unsafe_allow_html=True
)

# Fixed implementation: passing a functional label name avoids layout warnings, while our custom CSS blocks layout height allocation
page = st.sidebar.radio(
    label="sidebar_navigation_menu", 
    options=["🔮 Insurance Advisor Form", "📊 System Accuracy Metrics"],
    label_visibility="collapsed" 
)

# ==========================================
# PAGE 1: INSURANCE ADVISOR
# ==========================================
if "Advisor" in page:
    st.header("🏥 Health Insurance Premium Advisor")
    st.markdown("---")

    # SECTION 1: DEMOGRAPHICS & LIFESTYLE
    st.subheader("📋 1. Demographics & Lifestyle Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        children = st.number_input("Children", min_value=0, max_value=10, value=0)
    with col2:
        occupation = st.selectbox("Occupation", ["Engineer", "Doctor", "Teacher", "Student", "Other"])
        annual_income = st.number_input("Annual income (USD)", min_value=0, value=50000, step=1000)
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    with col3:
        exercise_level = st.selectbox("Exercise level", ["None", "Light", "Moderate", "Heavy"])
        alcohol = st.number_input("Alcohol drinks/week", min_value=0, value=2)
        sleep = st.slider("Sleep hours/night", min_value=4.0, max_value=12.0, value=7.0, step=0.5)
        stress = st.slider("Stress level (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

    st.markdown("---")

    # SECTION 2: MEDICAL & HEALTH METRICS
    st.subheader("🩺 2. Clinical & Medical History")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=22.5, step=0.1)
        bp = st.text_input("Blood pressure (systolic/diastolic)", value="120/80")
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=190)
    with m_col2:
        smoker = st.selectbox("Smoker", ["No", "Yes"])
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        chronic_diseases = st.number_input("Chronic disease count", min_value=0, max_value=10, value=0)
    with m_col3:
        doctor_visits = st.number_input("Doctor visits/yr", min_value=0, max_value=50, value=2)
        hospitalizations = st.number_input("Hospitalizations last yr", min_value=0, max_value=10, value=0)
    
    st.markdown("---")
    
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        predict_btn = st.button("🔮 Predict Insurance Premium", use_container_width=True)

    if predict_btn:
        exercise_level_map = {"None": "Low", "Light": "Low", "Moderate": "Moderate", "Heavy": "High"}
        mapped_exercise_level = exercise_level_map.get(exercise_level, "Moderate")
        payload = {
            "age": int(age), "gender": gender, "bmi": float(bmi), "children": int(children),
            "smoker": smoker, "region": region, "occupation": occupation, "annual_income_usd": int(annual_income),
            "exercise_level": mapped_exercise_level, "chronic_diseases": int(chronic_diseases),
            "doctor_visits_per_year": int(doctor_visits), "hospitalizations_last_year": int(hospitalizations),
            "alcohol_consumption_per_week": int(alcohol), "blood_pressure": bp, "diabetes": diabetes,
            "cholesterol": int(cholesterol), "sleep_hours": float(sleep), "stress_level": float(stress),
            "marital_status": marital_status,
        }
        data = None
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(API_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Prediction successful (via API)!")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
        except Exception as api_exc:
            st.info("⚠️ FastAPI server is offline. Running local prediction fallback...")
            try:
                data = predict_local(payload)
                st.success("✅ Prediction successful (via local fallback)!")
            except Exception as local_exc:
                st.error(f"Failed to run local prediction: {local_exc}")
                st.error(f"FastAPI connection error was: {api_exc}")

        if data is not None:
            with st.container(border=True):
                st.subheader("📈 Prediction Summary")
                
                insurance_type = data.get('predicted_insurance_type', 'Standard')
                premium_val = data.get('predicted_annual_premium', 0.0)
                
                # Dynamic color theme based on tier
                if insurance_type == "Basic":
                    text_color = "#10b981"  # Green
                    bg_color = "#ecfdf5"
                    border_color = "#a7f3d0"
                elif insurance_type == "Standard":
                    text_color = "#3b82f6"  # Blue
                    bg_color = "#eff6ff"
                    border_color = "#bfdbfe"
                else:  # Premium
                    text_color = "#8b5cf6"  # Purple/Indigo
                    bg_color = "#f5f3ff"
                    border_color = "#ddd6fe"

                st.markdown(
                    f'<div class="result-premium-title" style="color: {text_color} !important;">{insurance_type} Insurance Tier</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="result-premium-price" style="color: {text_color} !important; background-color: {bg_color} !important; border: 1px solid {border_color} !important;">${premium_val:,.2f} / year</div>',
                    unsafe_allow_html=True
                )
                
                confidence = data.get('tier_confidence', 1.0)
                boundaries = data.get('tier_boundaries', {})
                
                st.markdown(f'<div class="result-suggestion">💡 <b>Boundary Proximity Confidence:</b> {confidence*100:.1f}%</div>', unsafe_allow_html=True)
                
                if boundaries:
                    st.markdown("#### **Tier Thresholds**")
                    bound_data = {
                        "Tier": ["Basic", "Standard", "Premium"],
                        "Premium Range": [
                            "below $15,000",
                            "Between (15,000 - 30,000)",
                            "above $30,000"
                        ]
                    }
                    st.table(bound_data)

# ==========================================
# PAGE 2: ACCURACY METRICS
# ==========================================
elif "Accuracy" in page:
    st.header("📊 Model Performance & Accuracy Metrics")
    st.markdown("---")

    metrics = load_training_metrics(str(METRICS_PATH))

    if metrics is None:
        st.warning(
            f"No training metrics found at `{METRICS_PATH}`. "
            "Run `python -m src.train` to generate `metrics/training_metrics.json`."
        )
    else:
        st.caption(f"Source: `{METRICS_PATH.name}` (test split highlighted below)")

        st.subheader("Test Set Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² (Regression)", _format_r2(metrics["regression_test_r2"]))
        col2.metric("RMSE (Regression)", _format_usd(metrics["regression_test_rmse"]))
        col3.metric("MAE (Regression)", _format_usd(metrics["regression_test_mae"]))

        st.markdown("---")
        st.subheader("Regression — Premium Prediction")
        st.table(
            {
                "Split": ["Train", "Validation", "Test"],
                "R²": [
                    _format_r2(metrics["regression_train_r2"]),
                    _format_r2(metrics["regression_val_r2"]),
                    _format_r2(metrics["regression_test_r2"]),
                ],
                "RMSE": [
                    _format_usd(metrics["regression_train_rmse"]),
                    _format_usd(metrics["regression_val_rmse"]),
                    _format_usd(metrics["regression_test_rmse"]),
                ],
                "MAE": [
                    _format_usd(metrics["regression_train_mae"]),
                    _format_usd(metrics["regression_val_mae"]),
                    _format_usd(metrics["regression_test_mae"]),
                ],
            }
        )

        st.markdown("---")
        st.info(
            "Metrics are from the latest training run (75% train / 10% validation / 15% test). "
            "Re-run training to refresh these values."
        )