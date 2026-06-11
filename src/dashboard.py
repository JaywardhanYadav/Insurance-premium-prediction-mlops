"""Streamlit dashboard for insurance chained prediction."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001").rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

TIER_COLORS = {
    "Basic": "#4CAF50",
    "Standard": "#FF9800",
    "Premium": "#E53935",
}

TIER_DESCRIPTIONS = {
    "Basic": "Essential coverage for lower projected medical costs.",
    "Standard": "Balanced coverage for moderate projected medical costs.",
    "Premium": "Comprehensive coverage for elevated projected medical costs.",
}


def _post_prediction(payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    """Send async HTTP POST request to FastAPI prediction endpoint."""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(PREDICT_URL, json=payload)
        response.raise_for_status()
        return response.json()


def _get_health() -> Optional[Dict[str, Any]]:
    """Check API health status."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(HEALTH_URL)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        return None


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def main() -> None:
    """Render Streamlit insurance advisor UI."""
    st.set_page_config(
        page_title="Insurance Plan Advisor",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title("🏥 Insurance Plan Advisor")
    st.markdown(
        "Enter your health metrics to receive an **AI-powered annual premium estimate** "
        "and **recommended insurance tier** via our chained XGBoost pipeline."
    )

    with st.sidebar:
        st.header("API Status")
        health = _get_health()
        if health:
            status_color = "green" if health.get("status") == "healthy" else "orange"
            st.markdown(
                f":{status_color}[**{health.get('status', 'unknown').title()}**]"
            )
            st.write(f"Database: {'✅' if health.get('database_connected') else '❌'}")
            st.write(f"Models: {'✅' if health.get('models_loaded') else '❌'}")
            if health.get("mlflow_run_id"):
                st.caption(f"Model Run: `{health['mlflow_run_id'][:12]}...`")
        else:
            st.error(f"API unreachable at `{API_BASE_URL}`")

        st.divider()
        st.caption("Powered by FastAPI + XGBoost + MLflow")

    with st.form("health_metrics_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            customer_id = st.text_input("Customer ID (optional)", placeholder="MIC100001")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
            smoker = st.selectbox("Smoker", ["No", "Yes"])
        with col2:
            region = st.selectbox(
                "Region",
                ["Northeast", "Northwest", "Central", "Southeast", "Southwest"],
            )
            occupation = st.text_input("Occupation", value="Engineer")
            annual_income = st.number_input(
                "Annual Income (USD)", min_value=0, max_value=1_000_000, value=65000, step=1000
            )
            exercise_level = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])
            marital_status = st.selectbox(
                "Marital Status", ["Single", "Married", "Divorced", "Widowed"]
            )

        st.subheader("Health Metrics")
        col3, col4 = st.columns(2)
        with col3:
            blood_pressure = st.text_input("Blood Pressure", value="120/80", help="Format: systolic/diastolic")
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
            sleep_hours = st.number_input("Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.1)
        with col4:
            chronic_diseases = st.number_input("Chronic Diseases", min_value=0, max_value=10, value=0)
            doctor_visits = st.number_input("Doctor Visits / Year", min_value=0, max_value=50, value=2)
            hospitalizations = st.number_input(
                "Hospitalizations (Last Year)", min_value=0, max_value=20, value=0
            )
            alcohol = st.number_input(
                "Alcohol Consumption / Week", min_value=0, max_value=50, value=3
            )
            stress_level = st.number_input(
                "Stress Level (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1
            )

        submitted = st.form_submit_button("Get Recommendation", type="primary", use_container_width=True)

    if submitted:
        payload: Dict[str, Any] = {
            "customer_id": customer_id or None,
            "age": int(age),
            "gender": gender,
            "bmi": float(bmi),
            "children": int(children),
            "smoker": smoker,
            "region": region,
            "occupation": occupation,
            "annual_income_usd": int(annual_income),
            "exercise_level": exercise_level,
            "chronic_diseases": int(chronic_diseases),
            "doctor_visits_per_year": int(doctor_visits),
            "hospitalizations_last_year": int(hospitalizations),
            "alcohol_consumption_per_week": int(alcohol),
            "blood_pressure": blood_pressure,
            "diabetes": diabetes,
            "cholesterol": int(cholesterol),
            "sleep_hours": float(sleep_hours),
            "stress_level": float(stress_level),
            "marital_status": marital_status,
        }

        with st.spinner("Running chained inference pipeline..."):
            try:
                result = _post_prediction(payload)
            except httpx.HTTPStatusError as exc:
                detail = "Request failed."
                try:
                    detail = exc.response.json().get("detail", detail)
                except Exception:
                    pass
                st.error(f"API Error ({exc.response.status_code}): {detail}")
                return
            except httpx.HTTPError as exc:
                st.error(f"Connection error: {exc}")
                return

        st.success("Prediction complete!")
        tier = result["predicted_insurance_type"]
        premium = result["predicted_annual_premium"]
        tier_color = TIER_COLORS.get(tier, "#607D8B")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Estimated Annual Premium", _format_currency(premium))
        metric_col2.metric("Recommended Plan", tier)
        metric_col3.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")

        st.markdown(
            f"""
            <div style="
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 6px solid {tier_color};
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                margin: 1rem 0;
            ">
                <h3 style="margin:0; color:{tier_color};">{tier} Plan Recommended</h3>
                <p style="margin:0.5rem 0 0 0; color:#555;">
                    {TIER_DESCRIPTIONS.get(tier, '')}
                </p>
                <p style="margin:0.5rem 0 0 0; font-size:1.4rem; font-weight:bold;">
                    Projected Cost: {_format_currency(premium)} / year
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Classification Confidence")
        probabilities = result.get("classification_probabilities", {})
        if probabilities:
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_probs:
                st.progress(min(max(prob, 0.0), 1.0), text=f"{label}: {prob * 100:.1f}%")

        with st.expander("Request Details"):
            st.json(result)


if __name__ == "__main__":
    main()
