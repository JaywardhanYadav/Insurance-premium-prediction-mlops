"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

InsuranceTier = Literal["Basic", "Standard", "Premium"]
GenderLiteral = Literal["Male", "Female", "M", "F"]
YesNoLiteral = Literal["Yes", "No"]
ExerciseLevel = Literal["Low", "Moderate", "High"]
RegionLiteral = Literal[
    "Northeast", "Northwest", "Central", "Southeast", "Southwest"
]
MaritalStatus = Literal["Single", "Married", "Divorced", "Widowed"]


class HealthMetricsRequest(BaseModel):
    """Raw user health metrics submitted for chained inference."""

    customer_id: Optional[str] = Field(
        default=None,
        description="Optional customer identifier for audit logging.",
        max_length=32,
    )
    age: int = Field(..., ge=18, le=100, description="Customer age in years.")
    gender: GenderLiteral = Field(..., description="Customer gender.")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index.")
    children: int = Field(..., ge=0, le=10, description="Number of dependents.")
    smoker: YesNoLiteral = Field(..., description="Smoking status.")
    region: RegionLiteral = Field(..., description="Geographic region.")
    occupation: str = Field(..., min_length=2, max_length=100)
    annual_income_usd: int = Field(..., ge=0, le=1_000_000)
    exercise_level: ExerciseLevel
    chronic_diseases: int = Field(..., ge=0, le=10)
    doctor_visits_per_year: int = Field(..., ge=0, le=50)
    hospitalizations_last_year: int = Field(..., ge=0, le=20)
    alcohol_consumption_per_week: int = Field(..., ge=0, le=50)
    blood_pressure: str = Field(
        ...,
        description="Blood pressure in 'systolic/diastolic' format, e.g. '149/87'.",
        examples=["120/80"],
    )
    diabetes: YesNoLiteral
    cholesterol: int = Field(..., ge=100, le=400)
    sleep_hours: float = Field(..., ge=3.0, le=12.0)
    stress_level: float = Field(..., ge=1.0, le=10.0)
    marital_status: MaritalStatus

    @field_validator("blood_pressure")
    @classmethod
    def validate_blood_pressure_format(cls, value: str) -> str:
        parts = value.strip().split("/")
        if len(parts) != 2:
            raise ValueError("blood_pressure must be in 'systolic/diastolic' format.")
        try:
            systolic = int(parts[0].strip())
            diastolic = int(parts[1].strip())
        except ValueError as exc:
            raise ValueError("blood_pressure components must be integers.") from exc
        if systolic <= diastolic:
            raise ValueError("Systolic value must exceed diastolic value.")
        return value.strip()

    model_config = {"json_schema_extra": {"title": "HealthMetricsRequest"}}


class PredictionResponse(BaseModel):
    """Joint chained inference response payload."""

    request_id: str
    customer_id: Optional[str] = None
    predicted_annual_premium: float = Field(
        ...,
        description="Estimated annual medical premium (USD) from XGBoost Regressor.",
    )
    predicted_insurance_type: InsuranceTier = Field(
        ...,
        description="Recommended insurance tier from XGBoost Classifier.",
    )
    classification_probabilities: Dict[str, float] = Field(
        ...,
        description="Class probability distribution for insurance tiers.",
    )
    model_version: Optional[str] = None
    latency_ms: float

    model_config = {"json_schema_extra": {"title": "PredictionResponse"}}


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str
    database_connected: bool
    models_loaded: bool
    mlflow_run_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard API error response."""

    detail: str
    error_type: str
