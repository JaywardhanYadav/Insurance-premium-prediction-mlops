"""Feature engineering pipeline for insurance prediction models."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config_loader import load_config

logger = logging.getLogger(__name__)

_BP_PATTERN = re.compile(r"^\s*(\d{2,3})\s*/\s*(\d{2,3})\s*$")

# ---------------------------------------------------------------------------
# Derived-feature names that are created *during* engineer_raw_features and
# therefore must NOT be expected in the raw input nor winsorised.
# ---------------------------------------------------------------------------
_DERIVED_FEATURES: frozenset[str] = frozenset(
    {
        "bp_systolic",
        "bp_diastolic",
        "pulse_pressure",
        "age_smoker",
        "bmi_exercise",
        "bmi_category",
        "age_group",
        "bp_category",
        "income_per_child",
        "risk_score",
        "smoker_cholesterol",
        "age_chronic",
        "stress_sleep",
        "stress_squared",
        "cholesterol_squared",
        "medical_visits_total",
        "age_bmi",
        "smoker_bmi",
        "smoker_obese",
        "hypertension",
        "smoker_hypertension",
        "age_smoker_bmi",
        "high_risk_chronic",
        "smoker_obese_interaction",
        "income_age_ratio",
    }
)


# =========================================================================
# Data-class for fitted preprocessing artefacts
# =========================================================================
@dataclass
class FeatureArtifacts:
    """Container for fitted preprocessing artifacts and feature metadata.

    Attributes
    ----------
    preprocessor : ColumnTransformer
        Fitted sklearn column transformer.
    feature_names : List[str]
        All output feature names produced by the preprocessor.
    """

    preprocessor: ColumnTransformer
    feature_names: List[str]


# =========================================================================
# Blood-pressure parsing
# =========================================================================
def parse_blood_pressure(bp_value: Any) -> Tuple[float, float]:
    """Parse blood pressure string (e.g., ``'149/87'``) into systolic and
    diastolic components.

    Parameters
    ----------
    bp_value:
        Blood pressure in ``'systolic/diastolic'`` format.

    Returns
    -------
    Tuple[float, float]
        ``(bp_systolic, bp_diastolic)``

    Raises
    ------
    ValueError
        If *bp_value* is null, malformed, or systolic ≤ diastolic.
    """
    if pd.isna(bp_value):
        raise ValueError("blood_pressure cannot be null.")

    text_value = str(bp_value).strip()
    match = _BP_PATTERN.match(text_value)
    if not match:
        raise ValueError(
            f"Invalid blood_pressure format '{bp_value}'. "
            "Expected 'systolic/diastolic'."
        )

    systolic = float(match.group(1))
    diastolic = float(match.group(2))
    if systolic <= diastolic:
        raise ValueError(
            f"Systolic ({systolic}) must be greater than diastolic ({diastolic})."
        )
    return systolic, diastolic


# =========================================================================
# Binary encoding helper
# =========================================================================
def _encode_binary_series(series: pd.Series, column_name: str) -> pd.Series:
    """Convert binary categorical columns to integer ``0`` / ``1``.

    Parameters
    ----------
    series:
        Raw column values (string-like).
    column_name:
        Column name — used to select the correct mapping (special-cased for
        ``gender``).

    Returns
    -------
    pd.Series
        Integer-encoded series.

    Raises
    ------
    ValueError
        If any value cannot be mapped.
    """
    normalized = series.astype(str).str.strip().str.lower()
    if column_name == "gender":
        mapping: Dict[str, int] = {"male": 1, "female": 0, "m": 1, "f": 0}
    else:
        mapping = {
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }

    encoded = normalized.map(mapping)
    if encoded.isna().any():
        invalid = series[encoded.isna()].unique().tolist()
        raise ValueError(f"Unrecognized values in '{column_name}': {invalid}")
    return encoded.astype(int)


# =========================================================================
# Tier derivation helper
# =========================================================================
def derive_tier_from_premium(premium: float, config: Optional[Dict[str, Any]] = None) -> str:
    """Map predicted premium to insurance tier using configured thresholds."""
    cfg = config or load_config()
    if "training" in cfg:
        tier_cfg = cfg["training"]["tier_derivation"]
    else:
        tier_cfg = cfg
    thresholds = tier_cfg["thresholds"]
    labels = tier_cfg["labels"]

    for val, label in zip(thresholds, labels):
        if premium < val:
            return label
    return labels[-1]


# =========================================================================
# Feature engineering
# =========================================================================
def engineer_raw_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Apply domain feature engineering on raw customer records.

    Performs the following transformations (in order):

    1. Parse ``blood_pressure`` → ``bp_systolic``, ``bp_diastolic``,
       ``pulse_pressure``.
    2. Binary-encode ``smoker``, ``diabetes``, ``gender``.
    3. Ordinal-encode ``exercise_level`` (Low → 1, Moderate → 2, High → 3).
    4. Cast and winsorise raw numeric columns.
    5. Create interaction / polynomial / binned features.
    6. Validate categorical columns.

    Parameters
    ----------
    df:
        Raw input dataframe (one row per customer).
    config:
        Pre-loaded configuration dict.  Loaded from disk when ``None``.

    Returns
    -------
    pd.DataFrame
        Engineered dataframe ready for preprocessing.
    """
    cfg = config or load_config()
    feature_cfg = cfg["features"]
    fe_cfg = cfg.get("feature_engineering", {})
    wins_q: List[float] = fe_cfg.get("winsorise_quantiles", [0.01, 0.99])
    working = df.copy()

    # ---- 1. Blood pressure parsing ------------------------------------
    if "blood_pressure" not in working.columns:
        raise KeyError("Required column 'blood_pressure' is missing from input data.")

    bp_parsed = working["blood_pressure"].apply(parse_blood_pressure)
    working["bp_systolic"] = bp_parsed.apply(lambda pair: pair[0])
    working["bp_diastolic"] = bp_parsed.apply(lambda pair: pair[1])
    working["pulse_pressure"] = working["bp_systolic"] - working["bp_diastolic"]

    # Log a single aggregated warning for out-of-range blood pressure values to avoid console flood
    invalid_bp = ~((working["bp_diastolic"] >= 70) & (working["bp_diastolic"] <= 140) & 
                   (working["bp_systolic"] >= 90) & (working["bp_systolic"] <= 220))
    num_invalid = invalid_bp.sum()
    if num_invalid > 0:
        logger.warning(
            "Found %d blood pressure record(s) with values outside the typical range (systolic: 90-220, diastolic: 70-140).",
            num_invalid
        )

    # ---- 2. Binary columns --------------------------------------------
    for column in feature_cfg["binary_columns"]:
        if column not in working.columns:
            raise KeyError(f"Required binary column '{column}' is missing.")
        working[column] = _encode_binary_series(working[column], column)

    # ---- 3. Exercise level (ordinal) ----------------------------------
    if "exercise_level" in working.columns:
        if working["exercise_level"].dtype == object:
            level_map: Dict[str, int] = {"Low": 1, "Moderate": 2, "High": 3}
            working["exercise_level"] = (
                working["exercise_level"].map(level_map).astype(int)
            )
        else:
            working["exercise_level"] = pd.to_numeric(
                working["exercise_level"], errors="raise"
            )

    # ---- 4. Cast and winsorise raw numeric columns --------------------
    # Only winsorise columns that are NOT derived during this step.
    required_numeric = [
        col
        for col in feature_cfg["numeric_columns"]
        if col not in _DERIVED_FEATURES
    ]
    for column in required_numeric:
        if column not in working.columns:
            raise KeyError(f"Required numeric column '{column}' is missing.")
        working[column] = pd.to_numeric(working[column], errors="raise")
        lower_q, upper_q = wins_q
        lower = working[column].quantile(lower_q)
        upper = working[column].quantile(upper_q)
        working[column] = working[column].clip(lower, upper)

    # ---- 5. Interaction / polynomial / binned features ----------------

    # 5a. Original interaction features
    if "age" in working.columns and "smoker" in working.columns:
        working["age_smoker"] = working["age"] * working["smoker"]
    if "bmi" in working.columns and "exercise_level" in working.columns:
        working["bmi_exercise"] = working["bmi"] * working["exercise_level"]

    # 5b. BMI category (ordinal bins)
    if "bmi" in working.columns:
        working["bmi_category"] = pd.cut(
            working["bmi"],
            bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
            labels=[0, 1, 2, 3],
            right=False,
        ).astype(int)

    # 5c. Age group (ordinal bins)
    if "age" in working.columns:
        working["age_group"] = pd.cut(
            working["age"],
            bins=[-np.inf, 31, 51, np.inf],
            labels=[0, 1, 2],
            right=False,
        ).astype(int)

    # 5d. Blood-pressure category
    working["bp_category"] = pd.cut(
        working["bp_systolic"],
        bins=[-np.inf, 120, 130, 140, np.inf],
        labels=[0, 1, 2, 3],
        right=False,
    ).astype(int)

    # 5e. Income per child (with +1 to avoid division by zero)
    if "annual_income_usd" in working.columns and "children" in working.columns:
        working["income_per_child"] = working["annual_income_usd"] / (
            working["children"] + 1
        )

    # 5f. Risk score (composite clinical risk indicator)
    if all(
        c in working.columns
        for c in [
            "smoker",
            "chronic_diseases",
            "hospitalizations_last_year",
            "bmi_category",
            "bp_category",
        ]
    ):
        working["risk_score"] = (
            working["smoker"] * 3
            + working["chronic_diseases"].clip(upper=3)
            + working["hospitalizations_last_year"].clip(upper=2)
            + (working["bmi_category"] >= 3).astype(int)
            + (working["bp_category"] >= 2).astype(int)
        )

    # 5g. Smoker × cholesterol interaction
    if "smoker" in working.columns and "cholesterol" in working.columns:
        working["smoker_cholesterol"] = working["smoker"] * working["cholesterol"]

    # 5h. Age × chronic-diseases interaction
    if "age" in working.columns and "chronic_diseases" in working.columns:
        working["age_chronic"] = working["age"] * working["chronic_diseases"]

    # 5i. Stress-sleep compound risk
    if "stress_level" in working.columns and "sleep_hours" in working.columns:
        working["stress_sleep"] = working["stress_level"] * (
            10 - working["sleep_hours"]
        )

    # 5j. Polynomial features
    if "stress_level" in working.columns:
        working["stress_squared"] = working["stress_level"] ** 2
    if "cholesterol" in working.columns:
        working["cholesterol_squared"] = working["cholesterol"] ** 2

    # 5k. Medical visits total (hospitalisations weighted ×5)
    if (
        "doctor_visits_per_year" in working.columns
        and "hospitalizations_last_year" in working.columns
    ):
        working["medical_visits_total"] = (
            working["doctor_visits_per_year"]
            + working["hospitalizations_last_year"] * 5
        )

    # 5l. Age × BMI interaction
    if "age" in working.columns and "bmi" in working.columns:
        working["age_bmi"] = working["age"] * working["bmi"]

    # 5m. Smoker × BMI and Smoker × Obesity
    if "smoker" in working.columns and "bmi" in working.columns:
        working["smoker_bmi"] = working["smoker"] * working["bmi"]
        working["smoker_obese"] = working["smoker"] * (working["bmi"] >= 30.0).astype(int)

    # 5n. Hypertension flag (systolic >= 130 or diastolic >= 80)
    if "bp_systolic" in working.columns and "bp_diastolic" in working.columns:
        working["hypertension"] = ((working["bp_systolic"] >= 130.0) | (working["bp_diastolic"] >= 80.0)).astype(int)

    # 5o. Smoker × Hypertension interaction
    if "smoker" in working.columns and "hypertension" in working.columns:
        working["smoker_hypertension"] = working["smoker"] * working["hypertension"]

    # 5p. Triple threat: Age × Smoker × BMI
    if "age" in working.columns and "smoker" in working.columns and "bmi" in working.columns:
        working["age_smoker_bmi"] = working["age"] * working["smoker"] * working["bmi"]

    # 5q. High Risk Chronic Condition Indicator (Age >= 50 and chronic diseases >= 2)
    if "age" in working.columns and "chronic_diseases" in working.columns:
        working["high_risk_chronic"] = ((working["chronic_diseases"] >= 2) & (working["age"] >= 50.0)).astype(int)

    # 5r. Smoker × Obese interaction
    if "smoker" in working.columns and "bmi" in working.columns:
        working["smoker_obese_interaction"] = (working["smoker"] * (working["bmi"] >= 30.0)).astype(int)

    # 5s. Income to Age ratio
    if "annual_income_usd" in working.columns and "age" in working.columns:
        working["income_age_ratio"] = working["annual_income_usd"] / (working["age"] + 1.0)

    # ---- 6. Validate categorical columns ------------------------------
    for column in feature_cfg["categorical_columns"]:
        if column not in working.columns:
            raise KeyError(f"Required categorical column '{column}' is missing.")
        working[column] = working[column].astype(str).str.strip()

    return working


# =========================================================================
# Sklearn preprocessing scaffold
# =========================================================================
def _build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    """Construct sklearn preprocessing pipeline for numeric and categorical
    features.

    Parameters
    ----------
    numeric_columns:
        List of numeric column names for the ``StandardScaler`` branch.
    categorical_columns:
        List of categorical column names for the ``OneHotEncoder`` branch.

    Returns
    -------
    ColumnTransformer
        Un-fitted transformer ready for ``.fit()``.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_columns)),
            ("cat", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _extract_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Return output feature names from a fitted ``ColumnTransformer``.

    Falls back to manual introspection when ``get_feature_names_out`` is
    unavailable (older sklearn versions).
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names: List[str] = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out(columns))
            else:
                names.extend(columns)
        return names


# =========================================================================
# Build, fit, and transform
# =========================================================================
def build_and_fit_preprocessor(
    train_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> FeatureArtifacts:
    """Fit preprocessing pipeline on training data only (prevents leakage).

    Parameters
    ----------
    train_df:
        Engineered training features **without** target columns.
    config:
        Pre-loaded configuration dict.  Loaded from disk when ``None``.

    Returns
    -------
    FeatureArtifacts
        Fitted preprocessor and feature name metadata.
    """
    cfg = config or load_config()
    feature_cfg = cfg["features"]

    numeric_cols: List[str] = list(feature_cfg["numeric_columns"])
    categorical_cols: List[str] = list(feature_cfg["categorical_columns"])

    # ---- Fit ColumnTransformer ----------------------------------------
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(train_df[numeric_cols + categorical_cols])

    base_feature_names = _extract_feature_names(preprocessor)

    return FeatureArtifacts(
        preprocessor=preprocessor,
        feature_names=base_feature_names,
    )


def transform_features(
    df: pd.DataFrame,
    artifacts: FeatureArtifacts,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Transform engineered dataframe into model-ready feature matrix.

    Parameters
    ----------
    df:
        Engineered dataframe (output of :func:`engineer_raw_features`).
    artifacts:
        Fitted preprocessing artifacts from
        :func:`build_and_fit_preprocessor`.
    config:
        Pre-loaded configuration dict.  Loaded from disk when ``None``.

    Returns
    -------
    np.ndarray
        2-D float64 feature matrix ready for model input.
    """
    cfg = config or load_config()
    feature_cfg = cfg["features"]
    input_cols = feature_cfg["numeric_columns"] + feature_cfg["categorical_columns"]
    matrix = artifacts.preprocessor.transform(df[input_cols])
    return np.asarray(matrix, dtype=np.float64)


# =========================================================================
# Training frame preparation
# =========================================================================
def prepare_training_frames(
    raw_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Engineer features and extract regression targets.

    Parameters
    ----------
    raw_df:
        Raw dataframe straight from ingestion (CSV / database).
    config:
        Pre-loaded configuration dict.  Loaded from disk when ``None``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        ``(engineered_features, premium_target)``

    Raises
    ------
    KeyError
        If expected target column is missing.
    """
    cfg = config or load_config()
    col_cfg = cfg["columns"]
    engineered = engineer_raw_features(raw_df, cfg)

    premium_col: str = col_cfg["target_regression"]

    if premium_col not in engineered.columns:
        raise KeyError(f"Regression target column '{premium_col}' not found.")

    y_regression = pd.to_numeric(engineered[premium_col], errors="raise")

    # Drop non-feature columns (IDs, targets, raw source columns)
    drop_cols = {
        col_cfg.get("customer_id", "customer_id"),
        premium_col,
        "blood_pressure",
        col_cfg.get("source_regression", ""),
        "insurance_plan",
        "insurance_type",
    }
    drop_cols.discard("")
    feature_frame = engineered.drop(
        columns=[col for col in drop_cols if col in engineered.columns],
        errors="ignore",
    )

    return feature_frame, y_regression
