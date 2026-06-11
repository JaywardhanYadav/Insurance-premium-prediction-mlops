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


@dataclass(frozen=True)
class FeatureArtifacts:
    """Container for fitted preprocessing artifacts and feature metadata."""

    preprocessor: ColumnTransformer
    feature_names: List[str]
    regression_feature_names: List[str]
    classification_feature_names: List[str]
    chained_feature_name: str


def parse_blood_pressure(bp_value: Any) -> Tuple[float, float]:
    """
    Parse blood pressure string (e.g., '149/87') into systolic and diastolic.

    Parameters
    ----------
    bp_value:
        Blood pressure in 'systolic/diastolic' format.

    Returns
    -------
    Tuple[float, float]
        (bp_systolic, bp_diastolic)
    """
    if pd.isna(bp_value):
        raise ValueError("blood_pressure cannot be null.")

    text_value = str(bp_value).strip()
    match = _BP_PATTERN.match(text_value)
    if not match:
        raise ValueError(
            f"Invalid blood_pressure format '{bp_value}'. Expected 'systolic/diastolic'."
        )

    systolic = float(match.group(1))
    diastolic = float(match.group(2))
    if not (70 <= diastolic <= 140 and 90 <= systolic <= 220):
        logger.warning(
            "Blood pressure values outside typical range: %s/%s", systolic, diastolic
        )
    if systolic <= diastolic:
        raise ValueError(
            f"Systolic ({systolic}) must be greater than diastolic ({diastolic})."
        )
    return systolic, diastolic


def _encode_binary_series(series: pd.Series, column_name: str) -> pd.Series:
    """Convert binary categorical columns to integer 0/1."""
    normalized = series.astype(str).str.strip().str.lower()
    if column_name == "gender":
        mapping = {"male": 1, "female": 0, "m": 1, "f": 0}
    else:
        mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}

    encoded = normalized.map(mapping)
    if encoded.isna().any():
        invalid = series[encoded.isna()].unique().tolist()
        raise ValueError(f"Unrecognized values in '{column_name}': {invalid}")
    return encoded.astype(int)


def engineer_raw_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Apply domain feature engineering on raw customer records.

    Creates bp_systolic, bp_diastolic, pulse_pressure, and binary encodings.
    """
    cfg = config or load_config()
    feature_cfg = cfg["features"]
    working = df.copy()

    if "blood_pressure" not in working.columns:
        raise KeyError("Required column 'blood_pressure' is missing from input data.")

    bp_parsed = working["blood_pressure"].apply(parse_blood_pressure)
    working["bp_systolic"] = bp_parsed.apply(lambda pair: pair[0])
    working["bp_diastolic"] = bp_parsed.apply(lambda pair: pair[1])
    working["pulse_pressure"] = working["bp_systolic"] - working["bp_diastolic"]

    for column in feature_cfg["binary_columns"]:
        if column not in working.columns:
            raise KeyError(f"Required binary column '{column}' is missing.")
        working[column] = _encode_binary_series(working[column], column)

    # Convert exercise_level categorical to numeric scale
    if "exercise_level" in working.columns:
        if working["exercise_level"].dtype == object:
            level_map = {"Low": 1, "Moderate": 2, "High": 3}
            working["exercise_level"] = working["exercise_level"].map(level_map).astype(int)
        else:
            # ensure numeric type
            working["exercise_level"] = pd.to_numeric(working["exercise_level"], errors="raise")

    required_numeric = [
        col
        for col in feature_cfg["numeric_columns"]
        if col not in {"bp_systolic", "bp_diastolic", "pulse_pressure"}
    ]
    for column in required_numeric:
        if column not in working.columns:
            raise KeyError(f"Required numeric column '{column}' is missing.")
        working[column] = pd.to_numeric(working[column], errors="raise")

    # Interaction features (new)
    if "age" in working.columns and "smoker" in working.columns:
        working["age_smoker"] = working["age"] * working["smoker"]
    if "bmi" in working.columns and "exercise_level" in working.columns:
        working["bmi_exercise"] = working["bmi"] * working["exercise_level"]

    for column in feature_cfg["categorical_columns"]:
        if column not in working.columns:
            raise KeyError(f"Required categorical column '{column}' is missing.")
        working[column] = working[column].astype(str).str.strip()

    return working


def _build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    """Construct sklearn preprocessing pipeline for numeric and categorical features."""
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
    """Return output feature names from a fitted ColumnTransformer."""
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


def build_and_fit_preprocessor(
    train_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> FeatureArtifacts:
    """
    Fit preprocessing pipeline on training data only (prevents leakage).

    Parameters
    ----------
    train_df:
        Engineered training features without target columns or chained features.

    Returns
    -------
    FeatureArtifacts
        Fitted preprocessor and feature name metadata.
    """
    cfg = config or load_config()
    feature_cfg = cfg["features"]
    chained_name = cfg["training"]["chained_feature_name"]

    numeric_cols = list(feature_cfg["numeric_columns"])
    categorical_cols = list(feature_cfg["categorical_columns"])

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(train_df[numeric_cols + categorical_cols])

    base_feature_names = _extract_feature_names(preprocessor)
    regression_feature_names = list(base_feature_names)
    classification_feature_names = list(base_feature_names) + [chained_name]

    return FeatureArtifacts(
        preprocessor=preprocessor,
        feature_names=base_feature_names,
        regression_feature_names=regression_feature_names,
        classification_feature_names=classification_feature_names,
        chained_feature_name=chained_name,
    )


def transform_features(
    df: pd.DataFrame,
    artifacts: FeatureArtifacts,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Transform engineered dataframe into model-ready feature matrix."""
    cfg = config or load_config()
    feature_cfg = cfg["features"]
    input_cols = feature_cfg["numeric_columns"] + feature_cfg["categorical_columns"]
    matrix = artifacts.preprocessor.transform(df[input_cols])
    return np.asarray(matrix, dtype=np.float64)


def append_chained_prediction(
    feature_matrix: np.ndarray,
    predicted_premiums: Sequence[float],
) -> np.ndarray:
    """
    Append regressor predictions as chained feature for classifier input.

    This must only use out-of-fold or holdout predictions during training
    to avoid target leakage. At inference time, use in-sample regressor output.
    """
    premiums = np.asarray(predicted_premiums, dtype=np.float64).reshape(-1, 1)
    if premiums.shape[0] != feature_matrix.shape[0]:
        raise ValueError(
            "predicted_premiums length must match feature_matrix row count "
            f"({premiums.shape[0]} != {feature_matrix.shape[0]})."
        )
    return np.hstack([feature_matrix, premiums])


def prepare_training_frames(
    raw_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Engineer features and extract regression/classification targets.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.Series]
        (engineered_features, premium_target, insurance_type_target)
    """
    cfg = config or load_config()
    col_cfg = cfg["columns"]
    engineered = engineer_raw_features(raw_df, cfg)

    premium_col = col_cfg["target_regression"]
    insurance_col = col_cfg["target_classification"]

    if premium_col not in engineered.columns:
        raise KeyError(f"Regression target column '{premium_col}' not found.")
    if insurance_col not in engineered.columns:
        raise KeyError(f"Classification target column '{insurance_col}' not found.")

    y_regression = pd.to_numeric(engineered[premium_col], errors="raise")
    y_classification = engineered[insurance_col].astype(str).str.strip()

    # Map legacy/alternate tier labels to canonical three-class schema
    tier_normalization = {"Gold": "Premium"}
    y_classification = y_classification.replace(tier_normalization)
    allowed_tiers = set(cfg["features"]["insurance_type_classes"])
    unknown_tiers = set(y_classification.unique()) - allowed_tiers
    if unknown_tiers:
        raise ValueError(
            f"Unrecognized insurance tiers: {sorted(unknown_tiers)}. "
            f"Expected one of {sorted(allowed_tiers)}."
        )

    drop_cols = {
        col_cfg.get("customer_id", "customer_id"),
        premium_col,
        insurance_col,
        "blood_pressure",
        col_cfg.get("source_regression", ""),
        col_cfg.get("source_classification", ""),
    }
    drop_cols.discard("")
    feature_frame = engineered.drop(
        columns=[col for col in drop_cols if col in engineered.columns],
        errors="ignore",
    )
    return feature_frame, y_regression, y_classification
