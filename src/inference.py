"""Chained inference orchestration for production scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from src.features import (
    FeatureArtifacts,
    append_chained_prediction,
    engineer_raw_features,
    transform_features,
)

logger = logging.getLogger(__name__)


@dataclass
class ChainedModelBundle:
    """Container for loaded MLflow artifacts used during inference."""

    regressor: xgb.XGBRegressor
    classifier: xgb.XGBClassifier
    preprocessor_artifacts: FeatureArtifacts
    label_encoder: LabelEncoder
    insurance_classes: List[str]
    mlflow_run_id: Optional[str] = None
    model_version: Optional[str] = None


class ChainedInferenceEngine:
    """Executes serial chained inference: Regressor -> Append -> Classifier."""

    def __init__(self, bundle: ChainedModelBundle) -> None:
        self.bundle = bundle

    def predict_single(self, raw_record: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
        """
        Run chained inference on a single raw health metrics record.

        Returns
        -------
        Tuple[float, str, Dict[str, float]]
            (predicted_premium, predicted_insurance_type, class_probabilities)
        """
        try:
            input_df = pd.DataFrame([raw_record])
            engineered = engineer_raw_features(input_df)
            feature_matrix = transform_features(engineered, self.bundle.preprocessor_artifacts)

            premium_prediction = float(
                self.bundle.regressor.predict(feature_matrix)[0]
            )
            chained_matrix = append_chained_prediction(
                feature_matrix, [premium_prediction]
            )

            class_proba = self.bundle.classifier.predict_proba(chained_matrix)[0]
            class_index = int(np.argmax(class_proba))
            predicted_label = self.bundle.label_encoder.inverse_transform([class_index])[0]

            probabilities = {
                str(label): float(prob)
                for label, prob in zip(self.bundle.insurance_classes, class_proba)
            }

            return premium_prediction, str(predicted_label), probabilities
        except Exception as exc:
            logger.exception("Chained inference failed.")
            raise RuntimeError(f"Chained inference failed: {exc}") from exc
