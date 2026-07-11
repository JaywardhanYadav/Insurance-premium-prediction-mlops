"""Inference orchestration for production scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import (
    FeatureArtifacts,
    derive_tier_from_premium,
    engineer_raw_features,
    transform_features,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Container for loaded MLflow artifacts used during inference."""

    regressor: Any
    preprocessor_artifacts: FeatureArtifacts
    tier_config: Dict[str, Any]
    mlflow_run_id: Optional[str] = None
    model_version: Optional[str] = None


class InferenceEngine:
    """Executes inference: Regressor -> Threshold-based Tier Derivation."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def _calculate_confidence(self, premium: float) -> float:
        """Calculate how confident the tier assignment is based on distance to nearest threshold."""
        thresholds = self.bundle.tier_config.get("thresholds", [15000, 30000])
        if not thresholds:
            return 1.0
        min_dist = min(abs(premium - t) for t in thresholds)
        # Scale: within $5000 of a boundary, confidence drops from 1.0 to 0.0
        return float(min(min_dist / 5000.0, 1.0))

    def predict_single(self, raw_record: Dict[str, Any]) -> Tuple[float, str, float, Dict[str, float]]:
        """
        Run inference on a single raw health metrics record.

        Returns
        -------
        Tuple[float, str, float, Dict[str, float]]
            (predicted_premium, predicted_insurance_type, tier_confidence, tier_boundaries)
        """
        try:
            input_df = pd.DataFrame([raw_record])
            engineered = engineer_raw_features(input_df)
            feature_matrix = transform_features(engineered, self.bundle.preprocessor_artifacts)

            premium_prediction = float(
                np.expm1(self.bundle.regressor.predict(feature_matrix)[0])
            )
            predicted_label = derive_tier_from_premium(premium_prediction, config=self.bundle.tier_config)
            confidence = self._calculate_confidence(premium_prediction)

            thresholds = self.bundle.tier_config.get("thresholds", [15000, 30000])
            labels = self.bundle.tier_config.get("labels", ["Basic", "Standard", "Premium"])
            boundaries = {
                f"{labels[i]} -> {labels[i+1]}": float(thresholds[i])
                for i in range(len(thresholds))
            }

            return premium_prediction, predicted_label, confidence, boundaries
        except Exception as exc:
            logger.exception("Inference failed.")
            raise RuntimeError(f"Inference failed: {exc}") from exc

    def predict_batch(
        self, raw_records: List[Dict[str, Any]]
    ) -> List[Tuple[float, str, float, Dict[str, float]]]:
        """
        Run inference on a batch of raw health metrics records.

        Parameters
        ----------
        raw_records:
            List of raw record dictionaries.

        Returns
        -------
        List[Tuple[float, str, float, Dict[str, float]]]
            List of (predicted_premium, predicted_insurance_type, tier_confidence, tier_boundaries)
        """
        try:
            input_df = pd.DataFrame(raw_records)
            engineered = engineer_raw_features(input_df)
            feature_matrix = transform_features(engineered, self.bundle.preprocessor_artifacts)

            premium_predictions = np.expm1(self.bundle.regressor.predict(feature_matrix))

            thresholds = self.bundle.tier_config.get("thresholds", [15000, 30000])
            labels = self.bundle.tier_config.get("labels", ["Basic", "Standard", "Premium"])
            boundaries = {
                f"{labels[i]} -> {labels[i+1]}": float(thresholds[i])
                for i in range(len(thresholds))
            }

            results = []
            for pred in premium_predictions:
                premium = float(pred)
                predicted_label = derive_tier_from_premium(premium, config=self.bundle.tier_config)
                confidence = self._calculate_confidence(premium)
                results.append((
                    premium,
                    predicted_label,
                    confidence,
                    boundaries,
                ))
            return results
        except Exception as exc:
            logger.exception("Batch inference failed.")
            raise RuntimeError(f"Batch inference failed: {exc}") from exc
