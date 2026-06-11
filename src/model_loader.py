"""MLflow model artifact loader for production inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from src.config_loader import get_project_root, load_config
from src.features import FeatureArtifacts
from src.inference import ChainedModelBundle

logger = logging.getLogger(__name__)


def _resolve_latest_run_id(experiment_name: str) -> str:
    """Find the most recent MLflow run ID for a given experiment."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"No runs found in experiment '{experiment_name}'. Train models first."
        )
    return str(runs.iloc[0]["run_id"])


def _reconstruct_feature_artifacts(
    preprocessor: ColumnTransformer,
    metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> FeatureArtifacts:
    """Rebuild FeatureArtifacts from logged MLflow metadata."""
    chained_name = config["training"]["chained_feature_name"]
    regression_names = metadata.get("regression_feature_names", [])
    classification_names = metadata.get("classification_feature_names", [])

    if not regression_names:
        try:
            regression_names = list(preprocessor.get_feature_names_out())
        except Exception:
            regression_names = []

    if not classification_names:
        classification_names = list(regression_names) + [chained_name]

    return FeatureArtifacts(
        preprocessor=preprocessor,
        feature_names=list(regression_names),
        regression_feature_names=list(regression_names),
        classification_feature_names=list(classification_names),
        chained_feature_name=chained_name,
    )


def load_chained_models(
    run_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ChainedModelBundle:
    """
    Load regressor, classifier, preprocessor, and label encoder from MLflow.

    Parameters
    ----------
    run_id:
        Optional explicit MLflow run ID. Defaults to latest run in experiment.
    """
    cfg = config or load_config()
    tracking_uri = cfg["paths"]["mlflow_tracking_uri"]
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_file = tracking_uri.replace("sqlite:///", "")
        tracking_uri = f"sqlite:///{(get_project_root() / db_file).as_posix()}"
    elif not tracking_uri.startswith(("sqlite:", "http:", "https:")):
        tracking_uri = str(get_project_root() / tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = cfg["mlflow"]["experiment_name"]
    resolved_run_id = run_id or _resolve_latest_run_id(experiment_name)
    logger.info("Loading models from MLflow run_id=%s", resolved_run_id)

    try:
        preprocessor = mlflow.sklearn.load_model(f"runs:/{resolved_run_id}/preprocessor")
        regressor = mlflow.xgboost.load_model(
            f"runs:/{resolved_run_id}/{cfg['mlflow']['model_names']['regressor']}"
        )
        classifier = mlflow.xgboost.load_model(
            f"runs:/{resolved_run_id}/{cfg['mlflow']['model_names']['classifier']}"
        )
        label_encoder: LabelEncoder = mlflow.sklearn.load_model(
            f"runs:/{resolved_run_id}/label_encoder"
        )

        metadata: Dict[str, Any] = {}
        try:
            meta_path = mlflow.artifacts.download_artifacts(
                run_id=resolved_run_id,
                artifact_path="feature_metadata.json",
            )
            with Path(meta_path).open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is not None:
                fallback = (
                    get_project_root()
                    / "mlruns"
                    / str(experiment.experiment_id)
                    / resolved_run_id
                    / "artifacts"
                    / "feature_metadata.json"
                )
                if fallback.exists():
                    with fallback.open("r", encoding="utf-8") as handle:
                        metadata = json.load(handle)
                else:
                    logger.warning(
                        "feature_metadata.json not found for run %s.", resolved_run_id
                    )

        feature_artifacts = _reconstruct_feature_artifacts(preprocessor, metadata, cfg)
        insurance_classes: List[str] = metadata.get(
            "insurance_classes", list(label_encoder.classes_)
        )

        if not isinstance(regressor, xgb.XGBRegressor):
            raise TypeError("Loaded regressor is not an XGBRegressor instance.")
        if not isinstance(classifier, xgb.XGBClassifier):
            raise TypeError("Loaded classifier is not an XGBClassifier instance.")

        return ChainedModelBundle(
            regressor=regressor,
            classifier=classifier,
            preprocessor_artifacts=feature_artifacts,
            label_encoder=label_encoder,
            insurance_classes=insurance_classes,
            mlflow_run_id=resolved_run_id,
            model_version=resolved_run_id,
        )
    except Exception as exc:
        logger.exception("Failed to load chained models from MLflow.")
        raise RuntimeError(f"Model loading failed: {exc}") from exc
