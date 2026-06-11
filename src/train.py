"""Training pipeline with feature chaining and MLflow experiment tracking."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config_loader import get_project_root, load_config
from src.database import DatabaseManager
from src.features import (
    FeatureArtifacts,
    append_chained_prediction,
    build_and_fit_preprocessor,
    prepare_training_frames,
    transform_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_training_dataframe(config: Dict[str, Any]) -> pd.DataFrame:
    """Load training data from MSSQL, falling back to local CSV if DB unavailable."""
    db_manager = DatabaseManager(config)
    try:
        if db_manager.health_check():
            return db_manager.fetch_training_data()
        logger.warning("MSSQL unavailable — loading training data from local CSV.")
    except Exception as exc:
        logger.warning("Database fetch failed (%s). Falling back to CSV.", exc)

    csv_path = get_project_root() / config["paths"]["raw_data"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found in database or at {csv_path}"
        )

    col_cfg = config["columns"]
    df = pd.read_csv(csv_path)
    rename_map = {
        col_cfg["source_regression"]: col_cfg["target_regression"],
        col_cfg["source_classification"]: col_cfg["target_classification"],
    }
    return df.rename(columns=rename_map)


def _compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": float(r2_score(y_true, y_pred)), "rmse": rmse}


def _compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _train_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> xgb.XGBRegressor:
    """Train XGBoost regressor with early stopping on validation set."""
    early_stopping = int(params.pop("early_stopping_rounds", 25))
    model = xgb.XGBRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    model.set_params(early_stopping_rounds=early_stopping)
    return model


def _train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    num_classes: int,
) -> xgb.XGBClassifier:
    """Train XGBoost classifier with early stopping on validation set."""
    early_stopping = int(params.pop("early_stopping_rounds", 20))
    model = xgb.XGBClassifier(
        **params,
        num_class=num_classes,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    model.set_params(early_stopping_rounds=early_stopping)
    return model


def _split_data(
    feature_frame: pd.DataFrame,
    y_regression: pd.Series,
    y_classification: pd.Series,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create stratified train/validation split without leakage."""
    train_cfg = config["training"]
    test_size = float(train_cfg["test_size"]) + float(train_cfg["validation_size"])

    if train_cfg.get("stratify_classification", True):
        stratify = y_classification
    else:
        stratify = None

    x_train, x_holdout, y_reg_train, y_reg_holdout, y_cls_train, y_cls_holdout = (
        train_test_split(
            feature_frame,
            y_regression,
            y_classification,
            test_size=test_size,
            random_state=config["project"]["random_seed"],
            stratify=stratify,
        )
    )

    relative_val_size = float(train_cfg["validation_size"]) / test_size
    if train_cfg.get("stratify_classification", True):
        stratify_holdout = y_cls_holdout
    else:
        stratify_holdout = None

    x_val, x_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = train_test_split(
        x_holdout,
        y_reg_holdout,
        y_cls_holdout,
        test_size=1.0 - relative_val_size,
        random_state=config["project"]["random_seed"],
        stratify=stratify_holdout,
    )
    return (
        x_train,
        x_val,
        y_reg_train,
        y_reg_val,
        y_cls_train,
        y_cls_val,
    )


def train_chained_pipeline(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute full chained training pipeline with MLflow logging.

    Feature chaining is leakage-safe:
    - Preprocessor fit on train only
    - Regressor trained on train, predicts train + val separately
    - Classifier trained on train with chained predictions from regressor
    """
    cfg = config or load_config()
    tracking_uri = cfg["paths"]["mlflow_tracking_uri"]
    mlruns_dir = get_project_root() / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_file = tracking_uri.replace("sqlite:///", "")
        tracking_uri = f"sqlite:///{(get_project_root() / db_file).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    raw_df = _load_training_dataframe(cfg)
    feature_frame, y_regression, y_classification = prepare_training_frames(raw_df, cfg)

    (
        x_train,
        x_val,
        y_reg_train,
        y_reg_val,
        y_cls_train,
        y_cls_val,
    ) = _split_data(feature_frame, y_regression, y_classification, cfg)

    artifacts: FeatureArtifacts = build_and_fit_preprocessor(x_train, cfg)
    x_train_matrix = transform_features(x_train, artifacts, cfg)
    x_val_matrix = transform_features(x_val, artifacts, cfg)

    reg_params = dict(cfg["training"]["regression"])
    # Log‑transform the premium target to stabilise variance
    y_reg_train_log = np.log1p(y_reg_train.to_numpy())
    y_reg_val_log = np.log1p(y_reg_val.to_numpy())

    regressor = _train_regressor(
        x_train_matrix,
        y_reg_train_log,
        x_val_matrix,
        y_reg_val_log,
        reg_params,
    )

    # Convert predictions back to original scale
    train_reg_preds = np.expm1(regressor.predict(x_train_matrix))
    val_reg_preds = np.expm1(regressor.predict(x_val_matrix))


    # Already back‑transformed above – keep same variable names
    # train_reg_preds and val_reg_preds contain predictions on original scale
    pass  # no further action needed

    reg_train_metrics = _compute_regression_metrics(
        y_reg_train.to_numpy(), train_reg_preds
    )
    reg_val_metrics = _compute_regression_metrics(y_reg_val.to_numpy(), val_reg_preds)

    x_train_chained = append_chained_prediction(x_train_matrix, train_reg_preds)
    x_val_chained = append_chained_prediction(x_val_matrix, val_reg_preds)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_cls_train.sort_values().unique())
    y_cls_train_encoded = label_encoder.transform(y_cls_train)
    y_cls_val_encoded = label_encoder.transform(y_cls_val)

    cls_params = dict(cfg["training"]["classification"])
    classifier = _train_classifier(
        x_train_chained,
        y_cls_train_encoded,
        x_val_chained,
        y_cls_val_encoded,
        cls_params,
        num_classes=len(label_encoder.classes_),
    )

    train_cls_preds = label_encoder.inverse_transform(
        classifier.predict(x_train_chained)
    )
    val_cls_preds = label_encoder.inverse_transform(classifier.predict(x_val_chained))

    cls_train_metrics = _compute_classification_metrics(y_cls_train, train_cls_preds)
    cls_val_metrics = _compute_classification_metrics(y_cls_val, val_cls_preds)

    processed_path = get_project_root() / cfg["paths"]["processed_data"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_parquet(processed_path, index=False)

    metrics_payload = {
        "regression_train_r2": reg_train_metrics["r2"],
        "regression_train_rmse": reg_train_metrics["rmse"],
        "regression_val_r2": reg_val_metrics["r2"],
        "regression_val_rmse": reg_val_metrics["rmse"],
        "classification_train_accuracy": cls_train_metrics["accuracy"],
        "classification_train_f1_weighted": cls_train_metrics["f1_weighted"],
        "classification_val_accuracy": cls_val_metrics["accuracy"],
        "classification_val_f1_weighted": cls_val_metrics["f1_weighted"],
    }

    metrics_dir = get_project_root() / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    run_name = f"{cfg['mlflow']['run_name_prefix']}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "regressor_objective": cfg["training"]["regression"]["objective"],
                "classifier_objective": cfg["training"]["classification"]["objective"],
                "chained_feature": artifacts.chained_feature_name,
                "train_rows": len(x_train),
                "val_rows": len(x_val),
            }
        )
        mlflow.log_params(
            {f"reg_{k}": v for k, v in cfg["training"]["regression"].items()}
        )
        mlflow.log_params(
            {f"cls_{k}": v for k, v in cfg["training"]["classification"].items()}
        )

        for metric_name, metric_value in metrics_payload.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(artifacts.preprocessor, "preprocessor")
        mlflow.xgboost.log_model(regressor, cfg["mlflow"]["model_names"]["regressor"])
        mlflow.xgboost.log_model(classifier, cfg["mlflow"]["model_names"]["classifier"])
        mlflow.sklearn.log_model(label_encoder, "label_encoder")

        mlflow.log_dict(
            {
                "regression_feature_names": artifacts.regression_feature_names,
                "classification_feature_names": artifacts.classification_feature_names,
                "insurance_classes": list(label_encoder.classes_),
            },
            "feature_metadata.json",
        )

        run_id = run.info.run_id
        logger.info("MLflow run completed: %s", run_id)

    summary = {
        "run_id": run_id,
        "metrics": metrics_payload,
        "processed_data": str(processed_path),
    }
    logger.info("Training complete. Metrics: %s", metrics_payload)
    return summary


def main() -> None:
    """CLI entrypoint for training pipeline."""
    try:
        result = train_chained_pipeline()
        print(json.dumps(result, indent=2))
    except Exception as exc:
        logger.exception("Training pipeline failed.")
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
