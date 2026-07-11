"""Training pipeline for regression with MLflow tracking.

This module implements a regression prediction pipeline:
  1. XGBRegressor predicts ``annual_medical_premium`` (log-transformed regression).
  2. All artefacts, metrics, and models are logged to MLflow.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add project root to sys.path to allow imports when running directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgb
from src.ensemble import EnsembleRegressor
from src.config_loader import get_project_root, load_config
from src.database import DatabaseManager
from src.features import (
    FeatureArtifacts,
    build_and_fit_preprocessor,
    prepare_training_frames,
    transform_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


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
        raise FileNotFoundError(f"Training data not found in database or at {csv_path}")
    col_cfg = config["columns"]
    df = pd.read_csv(csv_path)
    rename_map = {
        col_cfg["source_regression"]: col_cfg["target_regression"],
    }
    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics: R², RMSE, MAE, MSE.

    Parameters
    ----------
    y_true:
        Ground-truth values (original scale, **not** log-transformed).
    y_pred:
        Predicted values (original scale).

    Returns
    -------
    Dict[str, float]
        Dictionary keyed by metric name.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
    }


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------


def _train_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> xgb.XGBRegressor:
    """Train an XGBoost regressor with early stopping on a validation set.

    Parameters
    ----------
    x_train:
        Training feature matrix.
    y_train:
        Training targets (log-transformed premiums).
    x_val:
        Validation feature matrix.
    y_val:
        Validation targets (log-transformed premiums).
    params:
        XGBRegressor parameters. ``early_stopping_rounds`` is popped and
        forwarded to the constructor separately.

    Returns
    -------
    xgb.XGBRegressor
        Fitted regressor.
    """
    local_params = dict(params)
    early_stopping = int(local_params.pop("early_stopping_rounds", 25))
    model = xgb.XGBRegressor(**local_params, early_stopping_rounds=early_stopping)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model


def _train_lgb_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor with early stopping on a validation set."""
    local_params = dict(params)
    early_stopping = int(local_params.pop("early_stopping_rounds", 25))
    model = lgb.LGBMRegressor(**local_params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping, verbose=False)],
    )
    return model


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------


def _split_data(
    feature_frame: pd.DataFrame,
    y_regression: pd.Series,
    config: Dict[str, Any],
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """Create train / validation / test split without leakage.

    Parameters
    ----------
    feature_frame:
        Engineered feature columns (no targets).
    y_regression:
        Regression target series.
    config:
        Project configuration dictionary.

    Returns
    -------
    Tuple
        ``(x_train, x_val, x_test,
          y_reg_train, y_reg_val, y_reg_test)``
    """
    train_cfg = config["training"]
    test_size = float(train_cfg["test_size"])
    val_size = float(train_cfg["validation_size"])
    holdout_size = test_size + val_size

    x_train, x_holdout, y_reg_train, y_reg_holdout = train_test_split(
        feature_frame,
        y_regression,
        test_size=holdout_size,
        random_state=config["project"]["random_seed"],
    )

    val_prop = val_size / holdout_size
    x_val, x_test, y_reg_val, y_reg_test = train_test_split(
        x_holdout,
        y_reg_holdout,
        test_size=1.0 - val_prop,
        random_state=config["project"]["random_seed"],
    )

    return (
        x_train,
        x_val,
        x_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def train_chained_pipeline(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute the training pipeline with MLflow logging.

    Pipeline stages:
      1. Load and engineer features.
      2. Split into train / val / test.
      3. Fit preprocessor on training data.
      4. Optuna hyperparameter tuning for regressor (if enabled).
      5. Train final regressor.
      6. Log all artefacts, models, and metrics to MLflow.

    Parameters
    ----------
    config:
        Optional pre-loaded configuration dict.  Falls back to
        :func:`load_config` when *None*.

    Returns
    -------
    Dict[str, Any]
        Summary dict with ``run_id``, ``metrics``, and ``processed_data`` path.
    """
    cfg = config or load_config()

    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------
    tracking_uri: str = cfg["paths"]["mlflow_tracking_uri"]
    mlruns_dir = get_project_root() / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_file = tracking_uri.replace("sqlite:///", "")
        tracking_uri = f"sqlite:///{(get_project_root() / db_file).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # ------------------------------------------------------------------
    # Data loading & splitting
    # ------------------------------------------------------------------
    raw_df = _load_training_dataframe(cfg)
    feature_frame, y_regression = prepare_training_frames(raw_df, cfg)
    (
        x_train,
        x_val,
        x_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
    ) = _split_data(feature_frame, y_regression, cfg)

    # ------------------------------------------------------------------
    # Preprocessing (fit on training data only)
    # ------------------------------------------------------------------
    artifacts: FeatureArtifacts = build_and_fit_preprocessor(x_train, cfg)
    x_train_matrix = transform_features(x_train, artifacts, cfg)
    x_val_matrix = transform_features(x_val, artifacts, cfg)
    x_test_matrix = transform_features(x_test, artifacts, cfg)

    # ------------------------------------------------------------------
    # Regression — hyperparameter tuning
    # ------------------------------------------------------------------
    reg_params: Dict[str, Any] = dict(cfg["training"]["regression"])
    lgb_params: Dict[str, Any] = dict(cfg["training"]["lightgbm"])
    tuning_enabled: bool = cfg.get("hyperparameter_tuning", {}).get("enabled", False)

    if tuning_enabled:
        try:
            import optuna  # noqa: F811
        except ModuleNotFoundError:
            logger.warning("Optuna is not installed. Disabling hyperparameter tuning.")
            tuning_enabled = False

    if tuning_enabled:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        from sklearn.model_selection import KFold

        # Tuning XGBoost
        def _reg_objective(trial: optuna.Trial) -> float:
            """Optuna objective for regression: minimise validation RMSE using 5-fold CV with early stopping."""
            params: Dict[str, Any] = {
                "max_depth": trial.suggest_int("max_depth", 3, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 15),
                "gamma": trial.suggest_float("gamma", 0.1, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
                "objective": "reg:squarederror",
                "seed": cfg["project"]["random_seed"],
                "verbosity": 0,
                "n_jobs": 1,
            }
            
            kf = KFold(n_splits=5, shuffle=True, random_state=cfg["project"]["random_seed"])
            rmse_scores = []
            
            y_train_arr = y_reg_train.to_numpy()
            for train_idx, val_idx in kf.split(x_train_matrix):
                x_tr, x_va = x_train_matrix[train_idx], x_train_matrix[val_idx]
                y_tr, y_va = y_train_arr[train_idx], y_train_arr[val_idx]
                
                model = xgb.XGBRegressor(**params, early_stopping_rounds=25)
                model.fit(
                    x_tr,
                    np.log1p(y_tr),
                    eval_set=[(x_va, np.log1p(y_va))],
                    verbose=False,
                )
                
                val_preds = np.expm1(model.predict(x_va))
                val_rmse = float(np.sqrt(mean_squared_error(y_va, val_preds)))
                rmse_scores.append(val_rmse)
                
            return float(np.mean(rmse_scores))

        logger.info("Starting Optuna study for XGBoost...")
        study_reg = optuna.create_study(direction="minimize")
        study_reg.optimize(
            _reg_objective,
            n_trials=cfg["hyperparameter_tuning"]["n_trials"],
            n_jobs=int(cfg["hyperparameter_tuning"].get("max_threads", 1)),
        )
        reg_params.update(study_reg.best_trial.params)
        logger.info("XGBoost tuning complete. Best params: %s", study_reg.best_trial.params)

        # Tuning LightGBM
        def _lgb_objective(trial: optuna.Trial) -> float:
            """Optuna objective for LightGBM: minimise validation RMSE using 5-fold CV with early stopping."""
            params: Dict[str, Any] = {
                "max_depth": trial.suggest_int("max_depth", 3, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
                "objective": "regression",
                "random_state": cfg["project"]["random_seed"],
                "verbosity": -1,
                "n_jobs": 1,
            }
            
            kf = KFold(n_splits=5, shuffle=True, random_state=cfg["project"]["random_seed"])
            rmse_scores = []
            
            y_train_arr = y_reg_train.to_numpy()
            for train_idx, val_idx in kf.split(x_train_matrix):
                x_tr, x_va = x_train_matrix[train_idx], x_train_matrix[val_idx]
                y_tr, y_va = y_train_arr[train_idx], y_train_arr[val_idx]
                
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    x_tr,
                    np.log1p(y_tr),
                    eval_set=[(x_va, np.log1p(y_va))],
                    callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
                )
                
                val_preds = np.expm1(model.predict(x_va))
                val_rmse = float(np.sqrt(mean_squared_error(y_va, val_preds)))
                rmse_scores.append(val_rmse)
                
            return float(np.mean(rmse_scores))

        logger.info("Starting Optuna study for LightGBM...")
        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(
            _lgb_objective,
            n_trials=cfg["hyperparameter_tuning"]["n_trials"],
            n_jobs=int(cfg["hyperparameter_tuning"].get("max_threads", 1)),
        )
        lgb_params.update(study_lgb.best_trial.params)
        logger.info("LightGBM tuning complete. Best params: %s", study_lgb.best_trial.params)

    # ------------------------------------------------------------------
    # Regression — final training
    # ------------------------------------------------------------------
    y_reg_train_log = np.log1p(y_reg_train.to_numpy())
    y_reg_val_log = np.log1p(y_reg_val.to_numpy())

    logger.info("Training final XGBoost model...")
    xgb_regressor = _train_regressor(
        x_train_matrix,
        y_reg_train_log,
        x_val_matrix,
        y_reg_val_log,
        reg_params,
    )

    logger.info("Training final LightGBM model...")
    lgb_regressor = _train_lgb_regressor(
        x_train_matrix,
        y_reg_train_log,
        x_val_matrix,
        y_reg_val_log,
        lgb_params,
    )

    # Combine into EnsembleRegressor
    xgb_weight = float(cfg.get("training", {}).get("ensemble", {}).get("xgb_weight", 0.5))
    regressor = EnsembleRegressor(
        xgb_model=xgb_regressor,
        lgb_model=lgb_regressor,
        xgb_weight=xgb_weight,
    )

    # Predictions
    train_reg_preds = np.expm1(regressor.predict(x_train_matrix))
    val_reg_preds = np.expm1(regressor.predict(x_val_matrix))
    test_reg_preds = np.expm1(regressor.predict(x_test_matrix))

    # Regression metrics
    reg_train_metrics = _compute_regression_metrics(y_reg_train.to_numpy(), train_reg_preds)
    reg_val_metrics = _compute_regression_metrics(y_reg_val.to_numpy(), val_reg_preds)
    reg_test_metrics = _compute_regression_metrics(y_reg_test.to_numpy(), test_reg_preds)

    # ------------------------------------------------------------------
    # Persist processed features
    # ------------------------------------------------------------------
    processed_path = get_project_root() / cfg["paths"]["processed_data"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_parquet(processed_path, index=False)

    # ------------------------------------------------------------------
    # Metrics payload
    # ------------------------------------------------------------------
    metrics_payload: Dict[str, float] = {
        # Regression
        "regression_train_r2": reg_train_metrics["r2"],
        "regression_train_rmse": reg_train_metrics["rmse"],
        "regression_train_mae": reg_train_metrics["mae"],
        "regression_train_mse": reg_train_metrics["mse"],
        "regression_val_r2": reg_val_metrics["r2"],
        "regression_val_rmse": reg_val_metrics["rmse"],
        "regression_val_mae": reg_val_metrics["mae"],
        "regression_val_mse": reg_val_metrics["mse"],
        "regression_test_r2": reg_test_metrics["r2"],
        "regression_test_rmse": reg_test_metrics["rmse"],
        "regression_test_mae": reg_test_metrics["mae"],
        "regression_test_mse": reg_test_metrics["mse"],
    }

    metrics_dir = get_project_root() / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    run_name = (
        f"{cfg['mlflow']['run_name_prefix']}"
        f"_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "regressor_objective": cfg["training"]["regression"]["objective"],
                "xgb_weight": xgb_weight,
                "train_rows": len(x_train),
                "val_rows": len(x_val),
            }
        )
        mlflow.log_params({f"xgb_{k}": v for k, v in reg_params.items() if not isinstance(v, (dict, list))})
        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items() if not isinstance(v, (dict, list))})

        for metric_name, metric_value in metrics_payload.items():
            mlflow.log_metric(metric_name, metric_value)

        _trusted = [
            "numpy.dtype",
            "numpy.random.mtrand.RandomState",
            "builtins.object",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor",
            "lightgbm.sklearn.LGBMRegressor",
            "lightgbm.basic.Booster",
            "src.ensemble.EnsembleRegressor",
            "sklearn.compose._column_transformer.ColumnTransformer",
            "sklearn.pipeline.Pipeline",
            "sklearn.preprocessing._encoders.OneHotEncoder",
            "sklearn.preprocessing._data.StandardScaler",
            "sklearn.impute._base.SimpleImputer",
            "numpy.ndarray",
            "collections.OrderedDict",
        ]
        mlflow.sklearn.log_model(
            artifacts.preprocessor, "preprocessor", skops_trusted_types=_trusted
        )
        mlflow.sklearn.log_model(
            regressor,
            cfg["mlflow"]["model_names"]["regressor"],
            skops_trusted_types=_trusted,
        )
        mlflow.log_dict(
            {
                "regression_feature_names": artifacts.feature_names,
            },
            "feature_metadata.json",
        )

        run_id = run.info.run_id
        logger.info("MLflow run completed: %s", run_id)

    # ------------------------------------------------------------------
    # Export artifacts locally for standalone (e.g. Streamlit Cloud) use
    # ------------------------------------------------------------------
    import joblib
    models_dir = get_project_root() / cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, models_dir / "preprocessor.joblib")
    joblib.dump(regressor, models_dir / "regressor.joblib")
    logger.info("Saved local model copies to %s", models_dir)

    # ------------------------------------------------------------------
    # Summary & validation thresholds
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": metrics_payload,
        "processed_data": str(processed_path),
    }
    logger.info("Training complete. Metrics: %s", metrics_payload)

    # Final validation against user-specified targets
    if reg_test_metrics["rmse"] > 2100:
        logger.warning(
            "Regression RMSE (%.2f) exceeds target 2100", reg_test_metrics["rmse"]
        )
    if reg_test_metrics["mae"] > 1500:
        logger.warning(
            "Regression MAE (%.2f) exceeds target 1500", reg_test_metrics["mae"]
        )
    return summary


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for the training pipeline."""
    try:
        result = train_chained_pipeline()
        print(json.dumps(result, indent=2))
    except Exception as exc:
        logger.exception("Training pipeline failed.")
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
