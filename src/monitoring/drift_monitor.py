"""Evidently AI drift monitoring for production inference logs."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config_loader import get_project_root, load_config
from src.database import DatabaseManager
from src.features import engineer_raw_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_reference_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load reference dataset snapshot for drift comparison."""
    ref_path = get_project_root() / config["monitoring"]["reference_data_path"]
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference data not found at {ref_path}. Run prepare_reference_data.py first."
        )
    return pd.read_parquet(ref_path)


def _load_production_data(
    db_manager: DatabaseManager,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Load recent production inference logs as current dataset."""
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=lookback_days)
    production_df = db_manager.fetch_inference_logs(since=since)
    if production_df.empty:
        raise ValueError(
            f"No production inference logs found in the last {lookback_days} days."
        )
    return production_df


def _align_datasets(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Engineer and align feature columns across reference and current data."""
    ref_engineered = engineer_raw_features(reference_df, config)
    cur_engineered = engineer_raw_features(current_df, config)

    monitor_cfg = config["monitoring"]["evidently"]
    num_features = list(monitor_cfg["numerical_features"])
    cat_features = list(monitor_cfg["categorical_features"])

    for feature in num_features + cat_features:
        if feature not in ref_engineered.columns:
            logger.warning("Feature '%s' missing from reference data.", feature)
        if feature not in cur_engineered.columns:
            logger.warning("Feature '%s' missing from production data.", feature)

    available_num = [f for f in num_features if f in ref_engineered.columns and f in cur_engineered.columns]
    available_cat = [f for f in cat_features if f in ref_engineered.columns and f in cur_engineered.columns]

    ref_subset = ref_engineered[available_num + available_cat].copy()
    cur_subset = cur_engineered[available_num + available_cat].copy()
    return ref_subset, cur_subset, available_num, available_cat


def _run_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    target_regression: Optional[str] = None,
) -> tuple[Any, bool, Dict[str, Any]]:
    """Generate Evidently drift report and extract summary."""
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report
    except ImportError as exc:
        raise ImportError(
            "Evidently AI is required for drift monitoring. Install via requirements.txt."
        ) from exc

    column_mapping = ColumnMapping(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target=target_regression if target_regression in reference_df.columns else None,
        prediction=None,
    )

    metrics = [DataDriftPreset()]
    if (
        target_regression
        and target_regression in reference_df.columns
        and target_regression in current_df.columns
    ):
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    report_dict = report.as_dict()
    drift_detected = False
    summary: Dict[str, Any] = {"metrics": []}

    for metric_block in report_dict.get("metrics", []):
        metric_result = metric_block.get("result", {})
        metric_name = metric_block.get("metric", "unknown")
        block_summary: Dict[str, Any] = {"metric": metric_name}

        if "dataset_drift" in metric_result:
            drift_detected = drift_detected or bool(metric_result["dataset_drift"])
            block_summary["dataset_drift"] = metric_result["dataset_drift"]
            block_summary["drift_share"] = metric_result.get("drift_share")

        if "drift_by_columns" in metric_result:
            drifted_cols = [
                col
                for col, info in metric_result["drift_by_columns"].items()
                if info.get("drift_detected")
            ]
            if drifted_cols:
                drift_detected = True
            block_summary["drifted_columns"] = drifted_cols

        summary["metrics"].append(block_summary)

    summary["drift_detected"] = drift_detected
    return report, drift_detected, summary


def run_weekly_drift_check(
    config: Optional[Dict[str, Any]] = None,
    lookback_days: int = 7,
) -> Dict[str, Any]:
    """
    Execute weekly drift monitoring workflow.

    Compares reference training snapshot against recent production inference logs.
    """
    cfg = config or load_config()
    db_manager = DatabaseManager(cfg)

    reference_df = _load_reference_data(cfg)
    current_df = _load_production_data(db_manager, lookback_days=lookback_days)

    ref_aligned, cur_aligned, num_features, cat_features = _align_datasets(
        reference_df, current_df, cfg
    )

    monitor_cfg = cfg["monitoring"]["evidently"]
    report, drift_detected, summary = _run_evidently_report(
        reference_df=ref_aligned,
        current_df=cur_aligned,
        numerical_features=num_features,
        categorical_features=cat_features,
        target_regression=monitor_cfg.get("target_regression"),
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_dir = get_project_root() / cfg["monitoring"]["drift_report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    html_path = report_dir / f"drift_report_{timestamp}.html"
    json_path = report_dir / f"drift_summary_{timestamp}.json"

    try:
        report.save_html(str(html_path))
    except Exception:
        with html_path.open("w", encoding="utf-8") as handle:
            handle.write("<html><body><pre>")
            handle.write(json.dumps(report.as_dict(), indent=2, default=str))
            handle.write("</pre></body></html>")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    try:
        db_manager.log_drift_report(
            report_path=str(html_path),
            drift_detected=drift_detected,
            summary=summary,
        )
    except Exception as exc:
        logger.warning("Could not persist drift report to database: %s", exc)

    result = {
        "drift_detected": drift_detected,
        "report_html": str(html_path),
        "summary_json": str(json_path),
        "production_rows": len(cur_aligned),
        "reference_rows": len(ref_aligned),
        "summary": summary,
    }
    logger.info("Drift check complete. Drift detected: %s", drift_detected)
    return result


def main() -> None:
    """CLI entrypoint for scheduled drift monitoring."""
    try:
        outcome = run_weekly_drift_check()
        print(json.dumps(outcome, indent=2, default=str))
        if outcome["drift_detected"]:
            sys.exit(2)
    except Exception as exc:
        logger.exception("Drift monitoring failed.")
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
