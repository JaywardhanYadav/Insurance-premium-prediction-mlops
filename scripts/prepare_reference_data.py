"""Create reference data snapshot for Evidently drift monitoring."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import get_project_root, load_config
from src.features import engineer_raw_features, prepare_training_frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    csv_path = get_project_root() / config["paths"]["raw_data"]
    if not csv_path.exists():
        print(f"ERROR: Raw data not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    col_cfg = config["columns"]
    raw_df = pd.read_csv(csv_path)
    raw_df = raw_df.rename(
        columns={
            col_cfg["source_regression"]: col_cfg["target_regression"],
        }
    )

    feature_frame, y_reg = prepare_training_frames(raw_df, config)
    engineered = engineer_raw_features(raw_df, config)

    reference = engineered.copy()
    reference[config["columns"]["target_regression"]] = y_reg

    output_path = get_project_root() / config["monitoring"]["reference_data_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(output_path, index=False)
    logger.info("Reference snapshot saved to %s (%d rows).", output_path, len(reference))
    print(f"Reference data written: {output_path}")


if __name__ == "__main__":
    main()
