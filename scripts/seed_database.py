"""Seed MSSQL customers_raw table from local CSV dataset."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import get_project_root, load_config
from src.database import DatabaseManager, customers_raw_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def seed_from_csv(batch_size: int = 500) -> int:
    """
    Load CSV records into dbo.customers_raw.

    Returns
    -------
    int
        Number of rows inserted.
    """
    config = load_config()
    csv_path = get_project_root() / config["paths"]["raw_data"]
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    db = DatabaseManager(config)

    if not db.health_check():
        raise RuntimeError("Cannot connect to MSSQL. Ensure the database is running.")

    with db.engine.begin() as connection:
        connection.execute(text("DELETE FROM dbo.customers_raw"))

    records = df.to_dict(orient="records")
    inserted = 0

    with db.get_session() as session:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            session.execute(customers_raw_table.insert(), batch)
            inserted += len(batch)
            logger.info("Inserted %d / %d rows.", inserted, len(records))

    logger.info("Seeding complete. Total rows: %d", inserted)
    return inserted


def main() -> None:
    try:
        count = seed_from_csv()
        print(f"Successfully seeded {count} records.")
    except Exception as exc:
        logger.exception("Database seeding failed.")
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
