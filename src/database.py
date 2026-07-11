"""SQLAlchemy database layer for MSSQL data access and inference logging."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.config_loader import load_config

logger = logging.getLogger(__name__)

metadata = MetaData()

customers_raw_table = Table(
    "customers_raw",
    metadata,
    Column("customer_id", String(32), primary_key=True),
    Column("age", BigInteger, nullable=False),
    Column("gender", String(10), nullable=False),
    Column("bmi", Float, nullable=False),
    Column("children", BigInteger, nullable=False),
    Column("smoker", String(5), nullable=False),
    Column("region", String(50), nullable=False),
    Column("occupation", String(100), nullable=False),
    Column("annual_income_usd", BigInteger, nullable=False),
    Column("exercise_level", String(20), nullable=False),
    Column("chronic_diseases", BigInteger, nullable=False),
    Column("doctor_visits_per_year", BigInteger, nullable=False),
    Column("hospitalizations_last_year", BigInteger, nullable=False),
    Column("alcohol_consumption_per_week", BigInteger, nullable=False),
    Column("insurance_plan", String(20), nullable=False),
    Column("annual_medical_cost_usd", Float, nullable=False),
    Column("blood_pressure", String(20), nullable=False),
    Column("diabetes", String(5), nullable=False),
    Column("cholesterol", BigInteger, nullable=False),
    Column("sleep_hours", Float, nullable=False),
    Column("stress_level", Float, nullable=False),
    Column("marital_status", String(20), nullable=False),
    Column("created_at", DateTime, nullable=False),
)

inference_logs_table = Table(
    "inference_logs",
    metadata,
    Column("log_id", BigInteger, primary_key=True, autoincrement=True),
    Column("request_id", String(36), nullable=False),
    Column("customer_id", String(32), nullable=True),
    Column("request_payload", Text, nullable=False),
    Column("predicted_annual_premium", Float, nullable=False),
    Column("predicted_insurance_type", String(20), nullable=False),
    Column("classification_probabilities", Text, nullable=True),
    Column("model_version", String(100), nullable=True),
    Column("latency_ms", Float, nullable=False),
    Column("created_at", DateTime, nullable=False),
)

drift_reports_table = Table(
    "drift_reports",
    metadata,
    Column("report_id", BigInteger, primary_key=True, autoincrement=True),
    Column("report_path", String(500), nullable=False),
    Column("drift_detected", BigInteger, nullable=False),
    Column("summary_json", Text, nullable=True),
    Column("created_at", DateTime, nullable=False),
)


class DatabaseManager:
    """Manages MSSQL connections, training data retrieval, and inference logs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker[Session]] = None

    def _build_connection_url(self) -> str:
        db_cfg = self.config["database"]
        driver = quote_plus(db_cfg["driver"])
        host = db_cfg["host"]
        database = db_cfg["database"]
        
        # Check if trusted connection (Windows Authentication) is enabled
        trusted = str(db_cfg.get("trusted_connection", "false")).lower() in ("true", "yes", "1")
        
        # Determine host and port; omit port for named instances (containing '\') or if port is empty
        port = db_cfg.get("port")
        if port and str(port).lower() != "none" and "\\" not in host:
            host_with_port = f"{host}:{port}"
        else:
            host_with_port = host

        if trusted:
            return (
                f"mssql+pyodbc://@{host_with_port}/"
                f"{database}?driver={driver}&trusted_connection=yes&TrustServerCertificate=yes"
            )
        else:
            username = quote_plus(db_cfg["username"])
            password = quote_plus(db_cfg["password"])
            return (
                f"mssql+pyodbc://{username}:{password}@{host_with_port}/"
                f"{database}?driver={driver}&TrustServerCertificate=yes"
            )

    @property
    def engine(self) -> Engine:
        """Lazy-initialize SQLAlchemy engine with connection pooling."""
        if self._engine is None:
            db_cfg = self.config["database"]
            try:
                self._engine = create_engine(
                    self._build_connection_url(),
                    pool_size=int(db_cfg.get("pool_size", 5)),
                    max_overflow=int(db_cfg.get("max_overflow", 10)),
                    pool_pre_ping=bool(db_cfg.get("pool_pre_ping", True)),
                    echo=bool(db_cfg.get("echo", False)),
                    fast_executemany=True,
                )
                logger.info("SQLAlchemy engine initialized for MSSQL.")
            except SQLAlchemyError as exc:
                logger.exception("Failed to create database engine.")
                raise RuntimeError(f"Database engine initialization failed: {exc}") from exc
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Return configured session factory bound to engine."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
            )
        return self._session_factory

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a transactional database session scope."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def health_check(self) -> bool:
        """Verify database connectivity."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as exc:
            logger.warning("Database health check failed: %s", exc)
            return False

    def fetch_training_data(self) -> pd.DataFrame:
        """
        Fetch raw customer records from MSSQL for model training.

        Returns
        -------
        pd.DataFrame
            Raw training dataframe with canonical column aliases applied.
        """
        col_cfg = self.config["columns"]
        query = text(
            """
            SELECT
                customer_id,
                age,
                gender,
                bmi,
                children,
                smoker,
                region,
                occupation,
                annual_income_usd,
                exercise_level,
                chronic_diseases,
                doctor_visits_per_year,
                hospitalizations_last_year,
                alcohol_consumption_per_week,
                insurance_plan,
                annual_medical_cost_usd,
                blood_pressure,
                diabetes,
                cholesterol,
                sleep_hours,
                stress_level,
                marital_status
            FROM dbo.customers_raw
            """
        )
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(query, connection)
        except SQLAlchemyError as exc:
            logger.exception("Failed to fetch training data from MSSQL.")
            raise RuntimeError(f"Training data retrieval failed: {exc}") from exc

        if df.empty:
            raise ValueError("Training dataset is empty. Seed dbo.customers_raw first.")

        rename_map = {
            col_cfg["source_regression"]: col_cfg["target_regression"],
        }
        df = df.rename(columns=rename_map)
        logger.info("Fetched %d training records from MSSQL.", len(df))
        return df

    def log_inference(
        self,
        request_payload: Dict[str, Any],
        predicted_annual_premium: float,
        predicted_insurance_type: str,
        classification_probabilities: Optional[Dict[str, float]],
        latency_ms: float,
        model_version: Optional[str] = None,
        customer_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Persist real-time inference request and predictions to MSSQL.

        Returns
        -------
        str
            Generated request UUID.
        """
        resolved_request_id = request_id or str(uuid.uuid4())
        record = {
            "request_id": resolved_request_id,
            "customer_id": customer_id,
            "request_payload": json.dumps(request_payload, default=str),
            "predicted_annual_premium": float(predicted_annual_premium),
            "predicted_insurance_type": predicted_insurance_type,
            "classification_probabilities": (
                json.dumps(classification_probabilities)
                if classification_probabilities is not None
                else None
            ),
            "model_version": model_version,
            "latency_ms": float(latency_ms),
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }
        try:
            with self.get_session() as session:
                session.execute(insert(inference_logs_table), [record])
            logger.info("Inference logged with request_id=%s", resolved_request_id)
        except SQLAlchemyError as exc:
            logger.exception("Failed to log inference to database.")
            raise RuntimeError(f"Inference logging failed: {exc}") from exc
        return resolved_request_id

    def fetch_inference_logs(
        self,
        since: Optional[datetime] = None,
        limit: int = 10_000,
    ) -> pd.DataFrame:
        """Retrieve production inference logs for drift monitoring."""
        base_query = """
            SELECT TOP (:limit)
                log_id,
                request_id,
                customer_id,
                request_payload,
                predicted_annual_premium,
                predicted_insurance_type,
                classification_probabilities,
                model_version,
                latency_ms,
                created_at
            FROM dbo.inference_logs
        """
        params: Dict[str, Any] = {"limit": limit}
        if since is not None:
            base_query += " WHERE created_at >= :since"
            params["since"] = since
        base_query += " ORDER BY created_at DESC"

        try:
            with self.engine.connect() as connection:
                logs_df = pd.read_sql(text(base_query), connection, params=params)
        except SQLAlchemyError as exc:
            logger.exception("Failed to fetch inference logs.")
            raise RuntimeError(f"Inference log retrieval failed: {exc}") from exc

        if logs_df.empty:
            return logs_df

        parsed_rows: List[Dict[str, Any]] = []
        for _, row in logs_df.iterrows():
            payload = json.loads(row["request_payload"])
            payload["predicted_annual_premium"] = row["predicted_annual_premium"]
            payload["insurance_type"] = row["predicted_insurance_type"]
            payload["annual_medical_premium"] = row["predicted_annual_premium"]
            payload["created_at"] = row["created_at"]
            parsed_rows.append(payload)

        return pd.DataFrame(parsed_rows)

    def log_drift_report(
        self,
        report_path: str,
        drift_detected: bool,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist drift monitoring report metadata."""
        record = {
            "report_path": report_path,
            "drift_detected": int(drift_detected),
            "summary_json": json.dumps(summary) if summary else None,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }
        try:
            with self.get_session() as session:
                session.execute(insert(drift_reports_table), [record])
            logger.info("Drift report logged: %s", report_path)
        except SQLAlchemyError as exc:
            logger.exception("Failed to log drift report.")
            raise RuntimeError(f"Drift report logging failed: {exc}") from exc
