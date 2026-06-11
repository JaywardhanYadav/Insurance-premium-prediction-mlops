"""FastAPI production scoring server with chained ML inference."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config_loader import load_config
from src.database import DatabaseManager
from src.inference import ChainedInferenceEngine
from src.model_loader import load_chained_models
from src.schemas import (
    ErrorResponse,
    HealthMetricsRequest,
    HealthResponse,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = load_config()
DB_MANAGER = DatabaseManager(CONFIG)

_inference_engine: Optional[ChainedInferenceEngine] = None
_model_run_id: Optional[str] = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Load MLflow models on startup and release resources on shutdown."""
    global _inference_engine, _model_run_id
    try:
        bundle = load_chained_models(config=CONFIG)
        _inference_engine = ChainedInferenceEngine(bundle)
        _model_run_id = bundle.mlflow_run_id
        logger.info("Chained models loaded successfully (run_id=%s).", _model_run_id)
    except Exception as exc:
        logger.error("Model loading failed at startup: %s", exc)
        _inference_engine = None
        _model_run_id = None
    yield
    _inference_engine = None
    logger.info("API shutdown complete.")


from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="src/static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch unhandled exceptions and return structured error responses."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=str(exc.detail), error_type="HTTPException").model_dump(),
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            detail="An unexpected server error occurred.",
            error_type=type(exc).__name__,
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return service health including database and model status."""
    db_ok = DB_MANAGER.health_check()
    models_ok = _inference_engine is not None
    overall = "healthy" if db_ok and models_ok else "degraded"
    return HealthResponse(
        status=overall,
        database_connected=db_ok,
        models_loaded=models_ok,
        mlflow_run_id=_model_run_id,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Inference"],
)
async def predict(payload: HealthMetricsRequest) -> PredictionResponse:
    """
    Execute chained inference on raw health metrics.

    Pipeline: Feature Engineering -> XGBRegressor -> Append Premium -> XGBClassifier
    """
    if _inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded. Train and register models via MLflow first.",
        )

    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    try:
        raw_record = payload.model_dump()
        customer_id = raw_record.pop("customer_id", None)

        premium, insurance_type, probabilities = _inference_engine.predict_single(
            raw_record
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        try:
            logged_request_id = DB_MANAGER.log_inference(
                request_payload=payload.model_dump(),
                predicted_annual_premium=premium,
                predicted_insurance_type=insurance_type,
                classification_probabilities=probabilities,
                latency_ms=latency_ms,
                model_version=_model_run_id,
                customer_id=customer_id,
                request_id=request_id,
            )
            request_id = logged_request_id
        except Exception as db_exc:
            logger.warning(
                "Inference succeeded but database logging failed: %s", db_exc
            )

        return PredictionResponse(
            request_id=request_id,
            customer_id=customer_id,
            predicted_annual_premium=round(premium, 2),
            predicted_insurance_type=insurance_type,  # type: ignore[arg-type]
            classification_probabilities={
                k: round(v, 4) for k, v in probabilities.items()
            },
            model_version=_model_run_id,
            latency_ms=round(latency_ms, 2),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Prediction endpoint failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


def run_server() -> None:
    """Launch Uvicorn server (used by Docker CMD)."""
    import uvicorn

    uvicorn.run(
        "src.app:app",
        host=CONFIG["api"]["host"],
        port=int(CONFIG["api"]["port"]),
        reload=bool(CONFIG["api"].get("reload", False)),
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
