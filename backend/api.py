"""
MaintAI — FastAPI Backend
=========================
REST API for the NASA Turbofan Engine RUL (Remaining Useful Life) Predictor.
Supports Multi-Model Architectures:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. XGBoost Regressor (Tuned Production)
5. PyTorch LSTM Deep Recurrent Neural Network

Endpoints:
  GET  /              -> service info
  GET  /health        -> model load status & available architectures
  POST /predict       -> single-engine prediction (query param ?model=xgboost|lstm|linear_regression|random_forest|gradient_boosting)
  POST /predict/batch -> batch fleet prediction (query param ?model=xgboost|lstm|linear_regression|random_forest|gradient_boosting)
  GET  /features      -> required feature list
  GET  /model/info    -> model metadata and comparative benchmarks
  GET  /landing       -> serves standalone HTML landing page

Run with:
  uvicorn api:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Try importing PyTorch for LSTM inference
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Import prediction engine ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

from predict import (
    load_all_models,
    predict_single as core_predict_single,
    predict_batch as core_predict_batch,
    rul_status,
    normalize_model_type,
    MAX_RUL,
    SEQUENCE_LENGTH,
    TurbofanLSTM,
)


# ── App initialisation ────────────────────────────────────────────────────────
app = FastAPI(
    title       = "MaintAI — Turbofan Engine RUL API",
    description = (
        "Predicts Remaining Useful Life (RUL) of NASA C-MAPSS turbofan engines "
        "supporting Linear Regression, Random Forest, Gradient Boosting, XGBoost, and PyTorch LSTM."
    ),
    version     = "2.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Model state (loaded at startup) ──────────────────────────────────────────
_models_dict   : Dict[str, Any] = {}
_model_loaded  = False
_lstm_loaded   = False
_features      : list = []
_load_error    = ""
_load_time_ms  = 0.0


@app.on_event("startup")
def load_model_on_startup() -> None:
    """Load all model artifacts once when the server starts."""
    global _models_dict, _model_loaded, _lstm_loaded, _features, _load_error, _load_time_ms
    t0 = time.perf_counter()
    try:
        _models_dict = load_all_models(MODELS_DIR)
        _features    = _models_dict.get("features", [])
        _model_loaded = _models_dict.get("xgboost") is not None
        _lstm_loaded  = _models_dict.get("lstm") is not None

        loaded_names = [k for k, v in _models_dict.items() if v is not None and k not in ("scaler", "lstm_scaler", "features")]
        _load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        print(f"[MaintAI] [OK] Models loaded in {_load_time_ms} ms: {', '.join(loaded_names)}")
    except Exception as exc:
        _load_error   = str(exc)
        _model_loaded = False
        print(f"[MaintAI] [WARN] Model load failed: {exc}")


# ── Pydantic models ───────────────────────────────────────────────────────────

class SensorInput(BaseModel):
    """13-channel NASA C-MAPSS sensor reading for a single engine snapshot."""
    Cycle            : int   = Field(..., ge=1,   le=600,    description="Flight cycle age (accumulated mission cycles)")
    SensorMeasure2   : float = Field(..., description="LPC Outlet Temp (°R)")
    SensorMeasure3   : float = Field(..., description="HPC Outlet Temp (°R)")
    SensorMeasure4   : float = Field(..., description="LPT Outlet Temp (°R)")
    SensorMeasure7   : float = Field(..., description="HPC Static Pressure (psia)")
    SensorMeasure8   : float = Field(..., description="Fuel Flow Ratio (pps/psia)")
    SensorMeasure11  : float = Field(..., description="HPC Bleed Enthalpy (Btu/lbm)")
    SensorMeasure12  : float = Field(..., description="LP Turbine Coolant (lbm/s)")
    SensorMeasure13  : float = Field(..., description="HP Turbine Coolant (lbm/s)")
    SensorMeasure15  : float = Field(..., description="Bypass Ratio (ratio)")
    SensorMeasure17  : float = Field(..., description="Bleed Enthalpy (Btu/lbm)")
    SensorMeasure20  : float = Field(..., description="HPC Efficiency (ratio)")
    SensorMeasure21  : float = Field(..., description="LPT Efficiency (ratio)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Cycle": 1,
                    "SensorMeasure2" : 641.82,
                    "SensorMeasure3" : 1589.70,
                    "SensorMeasure4" : 1400.60,
                    "SensorMeasure7" : 554.36,
                    "SensorMeasure8" : 2388.02,
                    "SensorMeasure11": 47.47,
                    "SensorMeasure12": 521.66,
                    "SensorMeasure13": 2388.02,
                    "SensorMeasure15": 8.4195,
                    "SensorMeasure17": 392.0,
                    "SensorMeasure20": 39.06,
                    "SensorMeasure21": 23.419,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """RUL prediction result for a single engine snapshot."""
    rul              : float = Field(..., description="Predicted Remaining Useful Life in cycles")
    health_pct       : float = Field(..., description="Engine health as a percentage of max life (0–100)")
    status           : str   = Field(..., description="Operational status text")
    status_color     : str   = Field(..., description="Severity colour: green | amber | red")
    action_directive : str   = Field(..., description="Recommended maintenance action")
    cycle_age        : int   = Field(..., description="Input flight cycle age")
    model_used       : str   = Field("XGBoost Regressor", description="Model architecture used for inference")


class BatchPredictionRequest(BaseModel):
    """A list of sensor rows for bulk fleet prediction."""
    readings: List[SensorInput] = Field(..., description="List of engine sensor snapshots")


class BatchPredictionResponse(BaseModel):
    """Bulk prediction results."""
    count         : int                    = Field(..., description="Number of engines processed")
    predictions   : List[PredictionResponse]
    fleet_avg_rul : float                  = Field(..., description="Fleet-average predicted RUL")
    fleet_min_rul : float                  = Field(..., description="Lowest predicted RUL across fleet")
    critical_count: int                    = Field(..., description="Number of engines in CRITICAL status")
    model_used    : str                    = Field(..., description="Model architecture used for inference")


class HealthResponse(BaseModel):
    status          : str
    model_loaded    : bool
    lstm_loaded     : bool
    models_available: List[str]
    feature_count   : int
    load_time_ms    : float
    error           : Optional[str]


class ModelInfoResponse(BaseModel):
    features          : List[str]
    feature_count     : int
    models_available  : List[str]
    primary_models    : dict
    max_rul_ceiling   : float
    thresholds        : dict
    status_definitions: dict


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _fallback_prediction(sensor: SensorInput, model_type: str = "xgboost") -> PredictionResponse:
    """Simple heuristic prediction when the models are not loaded."""
    rul_val = float(max(0.0, 125.0 - sensor.Cycle * 0.65))
    health  = round((rul_val / MAX_RUL) * 100.0, 2)
    status_txt, color, action = rul_status(rul_val)
    return PredictionResponse(
        rul              = round(rul_val, 2),
        health_pct       = health,
        status           = status_txt,
        status_color     = color,
        action_directive = action + " [HEURISTIC MODE — model unavailable]",
        cycle_age        = sensor.Cycle,
        model_used       = f"Heuristic Baseline ({model_type})",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Service root / liveness check")
def root():
    """Returns basic service information."""
    return {
        "status"          : "online",
        "service"         : "MaintAI — Turbofan Engine RUL API",
        "models_supported": [
            "xgboost",
            "lstm",
            "linear_regression",
            "random_forest",
            "gradient_boosting",
        ],
        "default_model"   : "xgboost",
        "version"         : "2.1.0",
        "docs"            : "/docs",
        "landing"         : "/landing",
    }


@app.get("/landing", summary="Serve Standalone Landing Page")
def landing_page():
    """Serves the standalone HTML landing page directly over HTTP."""
    html_file = os.path.join(BASE_DIR, "..", "frontend", "landing", "index.html")
    if os.path.exists(html_file):
        return FileResponse(html_file, media_type="text/html")
    raise HTTPException(status_code=404, detail="Landing page HTML not found")


@app.get("/health", response_model=HealthResponse, summary="Model health status")
def health():
    """Returns the current model load status, feature count, and models available."""
    models_avail = [
        k for k in ["xgboost", "lstm", "linear_regression", "random_forest", "gradient_boosting"]
        if _models_dict.get(k) is not None
    ]

    return HealthResponse(
        status           = "ok" if len(models_avail) > 0 else "degraded",
        model_loaded     = _model_loaded,
        lstm_loaded      = _lstm_loaded,
        models_available = models_avail,
        feature_count    = len(_features),
        load_time_ms     = _load_time_ms,
        error            = _load_error if not _model_loaded else None,
    )


@app.get("/features", summary="Required sensor feature list")
def get_features():
    """Returns the ordered list of sensor features the models expect."""
    return {
        "features"      : _features,
        "feature_count" : len(_features),
        "description"   : "Submit values for all features in /predict -> SensorInput body",
    }


@app.get("/model/info", response_model=ModelInfoResponse, summary="Model metadata")
def model_info():
    """Returns metadata and benchmarks for all supported models."""
    models_avail = [
        k for k in ["xgboost", "lstm", "linear_regression", "random_forest", "gradient_boosting"]
        if _models_dict.get(k) is not None
    ]

    # Load from models_metadata.json if available
    meta_json_path = os.path.join(MODELS_DIR, "models_metadata.json")
    primary_models = {}
    if os.path.exists(meta_json_path):
        try:
            with open(meta_json_path) as f:
                meta = json.load(f)
            primary_models = meta.get("models", {})
            for m_k, m_v in primary_models.items():
                if isinstance(m_v, dict) and "metrics" in m_v:
                    for met_k, met_v in m_v["metrics"].items():
                        m_v[met_k] = met_v
        except Exception:
            pass

    if not primary_models:
        primary_models = {
            "linear_regression": {
                "name"     : "Linear Regression (Baseline)",
                "category" : "Classical Parametric Baseline",
                "test_rmse": 22.99,
                "test_mae" : 18.82,
                "test_r2"  : 0.6709,
            },
            "random_forest": {
                "name"     : "Random Forest Regressor",
                "category" : "Ensemble Bagging",
                "test_rmse": 19.34,
                "test_mae" : 14.35,
                "test_r2"  : 0.7671,
            },
            "gradient_boosting": {
                "name"     : "Gradient Boosting Regressor",
                "category" : "Ensemble Boosting",
                "test_rmse": 18.85,
                "test_mae" : 14.34,
                "test_r2"  : 0.7788,
            },
            "xgboost": {
                "name"     : "XGBoost Regressor (Tuned Production)",
                "category" : "Gradient Boosted Trees",
                "test_rmse": 18.91,
                "test_mae" : 14.35,
                "test_r2"  : 0.7773,
            },
            "lstm": {
                "name"     : "LSTM Neural Network (PyTorch Deep Learning)",
                "category" : "2-Layer Sequential Recurrent Neural Network",
                "test_rmse": 13.23,
                "test_mae" : 9.45,
                "test_r2"  : 0.8910,
            }
        }

    return ModelInfoResponse(
        features          = _features,
        feature_count     = len(_features),
        models_available  = models_avail,
        primary_models    = primary_models,
        max_rul_ceiling   = MAX_RUL,
        thresholds        = {
            "NOMINAL"  : {"min": 80,  "max": MAX_RUL, "description": "Engine operating within normal parameters"},
            "ADVISORY" : {"min": 30,  "max": 80,      "description": "Preventive maintenance recommended"},
            "CRITICAL" : {"min": 0,   "max": 30,      "description": "AOG — immediate intervention required"},
        },
        status_definitions= {
            "NOMINAL OPERATION"        : "Flight dispatch permitted; routine monitoring.",
            "MAINTENANCE ADVISORY"     : "NDI inspection required within 15–20 cycles.",
            "CRITICAL EXHAUSTION (AOG)": "Aircraft On Ground — overhaul before next mission.",
        },
    )


@app.post(
    "/predict",
    response_model      = PredictionResponse,
    status_code         = status.HTTP_200_OK,
    summary             = "Single-engine RUL prediction",
)
def predict(
    sensor: SensorInput,
    model: str = Query("xgboost", description="Model architecture: 'xgboost', 'lstm', 'linear_regression', 'random_forest', 'gradient_boosting'")
):
    """
    Accepts 13 sensor readings for a single engine snapshot and returns
    the predicted Remaining Useful Life (RUL) using the selected model.
    """
    if not _models_dict or all(v is None for k, v in _models_dict.items() if k not in ("scaler", "lstm_scaler", "features")):
        return _fallback_prediction(sensor, model_type=model)

    try:
        res = core_predict_single(sensor.model_dump(), model_type=model, extra_models=_models_dict)
        return PredictionResponse(**res)
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Prediction failed with model '{model}': {exc}",
        )


@app.post(
    "/predict/batch",
    response_model      = BatchPredictionResponse,
    status_code         = status.HTTP_200_OK,
    summary             = "Batch fleet RUL prediction",
)
def predict_batch(
    request: BatchPredictionRequest,
    model: str = Query("xgboost", description="Model architecture: 'xgboost', 'lstm', 'linear_regression', 'random_forest', 'gradient_boosting'")
):
    """
    Accepts a list of engine sensor snapshots and returns a prediction for each,
    plus fleet-level aggregates using the selected model.
    """
    MAX_BATCH = 10_000
    if len(request.readings) > MAX_BATCH:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = f"Batch size {len(request.readings)} exceeds the limit of {MAX_BATCH}.",
        )

    if len(request.readings) == 0:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = "readings list must contain at least one entry.",
        )

    try:
        rows_payload = [r.model_dump() for r in request.readings]
        results = core_predict_batch(rows_payload, model_type=model, extra_models=_models_dict)

        pred_objs = [PredictionResponse(**r) for r in results]
        rul_values = [p.rul for p in pred_objs]
        critical   = sum(1 for p in pred_objs if p.status_color == "red")
        active_model_name = pred_objs[0].model_used if pred_objs else model

        return BatchPredictionResponse(
            count          = len(pred_objs),
            predictions    = pred_objs,
            fleet_avg_rul  = round(float(np.mean(rul_values)), 2),
            fleet_min_rul  = round(float(np.min(rul_values)), 2),
            critical_count = critical,
            model_used     = active_model_name,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Batch prediction failed with model '{model}': {exc}",
        )
