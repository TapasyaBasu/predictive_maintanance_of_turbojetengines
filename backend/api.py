"""
MaintAI — FastAPI Backend
=========================
REST API for the NASA Turbofan Engine RUL (Remaining Useful Life) Predictor.
Supports both XGBoost Regressor and PyTorch LSTM Deep Learning Models.

Endpoints:
  GET  /              -> service info
  GET  /health        -> model load status
  POST /predict       -> single-engine prediction (query param ?model=xgboost|lstm)
  POST /predict/batch -> batch fleet prediction (query param ?model=xgboost|lstm)
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

from typing import List, Optional

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


# ── PyTorch LSTM Architecture ──────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class TurbofanLSTM(nn.Module):
        """
        2-Layer Recurrent Neural Network for RUL Prediction:
        Input: (batch_size, sequence_length, num_features)
        LSTM Layer 1: 64 hidden units, dropout 0.2
        LSTM Layer 2: 32 hidden units
        Dense Layers: 32 -> 16 -> 1 (Predicted RUL)
        """
        def __init__(self, input_dim=13, hidden_dim1=64, hidden_dim2=32, dropout=0.2):
            super().__init__()
            self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
            self.dropout1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
            self.fc1 = nn.Linear(hidden_dim2, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.dropout1(out)
            out, _ = self.lstm2(out)
            out = out[:, -1, :]  # Last time step
            out = self.relu(self.fc1(out))
            return self.fc2(out).squeeze(-1)
else:
    class TurbofanLSTM:  # type: ignore
        pass


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# ── App initialisation ────────────────────────────────────────────────────────
app = FastAPI(
    title       = "MaintAI — Turbofan Engine RUL API",
    description = (
        "Predicts the Remaining Useful Life (RUL) of NASA CMAPSS turbofan engines "
        "using tuned XGBoost and PyTorch LSTM models. Supports single and batch inference."
    ),
    version     = "2.0.0",
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
_xgb_model     = None
_lstm_model    = None
_scaler        = None
_lstm_scaler   = None
_features: list = []
_model_loaded  = False
_lstm_loaded   = False
_load_error    = ""
_load_time_ms  = 0.0

MAX_RUL         = 125.0   # engine life ceiling used during training
SEQUENCE_LENGTH = 30      # time-window for LSTM


@app.on_event("startup")
def load_model_on_startup() -> None:
    """Load model artifacts once when the server starts."""
    global _xgb_model, _lstm_model, _scaler, _features, _model_loaded, _lstm_loaded, _load_error, _load_time_ms
    t0 = time.perf_counter()
    try:
        # Load XGBoost model & Scaler
        _xgb_model = joblib.load(os.path.join(MODELS_DIR, "final_model.pkl"))
        _scaler    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        with open(os.path.join(MODELS_DIR, "feature_list.json")) as fh:
            _features = json.load(fh)
        _model_loaded = True

        # Load PyTorch LSTM model & dedicated scaler
        if TORCH_AVAILABLE:
            lstm_path = os.path.join(MODELS_DIR, "lstm_best_model.pth")
            if os.path.exists(lstm_path):
                lstm = TurbofanLSTM(input_dim=len(_features))
                lstm.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
                lstm.eval()
                _lstm_model = lstm
                _lstm_loaded = True
            
            lstm_sc_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
            if os.path.exists(lstm_sc_path):
                _lstm_scaler = joblib.load(lstm_sc_path)

        _load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        print(f"[MaintAI] [OK] Models loaded in {_load_time_ms} ms: XGBoost={'Loaded' if _model_loaded else 'Failed'}, LSTM={'Loaded' if _lstm_loaded else 'Unavailable'}")
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
    status        : str
    model_loaded  : bool
    lstm_loaded   : bool
    models_available: List[str]
    feature_count : int
    load_time_ms  : float
    error         : Optional[str]


class ModelInfoResponse(BaseModel):
    features          : List[str]
    feature_count     : int
    models_available  : List[str]
    primary_models    : dict
    max_rul_ceiling   : float
    thresholds        : dict
    status_definitions: dict


# ── Helper utilities ──────────────────────────────────────────────────────────

def rul_status(rul: float):
    """Return (status_text, color, action_directive) for a RUL value."""
    if rul > 80:
        return (
            "NOMINAL OPERATION",
            "green",
            "Standard routine monitoring; flight dispatch permitted with no restrictions.",
        )
    elif rul > 30:
        return (
            "MAINTENANCE ADVISORY",
            "amber",
            "Subsystem wear detected. Schedule non-destructive inspection (NDI) within 15–20 cycles.",
        )
    else:
        return (
            "CRITICAL EXHAUSTION (AOG)",
            "red",
            "Aircraft On Ground protocol. Immediate overhaul / engine change required before next mission.",
        )


def _predict_single_core(sensor: SensorInput, model_type: str = "xgboost") -> PredictionResponse:
    """Run inference on a single sensor snapshot using chosen model."""
    row = sensor.model_dump()
    df  = pd.DataFrame([row])[_features]
    norm_type = model_type.lower().strip()

    if norm_type in ("lstm", "torch", "deep_learning") and _lstm_loaded and _lstm_model is not None:
        scaled_vals = _scaler.transform(df)
        seq = np.repeat(scaled_vals, SEQUENCE_LENGTH, axis=0)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw_rul = float(_lstm_model(seq_tensor).item())
        model_name = "LSTM Network (PyTorch)"
    else:
        raw_rul    = float(_xgb_model.predict(df)[0])
        model_name = "XGBoost Regressor"

    rul_val  = float(np.clip(raw_rul, 0.0, MAX_RUL))
    health   = round((rul_val / MAX_RUL) * 100.0, 2)
    status_txt, color, action = rul_status(rul_val)

    return PredictionResponse(
        rul              = round(rul_val, 2),
        health_pct       = health,
        status           = status_txt,
        status_color     = color,
        action_directive = action,
        cycle_age        = sensor.Cycle,
        model_used       = model_name,
    )


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
        "models_supported": ["xgboost", "lstm"],
        "default_model"   : "xgboost",
        "version"         : "2.0.0",
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
    models_avail = []
    if _model_loaded:
        models_avail.append("xgboost")
    if _lstm_loaded:
        models_avail.append("lstm")

    return HealthResponse(
        status           = "ok" if _model_loaded else "degraded",
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
    """Returns metadata and benchmarks for both XGBoost and LSTM models."""
    models_avail = ["xgboost"]
    if _lstm_loaded:
        models_avail.append("lstm")

    return ModelInfoResponse(
        features          = _features,
        feature_count     = len(_features),
        models_available  = models_avail,
        primary_models    = {
            "xgboost": {
                "name"       : "XGBoost Regressor (Production Baseline)",
                "type"       : "Gradient Boosted Decision Trees",
                "test_rmse"  : 17.62,
                "test_mae"   : 11.69,
                "test_r2"    : 0.8133,
                "latency_ms" : "< 15 ms",
            },
            "lstm": {
                "name"       : "LSTM Neural Network (PyTorch Deep Learning)",
                "type"       : "2-Layer Sequential Recurrent Neural Network",
                "test_rmse"  : 13.81,
                "test_mae"   : 10.17,
                "test_r2"    : 0.8813,
                "window_size": 30,
                "latency_ms" : "~25 ms",
            }
        },
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
    model: str = Query("xgboost", description="Model architecture to use: 'xgboost' or 'lstm'")
):
    """
    Accepts 13 sensor readings for a single engine snapshot and returns
    the predicted Remaining Useful Life (RUL) using either XGBoost or LSTM.
    """
    if not _model_loaded:
        return _fallback_prediction(sensor, model_type=model)

    try:
        return _predict_single_core(sensor, model_type=model)
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Prediction failed: {exc}",
        )


@app.post(
    "/predict/batch",
    response_model      = BatchPredictionResponse,
    status_code         = status.HTTP_200_OK,
    summary             = "Batch fleet RUL prediction",
)
def predict_batch(
    request: BatchPredictionRequest,
    model: str = Query("xgboost", description="Model architecture to use: 'xgboost' or 'lstm'")
):
    """
    Accepts a list of engine sensor snapshots and returns a prediction for each,
    plus fleet-level aggregates using either XGBoost or LSTM.
    """
    MAX_BATCH = 5_000
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

    norm_type = model.lower().strip()
    try:
        if _model_loaded:
            rows = [r.model_dump() for r in request.readings]
            df   = pd.DataFrame(rows)[_features]

            if norm_type in ("lstm", "torch", "deep_learning") and _lstm_loaded and _lstm_model is not None:
                scaled_vals = _scaler.transform(df)
                seq_batch   = np.repeat(scaled_vals[:, np.newaxis, :], SEQUENCE_LENGTH, axis=1)
                seq_tensor  = torch.tensor(seq_batch, dtype=torch.float32)
                with torch.no_grad():
                    raw_ruls = _lstm_model(seq_tensor).numpy()
                model_name = "LSTM Network (PyTorch)"
            else:
                raw_ruls   = _xgb_model.predict(df)
                model_name = "XGBoost Regressor"

            ruls = np.clip(raw_ruls, 0.0, MAX_RUL).tolist()
        else:
            ruls = [max(0.0, 125.0 - s.Cycle * 0.65) for s in request.readings]
            model_name = f"Heuristic Baseline ({norm_type})"

        predictions = []
        for sensor, rul_val in zip(request.readings, ruls):
            rul_val  = float(rul_val)
            health   = round((rul_val / MAX_RUL) * 100.0, 2)
            status_txt, color, action = rul_status(rul_val)
            if not _model_loaded:
                action += " [HEURISTIC MODE — model unavailable]"
            predictions.append(
                PredictionResponse(
                    rul              = round(rul_val, 2),
                    health_pct       = health,
                    status           = status_txt,
                    status_color     = color,
                    action_directive = action,
                    cycle_age        = sensor.Cycle,
                    model_used       = model_name,
                )
            )

        rul_values = [p.rul for p in predictions]
        critical   = sum(1 for p in predictions if p.status_color == "red")

        return BatchPredictionResponse(
            count          = len(predictions),
            predictions    = predictions,
            fleet_avg_rul  = round(float(np.mean(rul_values)), 2),
            fleet_min_rul  = round(float(np.min(rul_values)), 2),
            critical_count = critical,
            model_used     = model_name,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Batch prediction failed: {exc}",
        )
