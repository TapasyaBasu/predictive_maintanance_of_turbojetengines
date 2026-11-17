"""
predict.py — Standalone prediction logic for Multi-Model Prognostics
=====================================================================
Loads trained models (Linear Regression, Random Forest, Gradient Boosting,
XGBoost, and PyTorch LSTM) and exposes predict_single() and predict_batch()
functions supporting all architectures independently.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
MAX_RUL    = 125.0
SEQUENCE_LENGTH = 30


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


# ── Model loading ─────────────────────────────────────────────────────────────

def load_artifacts(models_dir: str = MODELS_DIR):
    """
    Backwards-compatible artifact loader.
    Returns: (xgb_model, scaler, features, lstm_model, lstm_scaler)
    """
    xgb_model = joblib.load(os.path.join(models_dir, "final_model.pkl"))
    scaler    = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_list.json")) as fh:
        features = json.load(fh)

    lstm_scaler = None
    if os.path.exists(os.path.join(models_dir, "lstm_scaler.pkl")):
        try:
            lstm_scaler = joblib.load(os.path.join(models_dir, "lstm_scaler.pkl"))
        except Exception:
            lstm_scaler = scaler
    else:
        lstm_scaler = scaler

    lstm_model = None
    if TORCH_AVAILABLE:
        lstm_path = os.path.join(models_dir, "lstm_best_model.pth")
        if os.path.exists(lstm_path):
            try:
                lstm = TurbofanLSTM(input_dim=len(features))
                lstm.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
                lstm.eval()
                lstm_model = lstm
            except Exception as e:
                print(f"[predict.py] Could not load LSTM weights: {e}")

    return xgb_model, scaler, features, lstm_model, lstm_scaler


def load_all_models(models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    """
    Loads all available model artifacts, scalers, and metadata from disk.
    Returns a dictionary of loaded model instances and helpers.
    """
    xgb_m, sc, feats, lstm_m, lstm_sc = load_artifacts(models_dir)

    models_dict = {
        "xgboost"          : xgb_m,
        "lstm"             : lstm_m,
        "scaler"           : sc,
        "lstm_scaler"      : lstm_sc,
        "features"         : feats,
        "linear_regression": None,
        "random_forest"    : None,
        "gradient_boosting": None,
    }

    # Load Linear Regression
    lr_path = os.path.join(models_dir, "linear_regression.pkl")
    if os.path.exists(lr_path):
        try:
            models_dict["linear_regression"] = joblib.load(lr_path)
        except Exception as e:
            print(f"[predict.py] Could not load Linear Regression: {e}")

    # Load Random Forest
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    if os.path.exists(rf_path):
        try:
            models_dict["random_forest"] = joblib.load(rf_path)
        except Exception as e:
            print(f"[predict.py] Could not load Random Forest: {e}")

    # Load Gradient Boosting
    gb_path = os.path.join(models_dir, "gradient_boosting.pkl")
    if os.path.exists(gb_path):
        try:
            models_dict["gradient_boosting"] = joblib.load(gb_path)
        except Exception as e:
            print(f"[predict.py] Could not load Gradient Boosting: {e}")

    return models_dict


# ── Status helper ─────────────────────────────────────────────────────────────

def rul_status(rul: float) -> Tuple[str, str, str]:
    """Return (status_text, color, action_directive) for a predicted RUL."""
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


# ── Core predict functions ────────────────────────────────────────────────────

def normalize_model_type(model_type: str) -> str:
    """Normalize model string to standard identifier."""
    m = model_type.lower().strip()
    if any(k in m for k in ["xgb", "xgboost"]):
        return "xgboost"
    elif any(k in m for k in ["lstm", "torch", "deep", "recurrent"]):
        return "lstm"
    elif any(k in m for k in ["linear", "lr", "ols"]):
        return "linear_regression"
    elif any(k in m for k in ["forest", "rf", "random"]):
        return "random_forest"
    elif any(k in m for k in ["gradient", "gbdt"]) or m == "gb":
        return "gradient_boosting"
    else:
        return "xgboost"


def predict_single(
    sensor_data : Dict[str, float],
    model_type  : str = "xgboost",
    xgb_model   = None,
    scaler      = None,
    features    : Optional[List[str]] = None,
    lstm_model  = None,
    lstm_scaler = None,
    models_dir  : str = MODELS_DIR,
    extra_models: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Predict RUL for a single engine snapshot using any supported model architecture:
    'linear_regression', 'random_forest', 'gradient_boosting', 'xgboost', or 'lstm'.
    """
    target_model = normalize_model_type(model_type)

    # Load artifacts if not passed
    if extra_models is None:
        extra_models = load_all_models(models_dir)

    features = features or extra_models["features"]
    scaler = scaler or extra_models["scaler"]
    lstm_scaler = lstm_scaler or extra_models["lstm_scaler"]

    df = pd.DataFrame([sensor_data])[features]

    if target_model == "lstm":
        active_lstm = lstm_model or extra_models.get("lstm")
        if active_lstm is None or not TORCH_AVAILABLE:
            raise RuntimeError("LSTM model artifact (lstm_best_model.pth) is unavailable or PyTorch is not installed.")
        active_sc = lstm_scaler if lstm_scaler is not None else scaler
        scaled_vals = active_sc.transform(df)
        seq = np.repeat(scaled_vals, SEQUENCE_LENGTH, axis=0)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw = float(active_lstm(seq_tensor).item())
        model_name = "LSTM Network (PyTorch)"

    elif target_model == "linear_regression":
        active_lr = extra_models.get("linear_regression")
        if active_lr is None:
            raise FileNotFoundError("Linear Regression artifact (linear_regression.pkl) is not found in models/.")
        scaled_vals = scaler.transform(df)
        raw = float(active_lr.predict(scaled_vals)[0])
        model_name = "Linear Regression (Baseline)"

    elif target_model == "random_forest":
        active_rf = extra_models.get("random_forest")
        if active_rf is None:
            raise FileNotFoundError("Random Forest artifact (random_forest.pkl) is not found in models/.")
        raw = float(active_rf.predict(df)[0])
        model_name = "Random Forest Regressor"

    elif target_model == "gradient_boosting":
        active_gb = extra_models.get("gradient_boosting")
        if active_gb is None:
            raise FileNotFoundError("Gradient Boosting artifact (gradient_boosting.pkl) is not found in models/.")
        raw = float(active_gb.predict(df)[0])
        model_name = "Gradient Boosting Regressor"

    else:  # default: xgboost
        active_xgb = xgb_model or extra_models.get("xgboost")
        if active_xgb is None:
            raise FileNotFoundError("XGBoost artifact (final_model.pkl) is not found in models/.")
        raw = float(active_xgb.predict(df)[0])
        model_name = "XGBoost Regressor (Tuned)"

    rul_val = float(np.clip(raw, 0.0, MAX_RUL))
    health  = round((rul_val / MAX_RUL) * 100.0, 2)
    status_txt, color, action = rul_status(rul_val)

    return {
        "rul"              : round(rul_val, 2),
        "health_pct"       : health,
        "status"           : status_txt,
        "status_color"     : color,
        "action_directive" : action,
        "cycle_age"        : int(sensor_data.get("Cycle", 0)),
        "model_used"       : model_name,
    }


def predict_batch(
    rows        : List[Dict[str, float]],
    model_type  : str = "xgboost",
    xgb_model   = None,
    scaler      = None,
    features    : Optional[List[str]] = None,
    lstm_model  = None,
    lstm_scaler = None,
    models_dir  : str = MODELS_DIR,
    extra_models: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """
    Predict RUL for a list of engine sensor rows using any supported model architecture.
    """
    target_model = normalize_model_type(model_type)

    if extra_models is None:
        extra_models = load_all_models(models_dir)

    features = features or extra_models["features"]
    scaler = scaler or extra_models["scaler"]
    lstm_scaler = lstm_scaler or extra_models["lstm_scaler"]

    df = pd.DataFrame(rows)[features]

    if target_model == "lstm":
        active_lstm = lstm_model or extra_models.get("lstm")
        if active_lstm is None or not TORCH_AVAILABLE:
            raise RuntimeError("LSTM model artifact is unavailable.")
        active_sc = lstm_scaler if lstm_scaler is not None else scaler
        scaled_vals = active_sc.transform(df)
        seq_batch = np.repeat(scaled_vals[:, np.newaxis, :], SEQUENCE_LENGTH, axis=1)
        seq_tensor = torch.tensor(seq_batch, dtype=torch.float32)
        with torch.no_grad():
            raws = active_lstm(seq_tensor).numpy()
        model_name = "LSTM Network (PyTorch)"

    elif target_model == "linear_regression":
        active_lr = extra_models.get("linear_regression")
        if active_lr is None:
            raise FileNotFoundError("Linear Regression artifact is missing.")
        scaled_vals = scaler.transform(df)
        raws = active_lr.predict(scaled_vals)
        model_name = "Linear Regression (Baseline)"

    elif target_model == "random_forest":
        active_rf = extra_models.get("random_forest")
        if active_rf is None:
            raise FileNotFoundError("Random Forest artifact is missing.")
        raws = active_rf.predict(df)
        model_name = "Random Forest Regressor"

    elif target_model == "gradient_boosting":
        active_gb = extra_models.get("gradient_boosting")
        if active_gb is None:
            raise FileNotFoundError("Gradient Boosting artifact is missing.")
        raws = active_gb.predict(df)
        model_name = "Gradient Boosting Regressor"

    else:
        active_xgb = xgb_model or extra_models.get("xgboost")
        if active_xgb is None:
            raise FileNotFoundError("XGBoost artifact is missing.")
        raws = active_xgb.predict(df)
        model_name = "XGBoost Regressor (Tuned)"

    ruls = np.clip(raws, 0.0, MAX_RUL)

    results = []
    for i, rul_val in enumerate(ruls):
        rul_val = float(rul_val)
        health  = round((rul_val / MAX_RUL) * 100.0, 2)
        status_txt, color, action = rul_status(rul_val)
        results.append({
            "rul"              : round(rul_val, 2),
            "health_pct"       : health,
            "status"           : status_txt,
            "status_color"     : color,
            "action_directive" : action,
            "cycle_age"        : int(rows[i].get("Cycle", 0)),
            "model_used"       : model_name,
        })
    return results


# ── Smoke Test CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HEALTHY = {
        "Cycle"          : 1,
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

    models_to_test = ["linear_regression", "random_forest", "gradient_boosting", "xgboost", "lstm"]
    print("=" * 60)
    print("  SMOKE TESTING ALL INDEPENDENT INFERENCE PIPELINES")
    print("=" * 60)
    for m in models_to_test:
        try:
            res = predict_single(HEALTHY, model_type=m)
            print(f"[{m.upper():<18}] -> RUL: {res['rul']:>6.2f} cyc | Health: {res['health_pct']:>5.1f}% | Model: {res['model_used']}")
        except Exception as e:
            print(f"[{m.upper():<18}] -> FAILED: {e}")
    print("=" * 60)
