"""
predict.py — Standalone prediction logic for XGBoost and PyTorch LSTM
====================================================================
Loads the trained XGBoost model and PyTorch LSTM network and exposes
predict_single() and predict_batch() functions supporting both models.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

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
    Load models, scaler, and feature list from disk.
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

def predict_single(
    sensor_data : Dict[str, float],
    model_type  : str = "xgboost",
    xgb_model   = None,
    scaler      = None,
    features    : Optional[List[str]] = None,
    lstm_model  = None,
    models_dir  : str = MODELS_DIR,
) -> Dict:
    """
    Predict RUL for a single engine snapshot using either XGBoost or LSTM.

    Args:
        sensor_data : dict mapping feature name -> value (must include all 13 features).
        model_type  : 'xgboost' or 'lstm'
        xgb_model   : pre-loaded XGBoost model (loaded if None).
        scaler      : pre-loaded scaler (loaded if None).
        features    : ordered feature list (loaded if None).
        lstm_model  : pre-loaded PyTorch LSTM model (loaded if None).
        models_dir  : path to the models folder.

    Returns:
        dict with keys: rul, health_pct, status, status_color, action_directive, cycle_age, model_used
    """
    if xgb_model is None or scaler is None or features is None or (model_type == "lstm" and lstm_model is None):
        xgb_m, sc, feats, lstm_m = load_artifacts(models_dir)
        xgb_model  = xgb_model or xgb_m
        scaler     = scaler or sc
        features   = features or feats
        lstm_model = lstm_model or lstm_m

    df = pd.DataFrame([sensor_data])[features]

    norm_type = model_type.lower().strip()
    if norm_type in ("lstm", "torch", "deep_learning") and lstm_model is not None and TORCH_AVAILABLE:
        # Scale features using standard scaler
        active_scaler = lstm_scaler if lstm_scaler is not None else scaler
        scaled_vals = active_scaler.transform(df)
        # For a single snapshot, create a sequence of length SEQUENCE_LENGTH (repeating recent state)
        seq = np.repeat(scaled_vals, SEQUENCE_LENGTH, axis=0)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 30, 13)
        with torch.no_grad():
            raw = float(lstm_model(seq_tensor).item())
        model_name = "LSTM Network (PyTorch)"
    else:
        raw = float(xgb_model.predict(df)[0])
        model_name = "XGBoost Regressor"

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
    models_dir  : str = MODELS_DIR,
) -> List[Dict]:
    """
    Predict RUL for a list of engine sensor rows using XGBoost or LSTM.
    Returns a list of result dicts (same schema as predict_single).
    """
    if xgb_model is None or scaler is None or features is None or (model_type == "lstm" and lstm_model is None):
        xgb_m, sc, feats, lstm_m = load_artifacts(models_dir)
        xgb_model  = xgb_model or xgb_m
        scaler     = scaler or sc
        features   = features or feats
        lstm_model = lstm_model or lstm_m

    df = pd.DataFrame(rows)[features]
    norm_type = model_type.lower().strip()

    if norm_type in ("lstm", "torch", "deep_learning") and lstm_model is not None and TORCH_AVAILABLE:
        scaled_vals = scaler.transform(df)
        # Generate sequence batch (each row as sequence of length SEQUENCE_LENGTH)
        # Shape: (N, 30, 13)
        seq_batch = np.repeat(scaled_vals[:, np.newaxis, :], SEQUENCE_LENGTH, axis=1)
        seq_tensor = torch.tensor(seq_batch, dtype=torch.float32)
        with torch.no_grad():
            raws = lstm_model(seq_tensor).numpy()
        model_name = "LSTM Network (PyTorch)"
    else:
        raws = xgb_model.predict(df)
        model_name = "XGBoost Regressor"

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


# ── CLI convenience ───────────────────────────────────────────────────────────

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
    print("\n=== Smoke Test: XGBoost ===")
    res_xgb = predict_single(HEALTHY, model_type="xgboost")
    for k, v in res_xgb.items():
        print(f"  {k:20s}: {v}")

    print("\n=== Smoke Test: LSTM ===")
    res_lstm = predict_single(HEALTHY, model_type="lstm")
    for k, v in res_lstm.items():
        print(f"  {k:20s}: {v}")
