"""
train.py — Backend training script (refactored from TRAIN_MODEL.py)
====================================================================
Trains the XGBoost RUL model on NASA CMAPSS data and saves artefacts.

Usage:
    python train.py                         # uses default paths
    python train.py --data-dir <path>       # custom dataset path
    python train.py --models-dir <path>     # custom model output path
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', 'Dataset')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

SELECTED_SENSORS = [
    'Cycle',
    'SensorMeasure2',  'SensorMeasure3',  'SensorMeasure4',
    'SensorMeasure7',  'SensorMeasure8',  'SensorMeasure11',
    'SensorMeasure12', 'SensorMeasure13', 'SensorMeasure15',
    'SensorMeasure17', 'SensorMeasure20', 'SensorMeasure21',
]


# ── Data loader ───────────────────────────────────────────────────────────────

def load_cmapss_subset(n: int, data_dir: str) -> pd.DataFrame:
    """Load one FD00{n} training file and attach a capped RUL column."""
    path = os.path.join(data_dir, f"train_FD00{n}.txt")
    cols = (
        ['ID', 'Cycle', 'OpSet1', 'OpSet2', 'OpSet3']
        + [f'SensorMeasure{i}' for i in range(1, 22)]
    )
    df = pd.read_csv(path, sep=' ', header=None)
    df.drop(df.columns[[26, 27]], axis=1, inplace=True)
    df.columns = cols

    max_cycles = df.groupby('ID')['Cycle'].max().reset_index()
    max_cycles.columns = ['ID', 'MaxCycle']
    df = df.merge(max_cycles, on='ID')
    df['RUL'] = (df['MaxCycle'] - df['Cycle']).clip(upper=125)
    df.drop('MaxCycle', axis=1, inplace=True)
    return df


def load_all(data_dir: str) -> pd.DataFrame:
    """Load all four CMAPSS subsets (FD001–FD004)."""
    frames = []
    for i in range(1, 5):
        try:
            frames.append(load_cmapss_subset(i, data_dir))
            print(f"   [OK] Loaded FD00{i}")
        except Exception as exc:
            print(f"   [WARN] Could not load FD00{i}: {exc}")
    if not frames:
        raise RuntimeError(f"No CMAPSS data found in: {data_dir}")
    return pd.concat(frames, ignore_index=True)


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir: str = DATA_DIR, models_dir: str = MODELS_DIR) -> dict:
    """
    Full training pipeline.

    Returns a metrics dict with train/test RMSE, MAE, R².
    """
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  MAINTAI — MODEL TRAINER (backend/train.py)")
    print("  Training XGBoost RUL Predictor on NASA CMAPSS")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading NASA CMAPSS dataset (FD001–FD004)...")
    df = load_all(data_dir)
    print(f"   Total rows: {len(df):,}")

    # 2. Feature/target split
    print("\n[2/5] Preparing features and train/test split...")
    X = df[SELECTED_SENSORS]
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)   # noqa: F841  (stored for reference)
    X_test_sc   = scaler.transform(X_test)        # noqa: F841

    print(f"   Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    # 3. Train XGBoost
    print("\n[3/5] Training XGBoost Regressor (1–3 minutes)...")
    model = xgb.XGBRegressor(
        n_estimators     = 400,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("   [OK] Training complete!")

    # 4. Evaluate
    print("\n[4/5] Evaluating model performance...")
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    result = {
        "train_rmse": float(np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))),
        "test_rmse" : float(np.sqrt(metrics.mean_squared_error(y_test,  y_pred_test))),
        "train_mae" : float(metrics.mean_absolute_error(y_train, y_pred_train)),
        "test_mae"  : float(metrics.mean_absolute_error(y_test,  y_pred_test)),
        "train_r2"  : float(metrics.r2_score(y_train, y_pred_train)),
        "test_r2"   : float(metrics.r2_score(y_test,  y_pred_test)),
    }

    print("\n   RESULTS:")
    print("   +------------------+----------+----------+")
    print("   | Metric           |  Train   |   Test   |")
    print("   +------------------+----------+----------+")
    print(f"   | RMSE             | {result['train_rmse']:8.2f} | {result['test_rmse']:8.2f} |")
    print(f"   | MAE              | {result['train_mae']:8.2f} | {result['test_mae']:8.2f}  |")
    print(f"   | R²               | {result['train_r2']:8.4f} | {result['test_r2']:8.4f} |")
    print("   +------------------+----------+----------+")

    # 5. Save artefacts
    print("\n[5/5] Saving model files...")
    joblib.dump(model,  os.path.join(models_dir, "final_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    with open(os.path.join(models_dir, "feature_list.json"), "w") as fh:
        json.dump(SELECTED_SENSORS, fh)

    comparison = pd.DataFrame([{
        "Model"    : "XGBoost (Final)",
        "Test RMSE": round(result["test_rmse"], 2),
        "Test MAE" : round(result["test_mae"],  2),
        "Test R2"  : round(result["test_r2"],   4),
    }])
    comparison.to_csv(os.path.join(models_dir, "model_comparison.csv"), index=False)

    print(f"   [SAVED] {models_dir}/final_model.pkl")
    print(f"   [SAVED] {models_dir}/scaler.pkl")
    print(f"   [SAVED] {models_dir}/feature_list.json")
    print(f"   [SAVED] {models_dir}/model_comparison.csv")

    print("\n" + "=" * 60)
    print("  MODEL TRAINING COMPLETE!")
    print("  Now run: python START_BACKEND.py")
    print("=" * 60 + "\n")

    return result


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaintAI XGBoost RUL model")
    parser.add_argument("--data-dir",   default=DATA_DIR,   help="Path to NASA CMAPSS dataset folder")
    parser.add_argument("--models-dir", default=MODELS_DIR, help="Path to save model artefacts")
    args = parser.parse_args()

    # Fix Windows console encoding
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    train(data_dir=args.data_dir, models_dir=args.models_dir)
