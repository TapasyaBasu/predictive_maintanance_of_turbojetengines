"""
RUN_INDIVIDUAL_MODEL.py
=======================
Command-line runner to execute, evaluate, and demonstrate any single machine learning
model individually, as requested for college project demonstrations and viva.

Usage:
  Interactive menu:
    python RUN_INDIVIDUAL_MODEL.py

  Direct model selection:
    python RUN_INDIVIDUAL_MODEL.py --model linear_regression
    python RUN_INDIVIDUAL_MODEL.py --model random_forest
    python RUN_INDIVIDUAL_MODEL.py --model gradient_boosting
    python RUN_INDIVIDUAL_MODEL.py --model xgboost
    python RUN_INDIVIDUAL_MODEL.py --model lstm

  Or via shortcuts:
    python RUN_INDIVIDUAL_MODEL.py --model lr
    python RUN_INDIVIDUAL_MODEL.py --model rf
    python RUN_INDIVIDUAL_MODEL.py --model gb
    python RUN_INDIVIDUAL_MODEL.py --model xgb
    python RUN_INDIVIDUAL_MODEL.py --model lstm
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
BACKEND_DIR = os.path.join(BASE_DIR, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from predict import load_all_models, predict_single, normalize_model_type, TurbofanLSTM

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


MODEL_CATALOG = {
    "linear_regression": {
        "title": "LINEAR REGRESSION (BASELINE PARAMETRIC)",
        "file": "linear_regression.pkl",
        "category": "Classical Parametric Baseline (Ordinary Least Squares)",
        "icon": "📈",
    },
    "random_forest": {
        "title": "RANDOM FOREST REGRESSOR (ENSEMBLE BAGGING)",
        "file": "random_forest.pkl",
        "category": "Ensemble Bagging (100 Decision Trees, Max Depth 15)",
        "icon": "🌲",
    },
    "gradient_boosting": {
        "title": "GRADIENT BOOSTING REGRESSOR (STAGE-WISE GBDT)",
        "file": "gradient_boosting.pkl",
        "category": "Sequential Ensemble Boosting (150 Trees, lr=0.08)",
        "icon": "🚀",
    },
    "xgboost": {
        "title": "XGBOOST REGRESSOR (TUNED EXTREME GRADIENT BOOSTING)",
        "file": "final_model.pkl",
        "category": "Regularized Tree Boosting (400 Trees, lr=0.05, Subsample 0.8)",
        "icon": "⚡",
    },
    "lstm": {
        "title": "PYTORCH LSTM NETWORK (DEEP SEQUENCE MODEL)",
        "file": "lstm_best_model.pth",
        "category": "2-Layer Recurrent Neural Network (30-Cycle Sliding Window)",
        "icon": "🧬",
    },
}


def load_test_data(features):
    test_path = os.path.join(DATASET_DIR, "test_FD001.txt")
    rul_path = os.path.join(DATASET_DIR, "RUL_FD001.txt")
    if not os.path.exists(test_path) or not os.path.exists(rul_path):
        return None, None

    cols = ["ID", "Cycle", "OpSet1", "OpSet2", "OpSet3"] + [f"SensorMeasure{i}" for i in range(1, 22)]
    df_raw = pd.read_csv(test_path, sep=r"\s+", header=None).iloc[:, :26]
    df_raw.columns = cols

    rul_truth = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["TrueRUL"])
    rul_truth["ID"] = np.arange(1, len(rul_truth) + 1)

    last_df = df_raw.groupby("ID").last().reset_index().merge(rul_truth, on="ID")
    last_df["TrueRUL"] = last_df["TrueRUL"].clip(upper=125.0)
    return df_raw, last_df


def run_single_model(model_key: str):
    norm_key = normalize_model_type(model_key)
    meta = MODEL_CATALOG.get(norm_key)
    if not meta:
        print(f"[ERROR] Unknown model identifier: {model_key}")
        return

    print("\n" + "=" * 75)
    print(f"  {meta['icon']} MAINTAI — INDIVIDUAL MODEL EXECUTION: {meta['title']}")
    print("=" * 75)
    print(f"  Architecture Category : {meta['category']}")
    print(f"  Trained Artifact      : models/{meta['file']}")

    # 1. Load artifacts
    print("\n[1/4] Loading model and preprocessing pipeline...")
    models_dict = load_all_models(MODELS_DIR)
    features = models_dict["features"]
    scaler = models_dict["scaler"]
    lstm_scaler = models_dict["lstm_scaler"]
    model_obj = models_dict.get(norm_key)

    if model_obj is None:
        print(f"  [ERROR] Model artifact 'models/{meta['file']}' could not be loaded!")
        return

    art_path = os.path.join(MODELS_DIR, meta['file'])
    sz_kb = os.path.getsize(art_path) / 1024
    sz_str = f"{sz_kb/1024:.2f} MB" if sz_kb > 1024 else f"{sz_kb:.1f} KB"
    print(f"  [OK] Model successfully loaded ({model_obj.__class__.__name__}, File Size: {sz_str})")
    print(f"  [OK] Preprocessing Scaler ready (13 sensors input pipeline)")

    # 2. Display Model Configuration & Hyperparameters
    print("\n[2/4] Model Architecture & Hyperparameters:")
    meta_json_path = os.path.join(MODELS_DIR, "models_metadata.json")
    if os.path.exists(meta_json_path):
        try:
            with open(meta_json_path) as f:
                full_meta = json.load(f)
            m_info = full_meta["models"].get(norm_key, {})
            hparams = m_info.get("hyperparameters", {})
            for k, v in hparams.items():
                print(f"  - {k:<20}: {v}")

            if norm_key == "linear_regression":
                print(f"  - {'intercept':<20}: {m_info.get('intercept', 125.0):.4f}")
                print("  - Top 3 Degradation Features (Negative β):")
                coefs = sorted(m_info.get("coefficients", {}).items(), key=lambda x: x[1])
                for c_k, c_v in coefs[:3]:
                    print(f"      {c_k:<18}: β = {c_v:+.4f} (Surges as engine wears)")
            elif norm_key in ("random_forest", "gradient_boosting", "xgboost"):
                print("  - Top 3 Feature Importances (Gini/MDI):")
                imps = sorted(m_info.get("feature_importances", {}).items(), key=lambda x: x[1], reverse=True)
                for i_k, i_v in imps[:3]:
                    print(f"      {i_k:<18}: {i_v*100:.2f}% contribution")
        except Exception as e:
            print(f"  (Could not read metadata JSON: {e})")

    # 3. Evaluate on Official C-MAPSS Test Set (100 engines)
    print("\n[3/4] Evaluating on Official NASA C-MAPSS FD001 Test Set (100 Engines)...")
    df_raw, last_df = load_test_data(features)
    if last_df is not None:
        y_true = last_df["TrueRUL"].values

        if norm_key == "lstm":
            scaled_df = df_raw.copy()
            scaled_df[features] = lstm_scaler.transform(df_raw[features])
            seqs = []
            for eng_id in df_raw["ID"].unique():
                eng_data = scaled_df[scaled_df["ID"] == eng_id]
                mat = eng_data[features].values
                if len(eng_data) >= 30:
                    seqs.append(mat[-30:])
                else:
                    pad = np.repeat(mat[0:1], 30 - len(eng_data), axis=0)
                    seqs.append(np.vstack([pad, mat]))
            X_tensor = torch.tensor(np.array(seqs, dtype=np.float32))
            with torch.no_grad():
                y_pred = np.clip(model_obj(X_tensor).numpy(), 0, 125.0)
        elif norm_key == "linear_regression":
            X_sc = scaler.transform(last_df[features])
            y_pred = np.clip(model_obj.predict(X_sc), 0, 125.0)
        else:
            y_pred = np.clip(model_obj.predict(last_df[features]), 0, 125.0)

        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        mae = metrics.mean_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)

        print(f"  +-----------------------+-----------------------+")
        print(f"  | Metric                | Verified Test Score   |")
        print(f"  +-----------------------+-----------------------+")
        print(f"  | Root Mean Sq. Error   | {rmse:>10.2f} cycles   |")
        print(f"  | Mean Absolute Error   | {mae:>10.2f} cycles   |")
        print(f"  | R² Determination      | {r2:>10.4f}          |")
        print(f"  +-----------------------+-----------------------+")

    # 4. Live Sample Predictions on Representative Engines
    print("\n[4/4] Generating Live Independent Sample Predictions:")
    sample_engines = [
        (1, "Nominal / Early Life"),
        (24, "Wear Advisory"),
        (100, "Critical End-of-Life (AOG)"),
    ]

    for eng_id, state_desc in sample_engines:
        row = last_df[last_df["ID"] == eng_id].iloc[0]
        actual_rul = float(row["TrueRUL"])
        input_dict = {f: float(row[f]) for f in features}

        res = predict_single(input_dict, model_type=norm_key, extra_models=models_dict)
        pred_val = res["rul"]
        err = pred_val - actual_rul

        print(f"  Engine #{eng_id:<3} ({state_desc:<28}):")
        print(f"     Actual RUL     : {actual_rul:>5.1f} cycles")
        print(f"     Predicted RUL  : {pred_val:>5.1f} cycles (Error: {err:>+5.1f} cyc)")
        print(f"     Engine Status  : {res['status_color'].upper()} ({res['status']})")
        print(f"     Action         : {res['action_directive']}")

    print("\n" + "=" * 75)
    print(f"  DEMONSTRATION OF {meta['title']} COMPLETE!")
    print("=" * 75 + "\n")


def interactive_menu():
    print("\n" + "=" * 60)
    print("  MAINTAI — INDIVIDUAL MODEL RUNNER MENU")
    print("=" * 60)
    print("  Choose which machine learning model to execute:")
    print("    1. Linear Regression (Baseline Parametric)")
    print("    2. Random Forest Regressor (Ensemble Bagging)")
    print("    3. Gradient Boosting Regressor (Stage-wise GBDT)")
    print("    4. XGBoost Regressor (Tuned Extreme Gradient Boosting)")
    print("    5. PyTorch LSTM Network (Deep Sequence Recurrent Net)")
    print("    6. Exit")
    print("=" * 60)

    choice_map = {
        "1": "linear_regression",
        "2": "random_forest",
        "3": "gradient_boosting",
        "4": "xgboost",
        "5": "lstm",
    }

    try:
        choice = input("Enter choice (1-5): ").strip()
        if choice in choice_map:
            run_single_model(choice_map[choice])
        elif choice == "6":
            print("Exiting.")
        else:
            print("Invalid selection. Please choose 1, 2, 3, 4, or 5.")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an individual ML model independently.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to run: 'lr', 'rf', 'gb', 'xgb', 'lstm' or full name.",
    )
    args = parser.parse_args()

    if args.model:
        run_single_model(args.model)
    else:
        interactive_menu()

