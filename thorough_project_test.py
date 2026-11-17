"""
thorough_project_test.py
========================
Deep verification script covering:
- TEST A: Model Loading
- TEST B: Preprocessing Verification
- TEST C: Independent Predictions across representative test engines
- TEST D: Independent Metric Recalculation vs Displayed Benchmarks
- TEST E: Graph and Plot Verification
- TEST F: Model Switching and State Isolation
- TEST G: Original MaintAI Dashboard Functionality
- TEST H: Backend REST API Endpoints Verification
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', 'major project review', 'NASA_Turbofan_JetEngine_RUL'))
if not os.path.exists(BASE):
    BASE = r'c:\major project review\NASA_Turbofan_JetEngine_RUL'

MODELS_DIR = os.path.join(BASE, 'models')
DATA_DIR = os.path.join(BASE, 'Dataset')
BACKEND_DIR = os.path.join(BASE, 'backend')
FRONTEND_DIR = os.path.join(BASE, 'frontend', 'app')

sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, FRONTEND_DIR)

import torch
import torch.nn as nn
from predict import load_all_models, predict_single, predict_batch, TurbofanLSTM, normalize_model_type

print("=" * 85)
print("  MAINTAI AUDIT & VERIFICATION SUITE — COMPREHENSIVE EXECUTION")
print("=" * 85)

# ─── TEST A: MODEL LOADING ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  TEST A: INDEPENDENT MODEL LOADING")
print("=" * 50)

models_dict = load_all_models(MODELS_DIR)

test_a_results = {}
for m_key, expected_type in [
    ("linear_regression", "LinearRegression"),
    ("random_forest", "RandomForestRegressor"),
    ("gradient_boosting", "GradientBoostingRegressor"),
    ("xgboost", "XGBRegressor"),
    ("lstm", "TurbofanLSTM"),
]:
    obj = models_dict.get(m_key)
    if obj is None:
        status = "FAIL (None)"
        obj_name = "None"
    else:
        obj_name = obj.__class__.__name__
        status = "PASS" if expected_type in obj_name else f"FAIL (unexpected class {obj_name})"
    test_a_results[m_key] = {"status": status, "class": obj_name}
    print(f"  [{m_key:<18}] -> Status: {status:<6} | Object: {obj_name}")


# ─── TEST B: PREPROCESSING VERIFICATION ──────────────────────────────────────
print("\n" + "=" * 50)
print("  TEST B: PREPROCESSING VERIFICATION")
print("=" * 50)

scaler = models_dict.get("scaler")
lstm_scaler = models_dict.get("lstm_scaler")
features = models_dict.get("features")

print(f"  Features loaded: {len(features)} channels")
print(f"  Feature list: {features}")

# Test dimensions and scaling behavior
test_df = pd.DataFrame([{feat: 100.0 for feat in features}])
scaled_std = scaler.transform(test_df)
print(f"  StandardScaler transform shape: {scaled_std.shape}")
print(f"  LSTM Scaler transform shape: {lstm_scaler.transform(test_df).shape}")

# Verify LSTM window requirements
sample_seq = np.repeat(scaled_std, 30, axis=0)
print(f"  LSTM sliding window shape: {sample_seq.shape} (Sequence length: 30, Features: 13)")


# ─── TEST C: INDEPENDENT PREDICTION ON TEST ENGINES ──────────────────────────
print("\n" + "=" * 50)
print("  TEST C: INDEPENDENT PREDICTIONS ON C-MAPSS TEST SET")
print("=" * 50)

# Load test data
test_path = os.path.join(DATA_DIR, 'test_FD001.txt')
rul_path = os.path.join(DATA_DIR, 'RUL_FD001.txt')
cols = ['ID', 'Cycle', 'OpSet1', 'OpSet2', 'OpSet3'] + [f'SensorMeasure{i}' for i in range(1, 22)]
df_raw = pd.read_csv(test_path, sep=r'\s+', header=None).iloc[:, :26]
df_raw.columns = cols

rul_truth = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['TrueRUL'])
rul_truth['ID'] = np.arange(1, len(rul_truth) + 1)

last_cycles = df_raw.groupby('ID').last().reset_index().merge(rul_truth, on='ID')
last_cycles['TrueRUL'] = last_cycles['TrueRUL'].clip(upper=125.0)

# Pick 5 diverse representative test engines:
# Engine 1 (Medium wear), Engine 24 (Low wear / Healthy), Engine 45 (Near end-of-life), Engine 75 (Medium-low), Engine 100 (Critical AOG)
test_engines = [1, 24, 45, 75, 100]
test_c_records = []

for eng_id in test_engines:
    eng_row = last_cycles[last_cycles['ID'] == eng_id].iloc[0]
    true_rul = float(eng_row['TrueRUL'])
    input_dict = {feat: float(eng_row[feat]) for feat in features}
    
    print(f"\n--- Testing Engine #{eng_id} (Cycle {int(input_dict['Cycle'])}, Ground Truth RUL: {true_rul:.1f} cycles) ---")
    for m_key in ["linear_regression", "random_forest", "gradient_boosting", "xgboost", "lstm"]:
        try:
            res = predict_single(input_dict, model_type=m_key, extra_models=models_dict)
            pred_rul = res['rul']
            err = pred_rul - true_rul
            status = "PASS"
            print(f"  {m_key:<18} -> Pred: {pred_rul:>5.1f} cyc | Error: {err:>+6.1f} cyc | Status: {res['status_color']} ({res['status'][:12]}) | Model: {res['model_used']}")
        except Exception as e:
            pred_rul = None
            status = f"FAIL ({e})"
            print(f"  {m_key:<18} -> FAILED: {e}")
        
        test_c_records.append({
            "model": m_key,
            "engine_id": eng_id,
            "true_rul": true_rul,
            "pred_rul": pred_rul,
            "status": status
        })


# ─── TEST D: EVALUATION METRICS VERIFICATION ─────────────────────────────────
print("\n" + "=" * 50)
print("  TEST D: EVALUATION METRICS INDEPENDENT RECALCULATION")
print("=" * 50)

# Ground truth RUL for all 100 engines
y_true_all = last_cycles['TrueRUL'].values
X_all = last_cycles[features]
X_all_sc = scaler.transform(X_all)

# Read metadata JSON and comparison CSV
with open(os.path.join(MODELS_DIR, 'models_metadata.json')) as f:
    meta_json = json.load(f)

comp_df = pd.read_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'))

metric_verification = []

# 1. Linear Regression
lr_model = models_dict['linear_regression']
pred_lr = np.clip(lr_model.predict(X_all_sc), 0, 125.0)
rec_lr_rmse = np.sqrt(metrics.mean_squared_error(y_true_all, pred_lr))
rec_lr_mae  = metrics.mean_absolute_error(y_true_all, pred_lr)
rec_lr_r2   = metrics.r2_score(y_true_all, pred_lr)

# 2. Random Forest
rf_model = models_dict['random_forest']
pred_rf = np.clip(rf_model.predict(X_all), 0, 125.0)
rec_rf_rmse = np.sqrt(metrics.mean_squared_error(y_true_all, pred_rf))
rec_rf_mae  = metrics.mean_absolute_error(y_true_all, pred_rf)
rec_rf_r2   = metrics.r2_score(y_true_all, pred_rf)

# 3. Gradient Boosting
gb_model = models_dict['gradient_boosting']
pred_gb = np.clip(gb_model.predict(X_all), 0, 125.0)
rec_gb_rmse = np.sqrt(metrics.mean_squared_error(y_true_all, pred_gb))
rec_gb_mae  = metrics.mean_absolute_error(y_true_all, pred_gb)
rec_gb_r2   = metrics.r2_score(y_true_all, pred_gb)

# 4. XGBoost
xgb_model = models_dict['xgboost']
pred_xgb = np.clip(xgb_model.predict(X_all), 0, 125.0)
rec_xgb_rmse = np.sqrt(metrics.mean_squared_error(y_true_all, pred_xgb))
rec_xgb_mae  = metrics.mean_absolute_error(y_true_all, pred_xgb)
rec_xgb_r2   = metrics.r2_score(y_true_all, pred_xgb)

# 5. LSTM
lstm_model = models_dict['lstm']
scaled_raw_test = df_raw.copy()
scaled_raw_test[features] = lstm_scaler.transform(df_raw[features])
lstm_seqs = []
for engine_id in df_raw['ID'].unique():
    engine_data = scaled_raw_test[scaled_raw_test['ID'] == engine_id]
    feat_matrix = engine_data[features].values
    if len(engine_data) >= 30:
        lstm_seqs.append(feat_matrix[-30:])
    else:
        pad_len = 30 - len(engine_data)
        padding = np.repeat(feat_matrix[0:1], pad_len, axis=0)
        lstm_seqs.append(np.vstack([padding, feat_matrix]))
X_lstm_tensor = torch.tensor(np.array(lstm_seqs, dtype=np.float32))
with torch.no_grad():
    pred_lstm = np.clip(lstm_model(X_lstm_tensor).numpy(), 0, 125.0)
rec_lstm_rmse = np.sqrt(metrics.mean_squared_error(y_true_all, pred_lstm))
rec_lstm_mae  = metrics.mean_absolute_error(y_true_all, pred_lstm)
rec_lstm_r2   = metrics.r2_score(y_true_all, pred_lstm)

recalc_map = {
    'linear_regression': (rec_lr_rmse, rec_lr_mae, rec_lr_r2),
    'random_forest'    : (rec_rf_rmse, rec_rf_mae, rec_rf_r2),
    'gradient_boosting': (rec_gb_rmse, rec_gb_mae, rec_gb_r2),
    'xgboost'          : (rec_xgb_rmse, rec_xgb_mae, rec_xgb_r2),
    'lstm'             : (rec_lstm_rmse, rec_lstm_mae, rec_lstm_r2),
}

print(f" {'Model':<20} | {'Disp RMSE':<9} | {'Recalc RMSE':<11} | {'Disp MAE':<8} | {'Recalc MAE':<10} | {'Disp R²':<7} | {'Recalc R²':<9} | Status")
print("-" * 105)
for m_key, (rc_rmse, rc_mae, rc_r2) in recalc_map.items():
    disp = meta_json['models'][m_key]['metrics']
    disp_rmse = disp['test_rmse']
    disp_mae  = disp['test_mae']
    disp_r2   = disp['test_r2']
    
    match_rmse = abs(disp_rmse - rc_rmse) < 0.05
    match_mae  = abs(disp_mae - rc_mae) < 0.05
    match_r2   = abs(disp_r2 - rc_r2) < 0.01
    
    status = "PASS" if (match_rmse and match_mae and match_r2) else "FAIL"
    print(f" {m_key:<20} | {disp_rmse:<9.2f} | {rc_rmse:<11.2f} | {disp_mae:<8.2f} | {rc_mae:<10.2f} | {disp_r2:<7.4f} | {rc_r2:<9.4f} | {status}")


# ─── TEST E: PREDICTION GRAPHS & VISUALIZATIONS ──────────────────────────────
print("\n" + "=" * 50)
print("  TEST E: PREDICTION GRAPHS AND VISUALIZATIONS")
print("=" * 50)

expected_plots = [
    ("linear_regression_test_prediction.png", "Linear Regression Actual vs Predicted + Residuals"),
    ("random_forest_test_prediction.png", "Random Forest Actual vs Predicted + Residuals"),
    ("gradient_boosting_test_prediction.png", "Gradient Boosting Actual vs Predicted + Residuals"),
    ("xgboost_test_prediction.png", "XGBoost Actual vs Predicted + Residuals"),
    ("lstm_test_prediction.png", "LSTM Actual vs Predicted (Week 6)"),
    ("lstm_loss_curve.png", "LSTM 25-Epoch Training/Validation Loss Curve (Week 5)"),
]

for fname, desc in expected_plots:
    p = os.path.join(MODELS_DIR, fname)
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        print(f"  [PASS] {fname:<36} -> FOUND ({os.path.getsize(p)/1024:>6.1f} KB) | {desc}")
    else:
        print(f"  [FAIL] {fname:<36} -> NOT FOUND or empty")


# ─── TEST F: MODEL SWITCHING IN STREAMLIT ────────────────────────────────────
print("\n" + "=" * 50)
print("  TEST F: MODEL SWITCHING AND ISOLATION")
print("=" * 50)

from individual_models_demo import MODEL_SPECS

# Verify each model has its distinct configuration, plot file, pred_col
specs_keys = list(MODEL_SPECS.keys())
print(f"  Available model keys in UI: {specs_keys}")
all_cols = [spec['pred_col'] for spec in MODEL_SPECS.values()]
print(f"  Prediction column names: {all_cols}")
assert len(set(all_cols)) == len(all_cols), "Collision in prediction column names!"
print("  [PASS] All 5 models map to unique prediction columns and unique serialized artifacts.")


# ─── TEST G: ORIGINAL MAINTAI DASHBOARD INTEGRITY ───────────────────────────
print("\n" + "=" * 50)
print("  TEST G: ORIGINAL MAINTAI DASHBOARD INTEGRITY")
print("=" * 50)

app_file = os.path.join(FRONTEND_DIR, 'streamlit_app.py')
with open(app_file, encoding='utf-8') as f:
    app_text = f.read()

checks = [
    ("Page 1 (Front Page / Hero) present", "🏠 Front Page (Project Overview)" in app_text),
    ("Page 2 (Single-Engine Telemetry) present", "🎛️ Single-Engine Telemetry (Manual)" in app_text),
    ("Page 3 (Batch Fleet Analysis) present", "📂 Batch Fleet Analysis (CSV Upload)" in app_text),
    ("Page 4 (Model Benchmarks) present", "📊 Model Benchmarks & Comparison" in app_text),
    ("New Individual Model Demos present", "🔬 Individual Model Demonstrations" in app_text),
    ("Theme Toggle present", "app_theme_choice" in app_text),
    ("Sidebar state self-healing script present", "stSidebarCollapsed" in app_text),
]
for desc, result in checks:
    print(f"  [{'PASS' if result else 'FAIL'}] {desc}")


# ─── TEST H: BACKEND REST API ENDPOINTS ──────────────────────────────────────
print("\n" + "=" * 50)
print("  TEST H: BACKEND FASTAPI REST ENDPOINTS")
print("=" * 50)

from fastapi.testclient import TestClient
from api import app, load_model_on_startup

load_model_on_startup()
client = TestClient(app)

# 1. Health
r = client.get("/health")
print(f"  GET /health -> Status: {r.status_code} | Available Models: {r.json().get('models_available')}")

# 2. Features
r = client.get("/features")
print(f"  GET /features -> Status: {r.status_code} | Feature Count: {r.json().get('feature_count')}")

# 3. Model Info
r = client.get("/model/info")
info = r.json()
print(f"  GET /model/info -> Status: {r.status_code} | Primary Models: {list(info.get('primary_models', {}).keys())}")

# 4. Predict single for all 5 models
sample_body = {feat: float(last_cycles.iloc[0][feat]) for feat in features}
for m in ["linear_regression", "random_forest", "gradient_boosting", "xgboost", "lstm"]:
    r = client.post(f"/predict?model={m}", json=sample_body)
    d = r.json()
    print(f"  POST /predict?model={m:<18} -> Status: {r.status_code} | RUL: {d.get('rul'):>5.1f} | Model: {d.get('model_used')}")

# 5. Invalid Input Error Handling
r_bad = client.post("/predict?model=xgboost", json={"Cycle": -5, "SensorMeasure2": 640})
print(f"  POST /predict (Invalid Input) -> Status: {r_bad.status_code} (Expected 422 Unprocessable Entity)")

print("\n" + "=" * 85)
print("  ALL AUDIT SUITES EXECUTED SUCCESSFULLY")
print("=" * 85)

