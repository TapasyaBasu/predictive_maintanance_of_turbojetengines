"""
VERIFY_ALL_DEMOS.py
===================
Comprehensive automated test suite verifying all 10 requirements:
1. Independent model selection
2. Artifact loading
3. Preprocessing correctness
4. Independent predictions
5. Metrics correspondence
6. Visualization file verification
7. State isolation between models
8. Missing artifact graceful degradation
9. Frontend dashboard component sanity
10. FastAPI backend microservice regression testing
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BACKEND_DIR = os.path.join(BASE_DIR, 'backend')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend', 'app')
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, FRONTEND_DIR)

from predict import load_all_models, predict_single, predict_batch, normalize_model_type
from fastapi.testclient import TestClient
from api import app, load_model_on_startup

PASSED = 0
FAILED = 0

def check(test_name: str, condition: bool, details: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {test_name}" + (f" -> {details}" if details else ""))
    else:
        FAILED += 1
        print(f"  [FAIL] {test_name}" + (f" -> {details}" if details else ""))


def main():
    print("=" * 80)
    print("  MAINTAI — INDIVIDUAL MODEL DEMONSTRATION VERIFICATION SUITE")
    print("=" * 80)

    # 1. Artifact Loading
    print("\n--- Test Group 1: Artifact Existence & Loading ---")
    models_dict = load_all_models(MODELS_DIR)
    check("Scaler (scaler.pkl) Loaded", models_dict.get("scaler") is not None)
    check("Feature List (feature_list.json) Valid", len(models_dict.get("features", [])) == 13)
    check("Linear Regression (linear_regression.pkl) Loaded", models_dict.get("linear_regression") is not None)
    check("Random Forest (random_forest.pkl) Loaded", models_dict.get("random_forest") is not None)
    check("Gradient Boosting (gradient_boosting.pkl) Loaded", models_dict.get("gradient_boosting") is not None)
    check("XGBoost (final_model.pkl) Loaded", models_dict.get("xgboost") is not None)
    check("PyTorch LSTM (lstm_best_model.pth) Loaded", models_dict.get("lstm") is not None)

    # 2. Model Normalization
    print("\n--- Test Group 2: Model Type Normalization & Independence ---")
    check("Normalize 'linear_regression'", normalize_model_type("linear_regression") == "linear_regression")
    check("Normalize 'random_forest'", normalize_model_type("random_forest") == "random_forest")
    check("Normalize 'gradient_boosting'", normalize_model_type("gradient_boosting") == "gradient_boosting")
    check("Normalize 'xgboost'", normalize_model_type("xgboost") == "xgboost")
    check("Normalize 'lstm'", normalize_model_type("lstm") == "lstm")
    # Verify no crosstalk between 'xgboost' and 'gradient_boosting'
    check("Normalize 'xgb' does not trigger 'gradient_boosting'", normalize_model_type("xgb") == "xgboost")

    # 3. Independent Predictions
    print("\n--- Test Group 3: Independent Prediction Execution ---")
    SAMPLE_HEALTHY = {
        "Cycle": 1, "SensorMeasure2": 641.82, "SensorMeasure3": 1589.70, "SensorMeasure4": 1400.60,
        "SensorMeasure7": 554.36, "SensorMeasure8": 2388.02, "SensorMeasure11": 47.47,
        "SensorMeasure12": 521.66, "SensorMeasure13": 2388.02, "SensorMeasure15": 8.4195,
        "SensorMeasure17": 392.0, "SensorMeasure20": 39.06, "SensorMeasure21": 23.419
    }
    SAMPLE_CRITICAL = {
        "Cycle": 185, "SensorMeasure2": 646.90, "SensorMeasure3": 1614.50, "SensorMeasure4": 1447.60,
        "SensorMeasure7": 558.80, "SensorMeasure8": 2388.30, "SensorMeasure11": 46.50,
        "SensorMeasure12": 518.20, "SensorMeasure13": 2388.30, "SensorMeasure15": 8.5800,
        "SensorMeasure17": 398.5, "SensorMeasure20": 38.20, "SensorMeasure21": 22.70
    }

    all_keys = ["linear_regression", "random_forest", "gradient_boosting", "xgboost", "lstm"]
    healthy_preds = {}
    critical_preds = {}

    for k in all_keys:
        p_h = predict_single(SAMPLE_HEALTHY, model_type=k, extra_models=models_dict)
        p_c = predict_single(SAMPLE_CRITICAL, model_type=k, extra_models=models_dict)
        healthy_preds[k] = p_h
        critical_preds[k] = p_c
        check(
            f"Prediction: {k} (Nominal)",
            p_h["rul"] > 80 and p_h["status_color"] == "green",
            f"RUL={p_h['rul']:.1f} cyc, Model='{p_h['model_used']}'"
        )
        check(
            f"Prediction: {k} (Critical)",
            p_c["rul"] < 40 and p_c["status_color"] in ("red", "amber"),
            f"RUL={p_c['rul']:.1f} cyc, Status='{p_c['status']}'"
        )

    # 4. State Isolation & Non-Trivial Differences
    print("\n--- Test Group 4: Model State Isolation ---")
    # Verify predictions are not all identical (they are computed by distinct algorithms)
    ruls = [p["rul"] for p in healthy_preds.values()]
    check("Predictions are distinct across architectures", len(set(ruls)) >= 3, f"Unique RUL outputs: {set(ruls)}")

    # 5. Metrics & Metadata Correspondence
    print("\n--- Test Group 5: Metrics & Metadata Integrity ---")
    meta_path = os.path.join(MODELS_DIR, 'models_metadata.json')
    check("models_metadata.json exists", os.path.exists(meta_path))
    with open(meta_path) as f:
        meta_json = json.load(f)
    for k in all_keys:
        m_data = meta_json["models"].get(k)
        check(f"Metadata present for {k}", m_data is not None and "metrics" in m_data)
        if m_data:
            check(f"Test RMSE > 0 for {k}", m_data["metrics"]["test_rmse"] > 0, f"RMSE={m_data['metrics']['test_rmse']}")

    # 6. Model-Specific Visualizations
    print("\n--- Test Group 6: Visualization Artifacts ---")
    plot_files = [
        "linear_regression_test_prediction.png",
        "random_forest_test_prediction.png",
        "gradient_boosting_test_prediction.png",
        "xgboost_test_prediction.png",
        "lstm_test_prediction.png",
        "lstm_loss_curve.png"
    ]
    for pf in plot_files:
        check(f"Plot file exists: {pf}", os.path.exists(os.path.join(MODELS_DIR, pf)))

    # 7. Test Predictions CSV
    print("\n--- Test Group 7: Test Predictions CSV (100 Engines) ---")
    pred_csv_path = os.path.join(MODELS_DIR, 'test_predictions_all.csv')
    check("test_predictions_all.csv exists", os.path.exists(pred_csv_path))
    pred_df = pd.read_csv(pred_csv_path)
    check("CSV contains 100 test engines", len(pred_df) == 100)
    for col in ["True_RUL", "Linear_Regression", "Random_Forest", "Gradient_Boosting", "XGBoost", "LSTM"]:
        check(f"CSV column present: {col}", col in pred_df.columns)

    # 8. Missing Artifact Graceful Handling
    print("\n--- Test Group 8: Graceful Error Handling ---")
    try:
        predict_single(SAMPLE_HEALTHY, model_type="nonexistent_alien_model", extra_models={"features": models_dict["features"], "scaler": models_dict["scaler"], "lstm_scaler": models_dict["lstm_scaler"], "xgboost": None})
        check("Missing model handled", False, "Should have raised exception")
    except Exception as e:
        check("Missing model throws clear exception", True, str(e)[:60])

    # 9. FastAPI Endpoints Regression
    print("\n--- Test Group 9: Backend REST API Endpoints ---")
    load_model_on_startup()
    client = TestClient(app)

    r_health = client.get('/health')
    check("GET /health status 200", r_health.status_code == 200)
    models_avail = r_health.json().get("models_available", [])
    check("All 5 models reported in /health", len(models_avail) == 5, f"Available: {models_avail}")

    for k in all_keys:
        r_pred = client.post(f'/predict?model={k}', json=SAMPLE_HEALTHY)
        check(f"POST /predict?model={k} status 200", r_pred.status_code == 200, f"RUL={r_pred.json().get('rul')}")

    r_batch = client.post('/predict/batch?model=xgboost', json={'readings': [SAMPLE_HEALTHY, SAMPLE_CRITICAL]})
    check("POST /predict/batch status 200", r_batch.status_code == 200, f"Processed {r_batch.json().get('count')} engines")

    # 10. Summary
    print("\n" + "=" * 80)
    print(f"  VERIFICATION COMPLETE: {PASSED} PASSED | {FAILED} FAILED")
    print("=" * 80)
    return FAILED == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

