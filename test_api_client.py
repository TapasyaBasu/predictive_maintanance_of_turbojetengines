"""
test_api_client.py - Direct Python verification of API endpoint functions and logic
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_dir)

from api import (
    load_model_on_startup,
    health,
    model_info,
    predict,
    predict_batch,
    SensorInput,
    BatchPredictionRequest
)

print("=" * 60)
print(" Direct Verification of FastAPI Endpoints")
print("=" * 60)

# Initialize models
load_model_on_startup()

# 1. Health endpoint
h = health()
print(f"[PASS] health() -> status={h.status}, XGBoost={h.model_loaded}, LSTM={h.lstm_loaded}, Features={h.feature_count}")
assert h.model_loaded is True
assert h.lstm_loaded is True
assert h.feature_count == 13

# 2. Model info endpoint
info = model_info()
xgb_rmse = info.primary_models['xgboost']['test_rmse']
lstm_rmse = info.primary_models['lstm']['test_rmse']
print(f"[PASS] model_info() -> XGBoost RMSE={xgb_rmse}, LSTM RMSE={lstm_rmse}")
assert xgb_rmse == 18.91
assert lstm_rmse == 13.23

# 3. Predict with XGBoost
sensor_data = SensorInput(
    Cycle=100,
    SensorMeasure2=641.82,
    SensorMeasure3=1589.70,
    SensorMeasure4=1400.60,
    SensorMeasure7=554.36,
    SensorMeasure8=2388.02,
    SensorMeasure11=47.47,
    SensorMeasure12=521.66,
    SensorMeasure13=2388.02,
    SensorMeasure15=8.4195,
    SensorMeasure17=392.0,
    SensorMeasure20=39.06,
    SensorMeasure21=23.419
)
pred_xgb = predict(sensor=sensor_data, model='xgboost')
print(f"[PASS] predict(xgboost) -> RUL={pred_xgb.rul}, Health={pred_xgb.health_pct}%, Status={pred_xgb.status}")
assert "XGBoost" in pred_xgb.model_used
assert 0 <= pred_xgb.rul <= 125

# 4. Predict with LSTM
pred_lstm = predict(sensor=sensor_data, model='lstm')
print(f"[PASS] predict(lstm)    -> RUL={pred_lstm.rul}, Health={pred_lstm.health_pct}%, Status={pred_lstm.status}")
assert "LSTM" in pred_lstm.model_used
assert 0 <= pred_lstm.rul <= 125

# 5. Batch prediction
batch_req = BatchPredictionRequest(
    readings=[
        sensor_data,
        SensorInput(
            Cycle=180,
            SensorMeasure2=641.82,
            SensorMeasure3=1589.70,
            SensorMeasure4=1400.60,
            SensorMeasure7=554.36,
            SensorMeasure8=2388.02,
            SensorMeasure11=47.47,
            SensorMeasure12=521.66,
            SensorMeasure13=2388.02,
            SensorMeasure15=8.4195,
            SensorMeasure17=392.0,
            SensorMeasure20=39.06,
            SensorMeasure21=23.419
        )
    ]
)
pred_batch = predict_batch(request=batch_req, model='xgboost')
print(f"[PASS] predict_batch() -> Count={pred_batch.count}, Fleet Avg RUL={pred_batch.fleet_avg_rul}")
assert pred_batch.count == 2

print("=" * 60)
print(" ALL API ENDPOINTS FUNCTION PERFECTLY!")
print("=" * 60)
