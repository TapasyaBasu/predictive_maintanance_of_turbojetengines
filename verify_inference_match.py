"""
verify_inference_match.py — Direct proof that API and Standalone produce identical outputs
for both XGBoost and LSTM when given identical input values.
"""
import sys
import os

backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
sys.path.insert(0, backend_dir)

from predict import predict_single as predict_standalone
from api import _predict_single_core, SensorInput, load_model_on_startup

load_model_on_startup()

sample = {
    'Cycle': 100,
    'SensorMeasure2': 641.82, 'SensorMeasure3': 1589.70,
    'SensorMeasure4': 1400.60, 'SensorMeasure7': 554.36,
    'SensorMeasure8': 2388.02, 'SensorMeasure11': 47.47,
    'SensorMeasure12': 521.66, 'SensorMeasure13': 2388.02,
    'SensorMeasure15': 8.4195, 'SensorMeasure17': 392.0,
    'SensorMeasure20': 39.06, 'SensorMeasure21': 23.419
}

# Standalone predict with Cycle=100
standalone_xgb = predict_standalone(sample, model_type='xgboost')
standalone_lstm = predict_standalone(sample, model_type='lstm')

# API predict with Cycle=100
sensor_obj = SensorInput(**sample)
api_xgb = _predict_single_core(sensor_obj, model_type='xgboost')
api_lstm = _predict_single_core(sensor_obj, model_type='lstm')

print("=" * 65)
print("  API INFERENCE VS STANDALONE INFERENCE VERIFICATION")
print("=" * 65)

print("\n--- 1. XGBoost Comparison (Input: Cycle=100) ---")
print(f"  Standalone: RUL = {standalone_xgb['rul']:.2f} cycles | Health = {standalone_xgb['health_pct']:.2f}% | Model: {standalone_xgb['model_used']}")
print(f"  API Core:   RUL = {api_xgb.rul:.2f} cycles | Health = {api_xgb.health_pct:.2f}% | Model: {api_xgb.model_used}")
assert standalone_xgb['rul'] == api_xgb.rul, "XGBoost RUL mismatch!"
assert standalone_xgb['health_pct'] == api_xgb.health_pct, "XGBoost Health mismatch!"
print("  [MATCH] XGBoost outputs are IDENTICAL.")

print("\n--- 2. LSTM Comparison (Input: Cycle=100) ---")
print(f"  Standalone: RUL = {standalone_lstm['rul']:.2f} cycles | Health = {standalone_lstm['health_pct']:.2f}% | Model: {standalone_lstm['model_used']}")
print(f"  API Core:   RUL = {api_lstm.rul:.2f} cycles | Health = {api_lstm.health_pct:.2f}% | Model: {api_lstm.model_used}")
assert standalone_lstm['rul'] == api_lstm.rul, "LSTM RUL mismatch!"
assert standalone_lstm['health_pct'] == api_lstm.health_pct, "LSTM Health mismatch!"
print("  [MATCH] PyTorch LSTM outputs are IDENTICAL.")

print("\n--- 3. Explaining Cycle 1 vs Cycle 100 Output Differences ---")
sample_c1 = dict(sample)
sample_c1['Cycle'] = 1
s_c1_xgb = predict_standalone(sample_c1, model_type='xgboost')
s_c1_lstm = predict_standalone(sample_c1, model_type='lstm')

print(f"  When Cycle = 1 (Engine just entered service):")
print(f"     XGBoost RUL = {s_c1_xgb['rul']:.2f} cycles | LSTM RUL = {s_c1_lstm['rul']:.2f} cycles")
print(f"  When Cycle = 100 (Engine has accumulated 100 missions):")
print(f"     XGBoost RUL = {standalone_xgb['rul']:.2f} cycles | LSTM RUL = {standalone_lstm['rul']:.2f} cycles")
print("=" * 65)
