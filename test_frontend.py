"""
test_streamlit_inprocess.py — Comprehensive in-process execution test of streamlit_app.py
Mocks streamlit API calls to verify that all data structures, functions,
calculations, Plotly chart generations, and render blocks execute without errors.
"""
import sys
import os
from unittest.mock import MagicMock

# Create a mock streamlit module
class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value

mock_st = MagicMock()
mock_st.session_state = SessionState({
    'nav_page': '🏠 Front Page (Project Overview)',
    'sensor_vals': {
        'Cycle': 50,
        'SensorMeasure2': 641.82, 'SensorMeasure3': 1589.70,
        'SensorMeasure4': 1400.60, 'SensorMeasure7': 554.36,
        'SensorMeasure8': 2388.02, 'SensorMeasure11': 47.47,
        'SensorMeasure12': 521.66, 'SensorMeasure13': 2388.02,
        'SensorMeasure15': 8.4195, 'SensorMeasure17': 392.0,
        'SensorMeasure20': 39.06, 'SensorMeasure21': 23.419
    }
})

# Context managers like with st.sidebar, with st.columns, with st.form
mock_st.sidebar = MagicMock()
mock_st.sidebar.__enter__ = MagicMock(return_value=mock_st.sidebar)
mock_st.sidebar.__exit__ = MagicMock(return_value=None)

mock_col = MagicMock()
mock_col.__enter__ = MagicMock(return_value=mock_col)
mock_col.__exit__ = MagicMock(return_value=None)
def mock_columns(spec, *args, **kwargs):
    n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
    return [mock_col for _ in range(n)]

mock_st.columns.side_effect = mock_columns
mock_st.tabs.side_effect = lambda spec, *args, **kwargs: [mock_col for _ in range(len(spec))]

def mock_input(label, min_value=0.0, max_value=100.0, value=50.0, *args, **kwargs):
    return value
mock_st.slider.side_effect = mock_input
mock_st.number_input.side_effect = mock_input

mock_form = MagicMock()
mock_form.__enter__ = MagicMock(return_value=mock_form)
mock_form.__exit__ = MagicMock(return_value=None)
mock_st.form.return_value = mock_form

mock_st.cache_data = lambda *args, **kwargs: (lambda f: f)
mock_st.cache_resource = lambda *args, **kwargs: (lambda f: f)
mock_st.rerun = MagicMock()

# Mock requests so network doesn't block
import requests
from unittest.mock import patch

mock_health_resp = MagicMock()
mock_health_resp.status_code = 200
mock_health_resp.json.return_value = {
    'status': 'ok',
    'model_loaded': True,
    'lstm_loaded': True,
    'models_available': ['xgboost', 'lstm'],
    'feature_count': 13
}

mock_predict_resp = MagicMock()
mock_predict_resp.status_code = 200
mock_predict_resp.json.return_value = {
    'rul': 105.4,
    'status': 'NOMINAL OPERATION',
    'status_color': 'green',
    'action_directive': 'Standard routine monitoring',
    'health_pct': 84.3,
    'model_used': 'XGBoost Regressor'
}

requests.get = MagicMock(return_value=mock_health_resp)
requests.post = MagicMock(return_value=mock_predict_resp)

sys.modules['streamlit'] = mock_st

print("=" * 65)
print("  STREAMLIT APPLICATION IN-PROCESS EXECUTION TEST")
print("=" * 65)

# Read and execute streamlit_app.py
BASE = os.path.dirname(os.path.abspath(__file__))
app_file = os.path.join(BASE, "frontend", "app", "streamlit_app.py")
with open(app_file, encoding="utf-8") as f:
    app_code = f.read()

# Test 1: Page 1 (Front Page)
print("\n[1/4] Executing Page 1: Front Page (Project Overview)...")
mock_st.session_state['nav_page'] = "🏠 Front Page (Project Overview)"
exec(compile(app_code, app_file, 'exec'), {'__file__': os.path.abspath(app_file)})
print("   [PASS] Front Page executed cleanly.")

# Test 2: Page 2 (Single-Engine Telemetry)
print("\n[2/4] Executing Page 2: Single-Engine Telemetry (Manual)...")
mock_st.session_state['nav_page'] = "🎛️ Single-Engine Telemetry (Manual)"
mock_st.radio.return_value = "🎛️ Single-Engine Telemetry (Manual)"
exec(compile(app_code, app_file, 'exec'), {'__file__': os.path.abspath(app_file)})
print("   [PASS] Telemetry diagnostic calculations & plots executed cleanly.")

# Test 3: Page 3 (Batch Fleet Analysis)
print("\n[3/4] Executing Page 3: Batch Fleet Analysis (CSV Upload)...")
mock_st.session_state['nav_page'] = "📂 Batch Fleet Analysis (CSV Upload)"
mock_st.radio.return_value = "📂 Batch Fleet Analysis (CSV Upload)"
mock_st.file_uploader.return_value = None  # No file uploaded yet
exec(compile(app_code, app_file, 'exec'), {'__file__': os.path.abspath(app_file)})
print("   [PASS] Batch Fleet page (no file / empty state) executed cleanly.")

# Also test with a sample batch CSV upload
import io
csv_buffer = io.StringIO("""Cycle,SensorMeasure2,SensorMeasure3,SensorMeasure4,SensorMeasure7,SensorMeasure8,SensorMeasure11,SensorMeasure12,SensorMeasure13,SensorMeasure15,SensorMeasure17,SensorMeasure20,SensorMeasure21
50,641.82,1589.70,1400.60,554.36,2388.02,47.47,521.66,2388.02,8.4195,392.0,39.06,23.419
100,642.50,1590.20,1405.10,553.80,2388.05,47.60,521.20,2388.05,8.4300,393.0,38.90,23.350
""")
mock_st.file_uploader.return_value = csv_buffer
mock_predict_resp.json.return_value = {
    'count': 2,
    'predictions': [
        {'rul': 105.4, 'health_pct': 84.3, 'status': 'NOMINAL OPERATION', 'status_color': 'green', 'action_directive': 'Monitoring', 'cycle_age': 50, 'model_used': 'XGBoost'},
        {'rul': 75.2, 'health_pct': 60.1, 'status': 'MAINTENANCE ADVISORY', 'status_color': 'amber', 'action_directive': 'NDI inspection', 'cycle_age': 100, 'model_used': 'XGBoost'}
    ],
    'fleet_avg_rul': 90.3,
    'fleet_min_rul': 75.2,
    'critical_count': 0,
    'model_used': 'XGBoost Regressor'
}
exec(compile(app_code, app_file, 'exec'), {'__file__': os.path.abspath(app_file)})
print("   [PASS] Batch Fleet page (with 2-engine CSV processed) executed cleanly.")

# Test 4: Page 4 (Model Benchmarks & Comparison)
print("\n[4/4] Executing Page 4: Model Benchmarks & Comparison...")
mock_st.session_state['nav_page'] = "📊 Model Benchmarks & Comparison"
mock_st.radio.return_value = "📊 Model Benchmarks & Comparison"
exec(compile(app_code, app_file, 'exec'), {'__file__': os.path.abspath(app_file)})
print("   [PASS] Model Benchmarks Plotly generation & tables executed cleanly.")

print("\n" + "=" * 65)
print("  ALL 4 STREAMLIT PAGES EXECUTED WITH 0 RUNTIME/SYNTAX ERRORS!")
print("=" * 65)
