# MaintAI — Backend

FastAPI REST server that serves the XGBoost RUL (Remaining Useful Life) predictor for NASA CMAPSS turbofan engines.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service liveness / info |
| GET | `/landing` | Serves standalone aerospace HTML landing page |
| GET | `/health` | Model load status, feature count |
| GET | `/features` | Ordered list of required input features |
| GET | `/model/info` | Full model metadata and threshold definitions |
| POST | `/predict` | Single-engine RUL prediction |
| POST | `/predict/batch` | Batch fleet prediction (up to 5 000 rows) |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/redoc` | ReDoc API reference |

## Quick Start

```bash
# From the project root
pip install -r backend/requirements.txt
python START_BACKEND.py
```

API is available at: `http://localhost:8000`  
Swagger docs: `http://localhost:8000/docs`

## Example — single predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Cycle": 1,
    "SensorMeasure2": 641.82, "SensorMeasure3": 1589.70,
    "SensorMeasure4": 1400.60, "SensorMeasure7": 554.36,
    "SensorMeasure8": 2388.02, "SensorMeasure11": 47.47,
    "SensorMeasure12": 521.66, "SensorMeasure13": 2388.02,
    "SensorMeasure15": 8.4195, "SensorMeasure17": 392.0,
    "SensorMeasure20": 39.06, "SensorMeasure21": 23.419
  }'
```

## Status thresholds

| Status | RUL Range | Colour |
|--------|-----------|--------|
| NOMINAL OPERATION | > 80 cycles | 🟢 green |
| MAINTENANCE ADVISORY | 30–80 cycles | 🟡 amber |
| CRITICAL EXHAUSTION (AOG) | < 30 cycles | 🔴 red |

## Re-training

```bash
cd backend
python train.py
```

## Files

| File | Purpose |
|------|---------|
| `api.py` | FastAPI application with all endpoints |
| `predict.py` | Standalone prediction logic (importable) |
| `train.py` | Training pipeline (refactored from TRAIN_MODEL.py) |
| `requirements.txt` | Backend Python dependencies |
