# 🚀 MaintAI: Predictive Maintenance & RUL Prediction for Turbofan Engines

> **"Fix it before it breaks."**  
> MaintAI uses Machine Learning and Deep Learning to predict the **Remaining Useful Life (RUL)** of NASA C-MAPSS turbofan engines.

---

## 📌 Project Overview

**MaintAI** combines Exploratory Data Analysis, classical parametric models (Linear Regression), tree ensembles (Random Forest, Gradient Boosting, XGBoost), and deep recurrent neural networks (PyTorch 2-Layer LSTM) to predict Remaining Useful Life (RUL) of turbofan jet engines — deployed as a full-stack engineering application with a **FastAPI REST backend** and an interactive **Streamlit web dashboard**.

| Feature | Description |
|---|---|
| 📁 Dataset | NASA C-MAPSS (FD001–FD004) |
| 📈 Baseline ML Model | Ordinary Least Squares Linear Regression (Test RMSE: 21.61 cycles) |
| 🌲 Ensemble Bagging | Random Forest Regressor (100 Trees, Test RMSE: 17.51 cycles) |
| 🚀 Sequential Boosting | Gradient Boosting Regressor (150 Trees, Test RMSE: 17.21 cycles) |
| ⚡ Primary Tree Model | Tuned XGBoost Regressor (400 Trees, Test RMSE: 18.91 cycles) |
| 🧬 Deep Learning Model | 2-Layer PyTorch LSTM Network (Test RMSE: 13.23 cycles) |
| 📊 Dashboard | Streamlit — manual sensor input + CSV fleet upload + Individual Model Demos |
| 🚀 Backend | FastAPI REST API — single & batch inference supporting all 5 models |
| 🌐 Landing | Standalone responsive HTML landing page (`/landing`) |
| 📈 Metrics | Verified RMSE, MAE, R² across all 100 test engines |
| 🌐 Deployment | Local full-stack microservice (FastAPI + Streamlit) |

---

## 🗂️ Repository Structure

```
NASA_Turbofan_JetEngine_RUL/
├── backend/
│   ├── api.py                   # FastAPI REST API server (LR, RF, GB, XGBoost, LSTM)
│   ├── predict.py               # Unified inference engine for all 5 models & scalers
│   ├── train.py                 # Modular training script
│   ├── requirements.txt         # Backend dependencies
│   └── README.md                # API endpoint documentation
├── frontend/
│   ├── landing/
│   │   └── index.html           # Standalone aerospace HTML landing page
│   ├── app/
│   │   ├── streamlit_app.py     # Main Streamlit dashboard (5 pages, multi-model)
│   │   └── individual_models_demo.py # Dedicated 4-tab model demonstration UI
│   ├── .streamlit/
│   │   └── config.toml          # Streamlit theme configuration
│   └── requirements.txt         # Frontend dependencies
├── Dataset/                     # NASA C-MAPSS raw files (FD001–FD004)
├── models/                      # Saved weights, scalers, metadata, and benchmark plots
│   ├── linear_regression.pkl    # Trained Ordinary Least Squares baseline
│   ├── random_forest.pkl        # Trained Random Forest Regressor (compressed)
│   ├── gradient_boosting.pkl    # Trained Gradient Boosting Regressor
│   ├── final_model.pkl          # Trained XGBoost Regressor
│   ├── lstm_best_model.pth      # Trained PyTorch 2-layer LSTM model
│   ├── lstm_scaler.pkl          # Dedicated StandardScaler for LSTM
│   ├── scaler.pkl               # StandardScaler for tabular ML
│   ├── feature_list.json        # 13 selected sensor features
│   ├── models_metadata.json     # Hyperparameters, OLS coefficients, Gini importances
│   ├── model_comparison.csv     # Verified benchmark metrics across all models
│   ├── test_results.csv         # Verified test evaluation on 100 test engines
│   ├── test_predictions_all.csv # Ground truth and predictions for all 100 test engines
│   ├── *_test_prediction.png    # Actual vs Predicted RUL plots for each model
│   └── lstm_loss_curve.png      # Training & validation loss across 25 epochs
├── notebooks/                   # Project analysis and training notebooks
│   ├── Exploratory Data Analysis of Turbo Engine data set.ipynb
│   ├── Making_Training and Test_Data_set.ipynb
│   ├── Week4_Feature_Engineering.ipynb
│   ├── Week5_XGBoost_LinearRegression.ipynb
│   ├── Week5_LSTM_RUL_Prediction.ipynb
│   └── Week6_Model_Evaluation_Comparison.ipynb
├── RUN_ALL.py                   # One-click launcher for backend + frontend
├── RUN_PROJECT.py               # Master CLI interactive runner
├── RUN_INDIVIDUAL_MODEL.py      # Standalone runner with interactive model selector
├── RUN_LINEAR_REGRESSION.py     # 1-Click runner for Linear Regression alone
├── RUN_RANDOM_FOREST.py         # 1-Click runner for Random Forest alone
├── RUN_GRADIENT_BOOSTING.py     # 1-Click runner for Gradient Boosting alone
├── RUN_XGBOOST.py               # 1-Click runner for XGBoost alone
├── RUN_LSTM.py                  # 1-Click runner for PyTorch LSTM alone
├── TRAIN_ADDITIONAL_MODELS.py   # Training script for LR, RF, GB with verified evaluation
├── START_BACKEND.py             # Dedicated backend launcher
├── START_FRONTEND.py            # Dedicated frontend launcher
├── TRAIN_MODEL.py               # Trains XGBoost regressor
├── TRAIN_LSTM.py                # Trains sequential LSTM neural network
├── TEST_MODEL.py                # Evaluates all models on test engines
├── VERIFY_ALL_DEMOS.py          # Complete 58-test verification suite
├── thorough_project_test.py     # End-to-end model and metric verification suite
├── test_api_client.py           # Verification script for FastAPI endpoints
├── test_frontend.py             # Verification script for Streamlit frontend
├── verify_all_subsets.py        # Verification script for FD001–FD004 subsets
├── verify_inference_match.py    # Verification script for API vs standalone inference
├── requirements.txt             # Global Python requirements
└── README.md
```

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────────┐
                    │   Browser / Client Application   │
                    └────────┬─────────────────────────┘
                             │
             ┌───────────────┼──────────────────┐
             │               │                  │
    ┌────────▼─────┐  ┌──────▼──────┐  ┌───────▼───────┐
    │ Landing Page │  │  Streamlit  │  │  FastAPI Docs │
    │  index.html  │  │  Dashboard  │  │  /docs        │
    │  :8000/landing│ │  :8501      │  │  :8000/docs   │
    └──────────────┘  └──────┬──────┘  └───────────────┘
                             │ HTTP (requests)
                    ┌────────▼─────────────────────────┐
                    │     FastAPI Microservice :8000   │
                    │     GET  /health                 │
                    │     POST /predict?model=...      │
                    │     POST /predict/batch?model=...│
                    │     GET  /features               │
                    │     GET  /model/info             │
                    └────────┬─────────────────────────┘
                             │
                    ┌────────▼─────────────────────────┐
                    │             models/              │
                    │  ├── final_model.pkl (XGBoost)   │
                    │  ├── lstm_best_model.pth (LSTM)  │
                    │  ├── lstm_scaler.pkl             │
                    │  ├── scaler.pkl                  │
                    │  └── feature_list.json           │
                    └──────────────────────────────────┘
```

---

## 📅 8-Week Project Syllabus & Progress

| Week | Milestone / Deliverables | Status |
|---|---|---|
| 1 | Problem formulation, aerospace failure modes, dataset setup | Completed |
| 2 | Exploratory Data Analysis, sensor correlation, flatline sensor filtering | Completed |
| 3 | Signal normalization, piecewise linear RUL target clipping (cap = 125) | Completed |
| 4 | Temporal feature engineering & sliding time-window sequence generation (W = 30) | Completed |
| 5 | Model Training: Classical ML (XGBoost, RF) + Deep Learning (PyTorch 2-Layer LSTM) | Completed |
| 6 | Model Evaluation: RMSE, MAE, R², loss curves, and actual vs. predicted plots | Completed |
| 7 | Interactive Streamlit GUI: Telemetry diagnosis, gauge display, batch CSV fleet upload | Completed |
| 8 | Full-Stack Architecture: FastAPI REST microservice, landing page, dual model inference | Completed |

---

## 🧬 Machine Learning & Deep Learning Models

### 1. XGBoost Regressor (Production Baseline)
- **Algorithm**: Gradient Boosted Decision Trees
- **Hyperparameters**: `n_estimators=400`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`
- **Inference Latency**: < 15 ms on CPU
- **Best For**: Fast single-snapshot telemetry scoring and edge microservices.

### 2. PyTorch 2-Layer LSTM Network (Deep Sequence Recurrent Model)
- **Architecture**:
  - Input: `(batch_size, 30, 13)` (30 flight cycles across 13 selected sensor channels)
  - LSTM Layer 1: 64 hidden units, dropout = 0.2
  - LSTM Layer 2: 32 hidden units
  - Dense Layers: 32 -> 16 (ReLU) -> 1 (Predicted RUL)
- **Training**: 25 epochs, Adam optimizer, MSE loss, learning rate = 0.001
- **Best For**: Multi-cycle degradation trajectory analysis capturing non-linear dynamic degradation.

---

## 📊 Benchmark Evaluation (Unseen Test Data)

Evaluation performed on 100 unseen test turbofan engines from NASA C-MAPSS FD001:

| Model Architecture | Test RMSE (Cycles) | Test MAE (Cycles) | Test R² | Model Type |
|---|---|---|---|---|
| Linear Regression | 22.19 | 17.37 | 0.6935 | Baseline Classical |
| Random Forest | 19.76 | 14.95 | 0.7568 | Ensemble Bagging |
| **XGBoost Regressor** | **18.91** | **14.35** | **0.7773** | Gradient Boosted Trees |
| **LSTM Network (PyTorch)** | **13.23** | **9.45** | **0.8910** | Sequential Recurrent DL |

> **Key Takeaway**: The PyTorch LSTM network achieves the lowest Test RMSE (**13.23 cycles**) and lowest Test MAE (**9.45 cycles**), demonstrating the power of temporal sequence modeling for progressive mechanical degradation.

---

## 🛠️ How to Run

### Option 1: One-Click Runner (Recommended)
Open VS Code, navigate to the project directory, and run:
```bash
python RUN_ALL.py
```
This automatically launches:
- **FastAPI Backend**: `http://localhost:8000`
- **Streamlit Web Dashboard**: `http://localhost:8501`
- **Aerospace Landing Page**: `http://localhost:8000/landing`

### Option 2: Split Terminals
```bash
# Terminal 1: Start Backend API
python START_BACKEND.py

# Terminal 2: Start Streamlit Frontend
python START_FRONTEND.py
```

### Option 3: Run Individual Models (1-Click Standalone)
```bash
# Interactive Model Selection Menu
python RUN_INDIVIDUAL_MODEL.py

# Or run specific models directly:
python RUN_LINEAR_REGRESSION.py
python RUN_RANDOM_FOREST.py
python RUN_GRADIENT_BOOSTING.py
python RUN_XGBOOST.py
python RUN_LSTM.py
```

### Option 4: Train & Benchmark All Models
```bash
# Train & evaluate Linear Regression, Random Forest, Gradient Boosting
python TRAIN_ADDITIONAL_MODELS.py

# Train XGBoost regressor
python TRAIN_MODEL.py

# Train PyTorch LSTM model
python TRAIN_LSTM.py

# Run comparative test evaluation across all models on 100 test engines
python TEST_MODEL.py
```

### Option 5: Run Verification Test Suites
```bash
# Complete 58-test suite for individual model demonstrations
python VERIFY_ALL_DEMOS.py

# Comprehensive end-to-end model verification
python thorough_project_test.py

# Verify API endpoints & batch prediction logic
python test_api_client.py

# Verify Streamlit frontend components headlessly
python test_frontend.py

# Verify API vs standalone inference consistency
python verify_inference_match.py

# Audit all 4 NASA C-MAPSS subsets (FD001–FD004)
python verify_all_subsets.py
```

---

## 📡 FastAPI REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service root and liveness check |
| `GET` | `/health` | Model status check (reports status of LR, RF, GB, XGBoost & LSTM) |
| `GET` | `/features` | Returns list of 13 required sensor telemetry features |
| `GET` | `/model/info` | Returns full metadata, benchmarks, and operational thresholds across all 5 models |
| `GET` | `/landing` | Serves the standalone aerospace HTML landing page |
| `POST` | `/predict?model=<model>` | Single-engine RUL prediction (`linear_regression`, `random_forest`, `gradient_boosting`, `xgboost`, `lstm`) |
| `POST` | `/predict/batch?model=<model>` | Bulk fleet prediction with fleet-wide summary statistics |

### Example: Single Engine Prediction (`POST /predict?model=lstm`)
```bash
curl -X POST "http://localhost:8000/predict?model=lstm" \
     -H "Content-Type: application/json" \
     -d '{
       "Cycle": 1,
       "SensorMeasure2": 641.82,
       "SensorMeasure3": 1589.70,
       "SensorMeasure4": 1400.60,
       "SensorMeasure7": 554.36,
       "SensorMeasure8": 2388.02,
       "SensorMeasure11": 47.47,
       "SensorMeasure12": 521.66,
       "SensorMeasure13": 2388.02,
       "SensorMeasure15": 8.4195,
       "SensorMeasure17": 392.0,
       "SensorMeasure20": 39.06,
       "SensorMeasure21": 23.419
     }'
```

---

## 📚 References & Data Source

- **Dataset**: NASA Prognostics Center of Excellence (PCoE) Turbofan Engine Degradation Simulation Dataset (C-MAPSS)
- **Publication**: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. International Conference on Prognostics and Health Management.
- **Deep Learning Framework**: PyTorch (`torch`)
- **Web Stack**: FastAPI, Uvicorn, Streamlit, Plotly
