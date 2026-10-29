"""
TRAIN_MODEL.py
==============
This script trains the XGBoost model and saves it to models/final_model.pkl
Run this ONCE before launching the Streamlit app.

Usage: python TRAIN_MODEL.py
"""

import sys, io
# Fix Windows encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import os, json, joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import xgboost as xgb

# Paths
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'Dataset')
MDL_DIR  = os.path.join(BASE, 'models')
os.makedirs(MDL_DIR, exist_ok=True)

SELECTED_SENSORS = [
    'Cycle',
    'SensorMeasure2', 'SensorMeasure3', 'SensorMeasure4',
    'SensorMeasure7', 'SensorMeasure8', 'SensorMeasure11',
    'SensorMeasure12','SensorMeasure13','SensorMeasure15',
    'SensorMeasure17','SensorMeasure20','SensorMeasure21'
]

print("\n" + "="*60)
print("  MAINTAI -- MODEL TRAINER")
print("  Training XGBoost RUL Predictor on NASA CMAPSS")
print("="*60)

# Step 1: Load data
print("\n[1/5] Loading NASA CMAPSS dataset (FD001-FD004)...")

def load_raw(n):
    path = os.path.join(DATA_DIR, 'train_FD00' + str(n) + '.txt')
    cols = ['ID', 'Cycle', 'OpSet1', 'OpSet2', 'OpSet3'] + \
           ['SensorMeasure' + str(i) for i in range(1, 22)]
    df = pd.read_csv(path, sep=' ', header=None)
    df.drop(df.columns[[26, 27]], axis=1, inplace=True)
    df.columns = cols
    max_c = df.groupby('ID')['Cycle'].max().reset_index()
    max_c.columns = ['ID', 'MaxCycle']
    df = df.merge(max_c, on='ID')
    df['RUL'] = (df['MaxCycle'] - df['Cycle']).clip(upper=125)
    df.drop('MaxCycle', axis=1, inplace=True)
    return df

dfs = []
for i in range(1, 5):
    try:
        dfs.append(load_raw(i))
        print("   [OK] Loaded FD00" + str(i))
    except Exception as e:
        print("   [WARN] Could not load FD00" + str(i) + ": " + str(e))

df = pd.concat(dfs, ignore_index=True)
print("   Total rows: " + str(len(df)))

# Step 2: Prepare features
print("\n[2/5] Preparing features and train/test split...")
X = df[SELECTED_SENSORS]
y = df['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("   Train: " + str(len(X_train)) + " rows  |  Test: " + str(len(X_test)) + " rows")

# Step 3: Train XGBoost
print("\n[3/5] Training XGBoost Regressor (this takes 1-3 minutes)...")
model = xgb.XGBRegressor(
    n_estimators    = 400,
    max_depth       = 6,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    random_state    = 42,
    n_jobs          = -1,
    verbosity       = 0
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("   [OK] Training complete!")

# Step 4: Evaluate
print("\n[4/5] Evaluating model performance...")
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
test_rmse  = np.sqrt(metrics.mean_squared_error(y_test,  y_pred_test))
train_mae  = metrics.mean_absolute_error(y_train, y_pred_train)
test_mae   = metrics.mean_absolute_error(y_test,  y_pred_test)
train_r2   = metrics.r2_score(y_train, y_pred_train)
test_r2    = metrics.r2_score(y_test,  y_pred_test)

print("\n   RESULTS:")
print("   +------------------+----------+----------+")
print("   | Metric           |  Train   |   Test   |")
print("   +------------------+----------+----------+")
print("   | RMSE             | " + "{:8.2f}".format(train_rmse) + " | " + "{:8.2f}".format(test_rmse) + " |")
print("   | MAE              | " + "{:8.2f}".format(train_mae)  + " | " + "{:8.2f}".format(test_mae)  + " |")
print("   | R2               | " + "{:8.4f}".format(train_r2)   + " | " + "{:8.4f}".format(test_r2)   + " |")
print("   +------------------+----------+----------+")

# Step 5: Save model
print("\n[5/5] Saving model files...")

joblib.dump(model,  os.path.join(MDL_DIR, 'final_model.pkl'))
joblib.dump(scaler, os.path.join(MDL_DIR, 'scaler.pkl'))

with open(os.path.join(MDL_DIR, 'feature_list.json'), 'w') as f:
    json.dump(SELECTED_SENSORS, f)

comp = pd.DataFrame([{
    'Model'    : 'XGBoost (Final)',
    'Test RMSE': round(test_rmse, 2),
    'Test MAE' : round(test_mae, 2),
    'Test R2'  : round(test_r2, 4),
}])
comp.to_csv(os.path.join(MDL_DIR, 'model_comparison.csv'), index=False)

print("   [SAVED] models/final_model.pkl")
print("   [SAVED] models/scaler.pkl")
print("   [SAVED] models/feature_list.json")

print("\n" + "="*60)
print("  MODEL TRAINING COMPLETE!")
print("  Now run:  python LAUNCH_APP.py")
print("="*60 + "\n")
