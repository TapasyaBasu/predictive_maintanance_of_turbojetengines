"""
TRAIN_LSTM.py
==============
Week 5 & Week 6: Sequential LSTM Neural Network for Turbofan Engine RUL Prediction
NASA C-MAPSS Dataset (FD001)

Tasks Fulfilled:
- Week 4 (Member 3): Sliding time-window sequence generation grouped by engine ID
- Week 5 (Member 4): LSTM model architecture, training loop, loss monitoring
- Week 6 (Member 3): Evaluation metrics (RMSE, MAE, R^2), actual vs predicted RUL curves
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'Dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Configuration & Hyperparameters ─────────────────────────────────────────
SEQUENCE_LENGTH = 30      # Sliding time-window (cycles)
BATCH_SIZE      = 64
EPOCHS          = 25
LEARNING_RATE   = 0.001
MAX_RUL_CAP     = 125.0    # Piecewise linear RUL ceiling

SELECTED_SENSORS = [
    'Cycle',
    'SensorMeasure2', 'SensorMeasure3', 'SensorMeasure4',
    'SensorMeasure7', 'SensorMeasure8', 'SensorMeasure11',
    'SensorMeasure12', 'SensorMeasure13', 'SensorMeasure15',
    'SensorMeasure17', 'SensorMeasure20', 'SensorMeasure21'
]

print("=" * 65)
print("  MAINTAI — LSTM DEEP LEARNING MODEL FOR RUL PREDICTION")
print("  Dataset: NASA C-MAPSS (FD001) | Architecture: 2-Layer LSTM")
print("=" * 65)

# ── 2. Data Loading & Preprocessing ───────────────────────────────────────────
print("\n[1/5] Loading NASA C-MAPSS FD001 dataset...")

def load_fd001_train():
    path = os.path.join(DATA_DIR, 'train_FD001.txt')
    cols = ['ID', 'Cycle', 'OpSet1', 'OpSet2', 'OpSet3'] + [f'SensorMeasure{i}' for i in range(1, 22)]
    df = pd.read_csv(path, sep=r'\s+', header=None)
    if df.shape[1] > 26:
        df = df.iloc[:, :26]
    df.columns = cols
    
    # Calculate RUL per engine
    max_cycles = df.groupby('ID')['Cycle'].max().reset_index()
    max_cycles.columns = ['ID', 'MaxCycle']
    df = df.merge(max_cycles, on='ID')
    df['RUL'] = (df['MaxCycle'] - df['Cycle']).clip(upper=MAX_RUL_CAP)
    df.drop('MaxCycle', axis=1, inplace=True)
    return df

def load_fd001_test():
    path_test = os.path.join(DATA_DIR, 'test_FD001.txt')
    path_rul  = os.path.join(DATA_DIR, 'RUL_FD001.txt')
    
    cols = ['ID', 'Cycle', 'OpSet1', 'OpSet2', 'OpSet3'] + [f'SensorMeasure{i}' for i in range(1, 22)]
    test_df = pd.read_csv(path_test, sep=r'\s+', header=None)
    if test_df.shape[1] > 26:
        test_df = test_df.iloc[:, :26]
    test_df.columns = cols
    
    rul_truth = pd.read_csv(path_rul, sep=r'\s+', header=None, names=['RUL_truth'])
    rul_truth['ID'] = np.arange(1, len(rul_truth) + 1)
    
    # For each test engine, the true RUL at the last cycle is RUL_truth
    max_test_cycles = test_df.groupby('ID')['Cycle'].max().reset_index()
    max_test_cycles.columns = ['ID', 'MaxCycle']
    test_df = test_df.merge(max_test_cycles, on='ID')
    test_df = test_df.merge(rul_truth, on='ID')
    test_df['RUL'] = (test_df['MaxCycle'] + test_df['RUL_truth'] - test_df['Cycle']).clip(upper=MAX_RUL_CAP)
    test_df.drop(['MaxCycle', 'RUL_truth'], axis=1, inplace=True)
    return test_df, rul_truth

train_df = load_fd001_train()
test_df, true_ruls = load_fd001_test()

print(f"  Training samples: {len(train_df):,} rows from {train_df['ID'].nunique()} engines")
print(f"  Test samples:     {len(test_df):,} rows from {test_df['ID'].nunique()} engines")

# Standardize features
scaler = StandardScaler()
train_df[SELECTED_SENSORS] = scaler.fit_transform(train_df[SELECTED_SENSORS])
test_df[SELECTED_SENSORS]  = scaler.transform(test_df[SELECTED_SENSORS])
joblib.dump(scaler, os.path.join(MODELS_DIR, 'lstm_scaler.pkl'))
print(f"  [SAVED] {os.path.join(MODELS_DIR, 'lstm_scaler.pkl')}")

# ── 3. Sequence Generation (Sliding Window per Engine) ────────────────────────
print("\n[2/5] Generating temporal sequences (Window Length = 30 cycles)...")

def gen_train_sequences(df, seq_len, feature_cols):
    """Generate (seq_len, num_features) sliding windows strictly within each engine."""
    sequences = []
    labels    = []
    for engine_id in df['ID'].unique():
        engine_data = df[df['ID'] == engine_id]
        if len(engine_data) < seq_len:
            continue
        feat_matrix = engine_data[feature_cols].values
        rul_array   = engine_data['RUL'].values
        
        num_windows = len(engine_data) - seq_len + 1
        for i in range(num_windows):
            sequences.append(feat_matrix[i:i + seq_len])
            labels.append(rul_array[i + seq_len - 1])
            
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

def gen_test_sequences(df, seq_len, feature_cols):
    """Extract the last sequence for each engine to predict the final RUL."""
    sequences = []
    labels    = []
    for engine_id in df['ID'].unique():
        engine_data = df[df['ID'] == engine_id]
        feat_matrix = engine_data[feature_cols].values
        rul_array   = engine_data['RUL'].values
        
        if len(engine_data) >= seq_len:
            sequences.append(feat_matrix[-seq_len:])
            labels.append(rul_array[-1])
        else:
            # Pad short sequences with first row
            pad_len = seq_len - len(engine_data)
            padding = np.repeat(feat_matrix[0:1], pad_len, axis=0)
            padded  = np.vstack([padding, feat_matrix])
            sequences.append(padded)
            labels.append(rul_array[-1])
            
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

X_train, y_train = gen_train_sequences(train_df, SEQUENCE_LENGTH, SELECTED_SENSORS)
X_test,  y_test  = gen_test_sequences(test_df, SEQUENCE_LENGTH, SELECTED_SENSORS)

print(f"  Training sequences: {X_train.shape}  |  Labels: {y_train.shape}")
print(f"  Test sequences:     {X_test.shape}  |  Labels: {y_test.shape}")

# Train/Validation Split (80% train engines, 20% validation)
split_idx = int(len(X_train) * 0.8)
train_dataset = TensorDataset(torch.tensor(X_train[:split_idx]), torch.tensor(y_train[:split_idx]))
val_dataset   = TensorDataset(torch.tensor(X_train[split_idx:]), torch.tensor(y_train[split_idx:]))

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ── 4. Define LSTM Neural Network Architecture ───────────────────────────────
print("\n[3/5] Initializing 2-Layer LSTM Model Architecture...")

class TurbofanLSTM(nn.Module):
    """
    2-Layer Recurrent Neural Network for RUL Prediction:
    Input: (batch_size, sequence_length, num_features)
    LSTM Layer 1: 64 hidden units, dropout 0.2
    LSTM Layer 2: 32 hidden units
    Dense Layers: 32 -> 16 -> 1 (Predicted RUL)
    """
    def __init__(self, input_dim=13, hidden_dim1=64, hidden_dim2=32, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Take representation at the last time step
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = TurbofanLSTM(input_dim=len(SELECTED_SENSORS)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

print(f"  Execution Device: {device}")
print(model)

# ── 5. Train the LSTM Model (Week 5 Member 4 Deliverable) ─────────────────────
print("\n[4/5] Training LSTM Model across 25 epochs...")
train_losses = []
val_losses   = []

t0 = time.time()
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss  = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * len(batch_y)
    
    epoch_train_loss = np.sqrt(running_train_loss / len(train_dataset))
    
    # Validation evaluation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(device), val_y.to(device)
            val_preds = model(val_x)
            v_loss = criterion(val_preds, val_y)
            running_val_loss += v_loss.item() * len(val_y)
    
    epoch_val_loss = np.sqrt(running_val_loss / len(val_dataset))
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'lstm_best_model.pth'))
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:2d}/{EPOCHS}] - Train RMSE: {epoch_train_loss:6.2f} | Val RMSE: {epoch_val_loss:6.2f}")

print(f"  Training finished in {time.time() - t0:.1f}s. Best Val RMSE: {best_val_loss:.2f}")

# Load best checkpoint
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'lstm_best_model.pth')))

# ── 6. Evaluation & Plots (Week 6 Member 3 Deliverable) ───────────────────────
print("\n[5/5] Evaluating on Test Dataset (Actual vs Predicted RUL)...")
model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(X_test).to(device)
    raw_preds   = model(test_tensor).cpu().numpy()
    test_preds  = np.clip(raw_preds, 0.0, MAX_RUL_CAP)

test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
test_mae  = mean_absolute_error(y_test, test_preds)
test_r2   = r2_score(y_test, test_preds)

print("\n" + "=" * 50)
print("  LSTM MODEL TEST PERFORMANCE:")
print(f"  Test RMSE : {test_rmse:.2f} cycles")
print(f"  Test MAE  : {test_mae:.2f} cycles")
print(f"  Test R^2  : {test_r2:.4f}")
print("=" * 50)

# ── Plot 1: Training & Validation Loss Curve
plt.figure(figsize=(9, 4.5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss (RMSE)', color='#0ea5e9', lw=2)
plt.plot(range(1, EPOCHS + 1), val_losses,   label='Validation Loss (RMSE)', color='#f59e0b', lw=2, linestyle='--')
plt.title('Week 5: LSTM Training & Validation Loss Curves (FD001)', fontsize=12, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('RMSE (Cycles)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
loss_curve_path = os.path.join(MODELS_DIR, 'lstm_loss_curve.png')
plt.savefig(loss_curve_path, dpi=200)
plt.close()
print(f"  Saved plot: {loss_curve_path}")

# ── Plot 2: Actual vs Predicted RUL for Test Engines
plt.figure(figsize=(11, 5))
engine_indices = np.arange(1, len(y_test) + 1)
plt.plot(engine_indices, y_test, label='Ground Truth RUL', color='#10b981', lw=2)
plt.plot(engine_indices, test_preds, label='LSTM Predicted RUL', color='#ef4444', lw=1.8, linestyle='--')
plt.title(f'Week 6: LSTM Actual vs Predicted RUL on Test Engines (RMSE: {test_rmse:.2f}, R²: {test_r2:.2f})', fontsize=12, fontweight='bold')
plt.xlabel('Engine ID')
plt.ylabel('Remaining Useful Life (Cycles)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
pred_curve_path = os.path.join(MODELS_DIR, 'lstm_test_prediction.png')
plt.savefig(pred_curve_path, dpi=200)
plt.close()
print(f"  Saved plot: {pred_curve_path}")

# ── Update Model Comparison Table
comp_file = os.path.join(MODELS_DIR, 'model_comparison.csv')
comp_row = {
    'Model': 'LSTM Network (PyTorch)',
    'Test RMSE': round(test_rmse, 2),
    'Test MAE': round(test_mae, 2),
    'Test R2': round(test_r2, 4)
}

if os.path.exists(comp_file):
    comp_df = pd.read_csv(comp_file)
    # Remove existing LSTM row if present, then append
    comp_df = comp_df[comp_df['Model'] != 'LSTM Network (PyTorch)']
    comp_df = pd.concat([comp_df, pd.DataFrame([comp_row])], ignore_index=True)
else:
    comp_df = pd.DataFrame([comp_row])

comp_df.to_csv(comp_file, index=False)
print(f"  Updated model comparison file: {comp_file}")
print("\n[OK] LSTM training and evaluation completed successfully!\n")

