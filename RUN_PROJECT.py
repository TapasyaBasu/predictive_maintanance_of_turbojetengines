"""
HOW TO RUN THE PROJECT — COMPLETE BEGINNER GUIDE
=================================================
This script runs everything for you automatically.
Just double-click this file OR run it from terminal.
"""

import subprocess
import sys
import os
import time

def print_step(n, msg):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {msg}")
    print(f"{'='*60}")

def run(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-2000:])   # last 2000 chars
    if result.returncode != 0 and result.stderr:
        print("Error:", result.stderr[-1000:])
    return result.returncode

BASE = os.path.dirname(os.path.abspath(__file__))
NB   = os.path.join(BASE, 'notebooks')
APP  = os.path.join(BASE, 'frontend', 'app')

print("\n" + "="*60)
print("  MAINTAI — PROJECT RUNNER")
print("  Predictive Maintenance for Turbofan Engines")
print("="*60)

# ── STEP 1: Install packages ─────────────────────────────────────────────────
print_step(1, "Installing required Python packages...")
packages = "pandas numpy matplotlib seaborn scikit-learn xgboost streamlit plotly joblib notebook ipykernel"
code = run(f'pip install {packages}')
if code == 0:
    print("✅ All packages installed successfully!")
else:
    print("⚠️  Some packages may have failed. Continuing anyway...")

# ── STEP 2: Run the Model Training Notebook ──────────────────────────────────
print_step(2, "Running the Model Training & Evaluation Notebook...")
print("  This will train all models and save 'final_model.pkl'")
print("  (This may take 5–15 minutes depending on your computer)\n")

nb_path = os.path.join(NB, 'Week6_Model_Evaluation_Comparison.ipynb')
out_path = os.path.join(NB, 'Week6_Model_Evaluation_Comparison_OUTPUT.ipynb')

code = run(
    f'jupyter nbconvert --to notebook --execute "{nb_path}" --output "{out_path}" --ExecutePreprocessor.timeout=600',
    cwd=NB
)

if code == 0:
    print("✅ Model trained and saved to 'models/final_model.pkl'!")
else:
    print("⚠️  Notebook execution had issues. The Streamlit app will still run in demo mode.")

# ── STEP 3: Launch Streamlit App ─────────────────────────────────────────────
print_step(3, "Launching the Streamlit Web Dashboard...")
print("  Opening your browser at: http://localhost:8501")
print("  Press Ctrl+C in this window to stop the app.\n")
time.sleep(2)

run(f'streamlit run "{os.path.join(APP, "streamlit_app.py")}"', cwd=APP)
