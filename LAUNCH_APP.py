"""
LAUNCH_APP.py
=============
Launches the MaintAI Streamlit Dashboard.
Run this AFTER running TRAIN_MODEL.py

Usage: python LAUNCH_APP.py
"""
import subprocess, os, sys, webbrowser, time

# Ensure UTF-8 output on Windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE    = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE, 'frontend', 'app')
MDL_DIR = os.path.join(BASE, 'models')

print("\n" + "="*60)
print("  MAINTAI — APP LAUNCHER")
print("="*60)

# Check if model exists
if not os.path.exists(os.path.join(MDL_DIR, 'final_model.pkl')):
    print("\n⚠️  Model not found! Running TRAIN_MODEL.py first...")
    result = subprocess.run([sys.executable, os.path.join(BASE, 'TRAIN_MODEL.py')])
    if result.returncode != 0:
        print("❌  Training failed. Please check errors above.")
        input("Press Enter to exit...")
        sys.exit(1)

print("\n✅  Model found! Launching Streamlit app...")
print("   Opening browser at: http://localhost:8501")
print("   Press Ctrl+C to stop the app.\n")

time.sleep(1)
subprocess.run(
    [sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'],
    cwd=APP_DIR
)
