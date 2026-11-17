"""
MaintAI — Single-Click Project Runner
=====================================
Launches both services:
  - Backend API: http://127.0.0.1:8000
  - Frontend Dashboard: http://localhost:8501
"""
import os
import sys
import subprocess

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    run_all_script = os.path.join(BASE, "RUN_ALL.py")
    subprocess.run([sys.executable, run_all_script])

