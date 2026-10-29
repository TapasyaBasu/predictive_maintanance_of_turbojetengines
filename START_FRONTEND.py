"""Launch the Streamlit frontend dashboard."""
import subprocess
import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))
APP  = os.path.join(BASE, 'frontend', 'app')

print("\n" + "=" * 50)
print("  Starting MaintAI Frontend Dashboard...")
print("  Dashboard: http://localhost:8501")
print("=" * 50 + "\n")

subprocess.run(
    [sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'],
    cwd=APP
)

