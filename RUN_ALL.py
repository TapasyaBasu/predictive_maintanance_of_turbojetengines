"""Launch full MaintAI stack: Backend API + Frontend Dashboard."""
import subprocess
import os
import sys
import time
import webbrowser

# Ensure UTF-8 output on Windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))

print("\n" + "=" * 60)
print("  MaintAI — Full Stack Launcher")
print("  Backend API  : http://localhost:8000")
print("  API Docs     : http://localhost:8000/docs")
print("  Dashboard    : http://localhost:8501")
print("  Landing Page : http://localhost:8000/landing")
print("=" * 60 + "\n")

backend_proc = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'api:app',
     '--host', '127.0.0.1', '--port', '8000'],
    cwd=os.path.join(BASE, 'backend')
)
print("[OK] Backend started (PID:", backend_proc.pid, ")")
time.sleep(2)

frontend_proc = subprocess.Popen(
    [sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
     '--server.headless', 'true', '--server.port', '8501'],
    cwd=os.path.join(BASE, 'frontend', 'app')
)
print("[OK] Frontend started (PID:", frontend_proc.pid, ")")
time.sleep(2)

# Open landing page and dashboard in browser
try:
    webbrowser.open('http://localhost:8000/landing')
    time.sleep(1)
    webbrowser.open('http://localhost:8501')
except Exception:
    pass

print("\n  Both services are running!")
print("  Press Ctrl+C in this terminal to stop both servers.\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n  Stopping servers...")
    backend_proc.terminate()
    frontend_proc.terminate()
    print("  Servers stopped successfully.")

