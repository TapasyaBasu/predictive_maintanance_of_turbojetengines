"""Launch the FastAPI backend server."""
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

BASE    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(BASE, 'backend')

print("\n" + "=" * 50)
print("  Starting MaintAI Backend API...")
print("  API Docs     : http://127.0.0.1:8000/docs")
print("  Landing Page : http://127.0.0.1:8000/landing")
print("=" * 50 + "\n")

subprocess.run(
    [sys.executable, '-m', 'uvicorn', 'api:app',
     '--host', '127.0.0.1', '--port', '8000', '--reload'],
    cwd=BACKEND
)

