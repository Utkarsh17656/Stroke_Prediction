@echo off
setlocal
cd /d "%~dp0"
echo 🧬 Starting StrokeRisk AI 2.0 Web Server...

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found. Please ensure 'venv' folder exists.
    pause
    exit /b
)

:: Activate venv and start server in a separate process
echo 🚀 Launching server on http://127.0.0.1:8000 ...
start "" cmd /k "call venv\Scripts\activate.bat && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

:: Wait a few seconds for server to start, then open browser
timeout /t 3 /nobreak > nul
start http://127.0.0.1:8000

echo ✅ Browser should open shortly.
echo ⚠️ Keep the other terminal window open while using the app.
pause
