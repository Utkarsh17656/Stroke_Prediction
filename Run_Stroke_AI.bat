@echo off
setlocal
cd /d "%~dp0"
echo 🧬 Starting StrokeRisk AI 2.0 Web Server...

set "PORT=8091"

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found. Please ensure 'venv' folder exists.
    pause
    exit /b
)

:: Activate venv and start server in a separate process
call :check_port %PORT%
if %errorlevel%==0 (
    echo ⚠️ Port %PORT% is already in use by another app.
    set "PORT=8092"
    call :check_port %PORT%
    if %errorlevel%==0 (
        echo ❌ Ports 8091 and 8092 are both in use.
        echo Please close the app using one of these ports, then run again.
        pause
        exit /b
    )
)

echo 🚀 Launching server on http://127.0.0.1:%PORT% ...
start "StrokeRisk AI Server" cmd /k "cd /d ""%~dp0"" && call venv\Scripts\activate.bat && python -m uvicorn main:app --app-dir . --host 127.0.0.1 --port %PORT%"

:: Wait a few seconds for server to start, then open browser
timeout /t 3 /nobreak > nul
start "" "http://127.0.0.1:%PORT%/?app=stroke_ai"

echo ✅ Browser should open shortly.
echo ⚠️ Keep the other terminal window open while using the app.
pause
exit /b

:check_port
set "CHECK_PORT=%~1"
netstat -ano | findstr /R /C:":%CHECK_PORT% .*LISTENING" >nul
if %errorlevel%==0 (
    exit /b 0
)
exit /b 1
