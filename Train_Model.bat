@echo off
cd /d "%~dp0"
echo 🏗️ Running AI 3.0 Calibrated Training Pipeline...
echo ⏳ This may take a minute depending on your hardware...
call venv\Scripts\activate.bat
python app.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Training Pipeline Finished Successfully!
    echo 🎯 Model bundle has been updated.
) else (
    echo.
    echo ❌ Training Pipeline Failed. Please check the errors above.
)
pause
