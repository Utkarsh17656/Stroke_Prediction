@echo off
cd /d "%~dp0"
echo 🏗️ Running AI 2.0 Training Pipeline...
echo ⏳ This may take a minute depending on your hardware...
call venv\Scripts\activate.bat
python app.py
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Training Pipeline Finished Successfully!
    echo 🎯 Models have been updated.
) else (
    echo.
    echo ❌ Training Pipeline Failed. Please check the errors above.
)
pause
