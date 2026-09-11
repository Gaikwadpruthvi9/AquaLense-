@echo off
title AquaLens Backend (FastAPI)
echo ====================================================
echo  Starting AquaLens / OceanScan AI - Backend Server
echo ====================================================
cd /d "%~dp0project\backend"

:: Redirect TEMP and PIP cache to F: drive to avoid C: drive out-of-space issues
set TEMP=%~dp0tmp
set TMP=%~dp0tmp
set PIP_CACHE_DIR=%~dp0.pip-cache
if not exist "%TEMP%" mkdir "%TEMP%"

if not exist venv (
    echo Creating Python virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Checking/Installing Python dependencies...
pip install --cache-dir "%PIP_CACHE_DIR%" -r requirements.txt

echo Starting FastAPI server at http://127.0.0.1:8000 ...
python main.py
pause
