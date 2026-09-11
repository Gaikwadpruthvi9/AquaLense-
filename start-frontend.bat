@echo off
title AquaLens Frontend (Vite + React)
echo ====================================================
echo  Starting AquaLens / OceanScan AI - Frontend (Vite)
echo ====================================================
cd /d "%~dp0project\frontend"

:: Redirect TEMP and NPM cache to F: drive to avoid C: drive out-of-space issues
set TEMP=%~dp0tmp
set TMP=%~dp0tmp
set npm_config_cache=%~dp0.npm-cache
if not exist "%TEMP%" mkdir "%TEMP%"

if not exist node_modules (
    echo Installing npm packages (this may take a minute on first run)...
    call npm install
)

echo Starting Vite dev server at http://localhost:5173 ...
call npm run dev
pause
