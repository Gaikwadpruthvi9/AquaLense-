@echo off
title AquaLens Launcher
echo Launching AquaLens Backend and Frontend...
start "AquaLens Backend" cmd /c "%~dp0start-backend.bat"
timeout /t 2 /nobreak >nul
start "AquaLens Frontend" cmd /c "%~dp0start-frontend.bat"
echo AquaLens servers have been started in separate windows.
