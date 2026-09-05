@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
echo ========================================
echo  Shot-by-Shot Video Processor
echo ========================================
echo.
echo Starting desktop application...
echo.
echo Press Alt+F4 to close the application
echo ========================================
echo.
"%PYTHON%" desktop.py
pause
