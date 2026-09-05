@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
echo ========================================
echo  Shot-by-Shot Video Processor
echo ========================================
echo.
echo Starting web server...
echo Open http://localhost:5000 in your browser
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.
"%PYTHON%" app.py
pause
