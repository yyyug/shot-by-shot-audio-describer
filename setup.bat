@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%setup.ps1"
if errorlevel 1 (
    echo.
    echo Setup finished with errors.
    pause
    exit /b 1
)
pause
