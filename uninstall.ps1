# Shot-by-Shot uninstaller
# Removes: Start Menu shortcuts, isolated .venv, pre-downloaded models,
# and (optionally) the entire application folder.
#
# Run via uninstall.bat (double-click) or: powershell -ExecutionPolicy Bypass -File uninstall.ps1

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$ModelsDir = Join-Path $Root "models\sensevoice"

function Write-Step([string]$Msg) { Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Write-Ok([string]$Msg) { Write-Host "    $Msg" -ForegroundColor Green }

Write-Step "Shot-by-Shot uninstaller"
Write-Host "App directory: $Root"

# ---------------------------------------------------------------- shortcuts
Write-Step "Removing Start Menu shortcuts..."
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
foreach ($name in @("Shot-by-Shot (Web).lnk", "Shot-by-Shot (Desktop).lnk", "Shot-by-Shot (Uninstall).lnk")) {
    $lnk = Join-Path $StartMenu $name
    if (Test-Path $lnk) { Remove-Item $lnk -Force; Write-Ok "Removed: $name" }
}

# ---------------------------------------------------------------- venv
Write-Step "Removing isolated virtual environment (.venv)..."
if (Test-Path $VenvDir) {
    Remove-Item $VenvDir -Recurse -Force
    Write-Ok "Removed: $VenvDir"
} else {
    Write-Ok ".venv not found."
}

# ---------------------------------------------------------------- models
Write-Step "Removing pre-downloaded SenseVoice models folder..."
if (Test-Path $ModelsDir) {
    $ans = Read-Host "Remove SenseVoice models folder too (saves ~1GB, needs re-download to use again)? [y/N]"
    if ($ans -match "^(y|Y|yes)$") {
        Remove-Item $ModelsDir -Recurse -Force
        Write-Ok "Removed: $ModelsDir"
    } else {
        Write-Ok "Kept: $ModelsDir"
    }
}

# ---------------------------------------------------------------- whole folder
Write-Step "Cleanup options"
$ans = Read-Host "Delete the ENTIRE application folder ($Root) and all your data? [y/N]"
if ($ans -match "^(y|Y|yes)$") {
    Remove-Item $Root -Recurse -Force
    Write-Ok "Removed entire application folder."
    exit 0
}
Write-Ok "Kept application folder."

Write-Step "Uninstall complete."
Write-Host "The system Python environment was never modified (everything was isolated in .venv)."
