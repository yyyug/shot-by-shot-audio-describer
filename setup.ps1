# Shot-by-Shot installer
# Creates an isolated Python virtual environment (.venv) so system Python is not
# polluted, installs dependencies, pre-downloads SenseVoice models, and creates
# Start Menu shortcuts.
#
# Run via setup.bat (double-click) or: powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$ModelsDir = Join-Path $Root "models\sensevoice"

function Write-Step([string]$Msg) { Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Write-Ok([string]$Msg) { Write-Host "    $Msg" -ForegroundColor Green }
function Write-Warn([string]$Msg) { Write-Host "    $Msg" -ForegroundColor Yellow }

Write-Step "Shot-by-Shot installer"
Write-Host "App directory: $Root"

# ---------------------------------------------------------------- locate python
$pyParts = $null
foreach ($candidate in @("py -3", "python")) {
    try {
        $parts = @($candidate -split " ")
        $rest = if ($parts.Count -gt 1) { $parts[1..($parts.Count - 1)] } else { @() }
        & $parts[0] @($rest + @("--version")) *> $null
        if ($LASTEXITCODE -eq 0) { $pyParts = $parts; break }
    } catch { }
}
if (-not $pyParts) {
    Write-Warn "Python 3 not found."
    Write-Warn "Please install Python 3.8+ from https://www.python.org/downloads/ (check 'Add to PATH')."
    exit 1
}
Write-Ok "Using Python: $($pyParts -join ' ')"

# ---------------------------------------------------------------- create venv
if (Test-Path (Join-Path $VenvDir "Scripts\python.exe")) {
    Write-Ok "Virtual environment already exists: $VenvDir"
} else {
    Write-Step "Creating isolated virtual environment (.venv)..."
    $rest = if ($pyParts.Count -gt 1) { $pyParts[1..($pyParts.Count - 1)] } else { @() }
    & $pyParts[0] @($rest + @("-m", "venv", $VenvDir))
    if ($LASTEXITCODE -ne 0) { Write-Warn "Failed to create venv"; exit 1 }
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

# ---------------------------------------------------------------- install deps
Write-Step "Installing Python dependencies (isolated, does not touch system Python)..."
& $VenvPython -m pip install --upgrade pip
& $VenvPip install -r (Join-Path $Root "requirements-cpu.txt")
& $VenvPip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Warn "pip install failed. Check your network and try again."; exit 1 }
Write-Ok "Dependencies installed."

Write-Step "Installing optional desktop dependencies (pywebview)..."
& $VenvPip install -r (Join-Path $Root "requirements-desktop.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pywebview install failed (desktop app may not open). The web version is unaffected."
}

# ---------------------------------------------------------------- models
$Manifest = Join-Path $ModelsDir "model_dirs.json"
if (Test-Path $Manifest) {
    Write-Step "SenseVoice models found (bundled with the package) - no download needed."
    Write-Ok "Models: $ModelsDir"
} else {
    Write-Step "Pre-downloading SenseVoice models (~1GB, one-time)..."
    & $VenvPython (Join-Path $Root "scripts\download_sensevoice.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Model download failed. The app will still work but will attempt to download models on first transcription."
    }
}

# ---------------------------------------------------------------- shortcuts
Write-Step "Creating Start Menu shortcuts..."
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$Wsh = New-Object -ComObject WScript.Shell
$webLnk = Join-Path $StartMenu "Shot-by-Shot (Web).lnk"
$deskLnk = Join-Path $StartMenu "Shot-by-Shot (Desktop).lnk"
$uninstLnk = Join-Path $StartMenu "Shot-by-Shot (Uninstall).lnk"

$sc = $Wsh.CreateShortcut($webLnk)
$sc.TargetPath = Join-Path $Root "app-start.bat"
$sc.WorkingDirectory = $Root
$sc.Save()

$sc = $Wsh.CreateShortcut($deskLnk)
$sc.TargetPath = Join-Path $Root "start-desktop.bat"
$sc.WorkingDirectory = $Root
$sc.Save()

$sc = $Wsh.CreateShortcut($uninstLnk)
$sc.TargetPath = Join-Path $Root "uninstall.bat"
$sc.WorkingDirectory = $Root
$sc.Save()

Write-Ok "Shortcuts created in Start Menu."

# ---------------------------------------------------------------- done
Write-Step "Installation complete."
Write-Host ""
Write-Host "  Start Menu > 'Shot-by-Shot (Web)'      -> opens the web interface in your browser"
Write-Host "  Start Menu > 'Shot-by-Shot (Desktop)'  -> opens the desktop app window"
Write-Host ""
Write-Host "To remove everything later, run: uninstall.bat (or Start Menu > Shot-by-Shot (Uninstall))"
