# Shot-by-Shot packaging build (DESKTOP version only)
# 1. Creates a fresh build venv (isolated, does not pollute system Python)
# 2. Installs CPU-only PyTorch + dependencies + PyInstaller (+ PyArmor)
# 3. (Optional) Obfuscates your source with PyArmor
# 4. Builds a one-folder desktop app (dist\ShotByShotDesktop)
#    including bundled SenseVoice models
# 5. If Inno Setup is installed, compiles a single-file UI installer
#    (dist\ShotByShot-Setup.exe) with shortcuts + uninstaller.
#
# Usage:  powershell -ExecutionPolicy Bypass -File build.ps1
#         powershell -ExecutionPolicy Bypass -File build.ps1 -SkipInno
#         powershell -ExecutionPolicy Bypass -File build.ps1 -Obfuscate

param(
    [switch]$SkipInno,
    [switch]$Obfuscate
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildVenv = Join-Path $Root ".build-venv"
$Ico = Join-Path $Root "packaging\shot.ico"

function Write-Step([string]$Msg) { Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Write-Ok([string]$Msg) { Write-Host "    $Msg" -ForegroundColor Green }
function Write-Warn([string]$Msg) { Write-Host "    $Msg" -ForegroundColor Yellow }

Write-Step "Shot-by-Shot packaging build"

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
if (-not $pyParts) { Write-Warn "Python 3 not found. Install Python 3.8+ first."; exit 1 }

# ---------------------------------------------------------------- build venv
if (Test-Path (Join-Path $BuildVenv "Scripts\python.exe")) {
    Write-Ok "Reusing build venv: $BuildVenv"
} else {
    Write-Step "Creating isolated build venv..."
    $rest = if ($pyParts.Count -gt 1) { $pyParts[1..($pyParts.Count - 1)] } else { @() }
    & $pyParts[0] @($rest + @("-m", "venv", $BuildVenv))
    if ($LASTEXITCODE -ne 0) { Write-Warn "Failed to create build venv"; exit 1 }
}
$VenvPython = Join-Path $BuildVenv "Scripts\python.exe"
$VenvPip = Join-Path $BuildVenv "Scripts\pip.exe"

# ---------------------------------------------------------------- deps
Write-Step "Installing dependencies + PyInstaller (this may take a while)..."
& $VenvPython -m pip install --upgrade pip
& $VenvPip install -r (Join-Path $Root "requirements-cpu.txt")
& $VenvPip install -r (Join-Path $Root "requirements.txt")
& $VenvPip install -r (Join-Path $Root "requirements-desktop.txt")
& $VenvPip install pyinstaller
if ($Obfuscate) { & $VenvPip install pyarmor }
if ($LASTEXITCODE -ne 0) { Write-Warn "pip install failed."; exit 1 }

# ---------------------------------------------------------------- icon
if (-not (Test-Path $Ico)) {
    Write-Step "Generating app icon..."
    & $VenvPython (Join-Path $Root "packaging\make_icon.py")
}

# ---------------------------------------------------------------- obfuscate
if ($Obfuscate) {
    $ObfDir = Join-Path $Root ".build-obf"
    $VenvPyarmor = Join-Path $BuildVenv "Scripts\pyarmor.exe"
    Write-Step "Obfuscating source with PyArmor..."
    Remove-Item $ObfDir -Recurse -Force -ErrorAction SilentlyContinue
    & $VenvPyarmor gen -O $ObfDir (Join-Path $Root "desktop.py") (Join-Path $Root "webapp_entry.py") (Join-Path $Root "app.py") (Join-Path $Root "processing")
    if ($LASTEXITCODE -ne 0) { Write-Warn "PyArmor obfuscation failed."; exit 1 }
    $env:SBS_ENTRY = (Join-Path $ObfDir "desktop.py")
    $env:SBS_WEB_ENTRY = (Join-Path $ObfDir "webapp_entry.py")
    Write-Ok "Obfuscated entry: $env:SBS_ENTRY"
} else {
    Remove-Item Env:SBS_ENTRY -ErrorAction SilentlyContinue
    Remove-Item Env:SBS_WEB_ENTRY -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------- pyinstaller
Write-Step "Running PyInstaller (desktop + web versions - bundles ~1GB of models, may take 10+ minutes)..."
& $VenvPython -m PyInstaller --noconfirm --clean `
    --distpath (Join-Path $Root "dist") `
    --workpath (Join-Path $Root "build") `
    (Join-Path $Root "packaging\ShotByShotDesktop.spec")
if ($LASTEXITCODE -ne 0) { Write-Warn "PyInstaller build (desktop) failed."; exit 1 }
Write-Ok "Built: $(Join-Path $Root 'dist\ShotByShotDesktop')"

# ---------------------------------------------------------------- trim long paths
Write-Step "Trimming deeply-nested third-party license files (avoids Inno long-path errors)..."
$trimmed = 0
Get-ChildItem (Join-Path $Root "dist\ShotByShotDesktop\_internal") -Directory -Recurse `
    | Where-Object { $_.FullName -match "torch-[^\\]+\.dist-info\\licenses\\third_party$" } | ForEach-Object {
    Remove-Item $_.FullName -Recurse -Force
    $trimmed++
}
if ($trimmed -gt 0) { Write-Ok "Removed $trimmed license trees." }

# ---------------------------------------------------------------- portable extras
Write-Step "Adding portable extras (Outputs folder shortcut)..."
$batSrc = Join-Path $Root "packaging\Shot-by-Shot Outputs.bat"
Copy-Item $batSrc (Join-Path $Root "dist\ShotByShotDesktop") -Force
Write-Ok "Added 'Shot-by-Shot Outputs.bat' next to the exe."

# ---------------------------------------------------------------- inno setup
if ($SkipInno) { Write-Ok "Skipping Inno Setup (as requested)."; exit 0 }

$iscc = $null
foreach ($candidate in @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 5\ISCC.exe"
)) {
    if (Test-Path $candidate) { $iscc = $candidate; break }
}

if ($iscc) {
    Write-Step "Building single-file installer with Inno Setup..."
    & $iscc (Join-Path $Root "packaging\ShotByShot.iss")
    if ($LASTEXITCODE -ne 0) { Write-Warn "Inno Setup compile failed."; exit 1 }
    Write-Ok "Installer: $(Join-Path $Root 'dist\ShotByShot-Setup.exe')"
} else {
    Write-Warn "Inno Setup not found - skipping installer step."
    Write-Warn "Download free Inno Setup 6 from https://jrsoftware.org/isinfo.php then re-run build.ps1"
    Write-Warn "The one-folder build in dist\ShotByShotDesktop can be zipped and distributed as-is."
}

Write-Step "Done."
