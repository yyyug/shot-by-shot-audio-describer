# Shot-by-Shot packaging build (DESKTOP version only)
# 1. Creates a fresh build venv (isolated, does not pollute system Python)
# 2. Installs CPU-only PyTorch + dependencies + PyInstaller (+ PyArmor)
# 3. (Optional) Obfuscates your source with PyArmor
# 4. Builds a one-folder app (dist\BuddyADPortable)
#    including bundled SenseVoice models
# 5. If Inno Setup is installed, compiles a single-file UI installer
#    (dist\ShotByShot-Setup.exe) with shortcuts + uninstaller.
#
# Usage:  powershell -ExecutionPolicy Bypass -File build.ps1
#         powershell -ExecutionPolicy Bypass -File build.ps1 -SkipInno
#         powershell -ExecutionPolicy Bypass -File build.ps1 -Obfuscate
#         powershell -ExecutionPolicy Bypass -File build.ps1 -Obfuscate -SkipInno `
#             -DistDir dist-pyinstaller -WorkDir build-pyinstaller
#
# -DistDir / -WorkDir let a build land in its own folder so it never overwrites
# an existing distribution; both default to the historical dist\ and build\.

param(
    [switch]$SkipInno,
    [switch]$Obfuscate,
    [string]$DistDir = "dist",
    [string]$WorkDir = "build"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildVenv = Join-Path $Root ".build-venv"
$Ico = Join-Path $Root "packaging\shot.ico"
$DistPath = if ([System.IO.Path]::IsPathRooted($DistDir)) { $DistDir } else { Join-Path $Root $DistDir }
$WorkPath = if ([System.IO.Path]::IsPathRooted($WorkDir)) { $WorkDir } else { Join-Path $Root $WorkDir }
$PortableDir = Join-Path $DistPath "BuddyADPortable"

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

# ---------------------------------------------------------------- bundled ffmpeg
# The ONNX transcription path extracts audio via a bundled static ffmpeg, so
# end users do NOT need to install ffmpeg themselves. Downloaded into
# tools\ffmpeg (gitignored) once, then packaged into _internal\tools\ffmpeg.
$FfmpegDir = Join-Path $Root "tools\ffmpeg"
$FfmpegExe = Join-Path $FfmpegDir "ffmpeg.exe"
if (Test-Path $FfmpegExe) {
    Write-Ok "Bundled ffmpeg already present: $FfmpegExe"
} else {
    Write-Step "Downloading static ffmpeg for bundling (~100MB, one-time)..."
    New-Item -ItemType Directory -Path $FfmpegDir -Force | Out-Null
    $ffZip = Join-Path (Join-Path $Root "tools") "ffmpeg-release-essentials.zip"
    $ffUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    try {
        Invoke-WebRequest -Uri $ffUrl -OutFile $ffZip -UseBasicParsing
    } catch {
        Write-Warn "ffmpeg download failed ($($_.Exception.Message))."
        Write-Warn "Place ffmpeg.exe + ffprobe.exe in tools\ffmpeg\ manually, then re-run."
        exit 1
    }
    $ffTmp = Join-Path (Join-Path $Root "tools") "ffmpeg-tmp"
    Remove-Item $ffTmp -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -Path $ffZip -DestinationPath $ffTmp -Force
    $ffBin = Get-ChildItem (Join-Path $ffTmp "ffmpeg-*") -Directory | Select-Object -First 1
    Copy-Item (Join-Path $ffBin.FullName "bin\ffmpeg.exe") $FfmpegExe -Force
    Copy-Item (Join-Path $ffBin.FullName "bin\ffprobe.exe") (Join-Path $FfmpegDir "ffprobe.exe") -Force
    Remove-Item $ffTmp -Recurse -Force
    Remove-Item $ffZip -Force
    Write-Ok "Bundled ffmpeg ready: $FfmpegExe"
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
    --distpath $DistPath `
    --workpath $WorkPath `
    (Join-Path $Root "packaging\ShotByShotDesktop.spec")
if ($LASTEXITCODE -ne 0) { Write-Warn "PyInstaller build (desktop) failed."; exit 1 }
Write-Ok "Built: $PortableDir"

# ---------------------------------------------------------------- standalone (Tauri shell)
# Optional extra: a Rust/Tauri shell that loads the same Flask backend it spawns
# (BuddyADWeb.exe) into its own WebView2 window. Skips gracefully if Rust or the
# MSVC build tools are missing.
Write-Step "Building standalone shell (Tauri/Rust, optional)..."
$CargoExe = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
$StandaloneManifest = Join-Path $Root "standalone\Cargo.toml"
$StandaloneExe = Join-Path $Root "standalone\target\release\buddy-ad-standalone.exe"
$StandaloneDst = Join-Path $PortableDir "BuddyADStandalone.exe"
if (Test-Path $StandaloneExe) { Remove-Item $StandaloneExe -Force }
if (Test-Path $CargoExe) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vsDir = $null
    if (Test-Path $vswhere) {
        $vsDir = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
    }
    $vcvars = if ($vsDir) { Join-Path $vsDir "VC\Auxiliary\Build\vcvars64.bat" } else { $null }
    if ($vcvars -and (Test-Path $vcvars)) {
        & cmd.exe /c "call `"$vcvars`" >nul 2>&1 && `"$CargoExe`" build --release --manifest-path `"$StandaloneManifest`""
        if ($LASTEXITCODE -eq 0 -and (Test-Path $StandaloneExe)) {
            Copy-Item $StandaloneExe $StandaloneDst -Force
            Write-Ok "Built: $StandaloneDst"
        } else {
            Write-Warn "Standalone build failed; continuing without BuddyADStandalone.exe"
        }
    } else {
        Write-Warn "MSVC Build Tools not found; skipping standalone shell."
    }
} else {
    Write-Warn "Cargo not found; skipping standalone shell."
}

# ---------------------------------------------------------------- few-shot training data
Write-Step "Bundling few-shot GT training data (stage2\gt_ad_train)..."
$gtSrc = Join-Path $Root "stage2\gt_ad_train"
$gtDst = Join-Path $PortableDir "_internal\stage2\gt_ad_train"
if ((Test-Path $gtSrc) -and -not (Test-Path $gtDst)) {
    New-Item -ItemType Directory -Path $gtDst -Force | Out-Null
    Copy-Item (Join-Path $gtSrc "cmdad_train.csv") $gtDst -Force
    Copy-Item (Join-Path $gtSrc "tvad_train.csv") $gtDst -Force
    Write-Ok "Copied GT training CSVs ($gtDst)"
} else {
    Write-Ok "GT training data present or source missing; skipped."
}

# ---------------------------------------------------------------- trim long paths
Write-Step "Trimming deeply-nested third-party license files (avoids Inno long-path errors)..."
$trimmed = 0
Get-ChildItem (Join-Path $PortableDir "_internal") -Directory -Recurse `
    | Where-Object { $_.FullName -match "torch-[^\\]+\.dist-info\\licenses\\third_party$" } | ForEach-Object {
    Remove-Item $_.FullName -Recurse -Force
    $trimmed++
}
if ($trimmed -gt 0) { Write-Ok "Removed $trimmed license trees." }

# ---------------------------------------------------------------- portable extras
Write-Step "Adding portable extras (Outputs folder shortcut)..."
$batSrc = Join-Path $Root "packaging\Shot-by-Shot Outputs.bat"
Copy-Item $batSrc $PortableDir -Force
Write-Ok "Added 'Shot-by-Shot Outputs.bat' next to the exe."

# ---------------------------------------------------------------- inno setup
# The installer .iss hardcodes dist\BuddyADPortable, so it only makes sense
# for the default output folder; a custom -DistDir is a portable-only build.
if ($SkipInno) { Write-Ok "Skipping Inno Setup (as requested)."; exit 0 }
if ($DistPath -ne (Join-Path $Root "dist")) {
    Write-Ok "Skipping Inno Setup (portable-only build in $DistPath)."
    exit 0
}

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
    Write-Ok "Installer: $(Join-Path $Root 'dist\BuddyAD-Setup.exe')"
} else {
    Write-Warn "Inno Setup not found - skipping installer step."
    Write-Warn "Download free Inno Setup 6 from https://jrsoftware.org/isinfo.php then re-run build.ps1"
    Write-Warn "The one-folder build in dist\BuddyADPortable can be zipped and distributed as-is."
}

Write-Step "Done."
