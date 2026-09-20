# Shot-by-Shot Nuitka build (OPTIONAL - stronger protection than PyInstaller)
#
# PyInstaller bundles .pyc bytecode that can be extracted (pyinstxtractor)
# and decompiled back to readable source. Nuitka instead COMPILES your Python
# source to native C / machine code, so the distribution contains no readable
# Python - much harder to reverse-engineer.
#
# Trade-offs vs build.ps1 (PyInstaller):
#   + Real native-compiled binaries (protects your source)
#   + Faster startup than PyInstaller onefile
#   - Compilation of torch/funasr is SLOW (can take 30-90 min on first run)
#   - Larger build output
#   - Needs a working C compiler (MSVC via Visual Studio Build Tools)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File build-nuitka.ps1
#
# Notes:
#   - Third-party .pyd modules (torch, opencv, etc.) are already compiled C
#     code and are copied as-is; only YOUR .py code gets compiled to native.
#   - If Nuitka misses dynamic imports, add --include-package / --include-module.

param()

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildVenv = Join-Path $Root ".build-venv"

function Write-Step([string]$Msg) { Write-Host "`n==> $Msg" -ForegroundColor Cyan }
function Write-Ok([string]$Msg) { Write-Host "    $Msg" -ForegroundColor Green }
function Write-Warn([string]$Msg) { Write-Host "    $Msg" -ForegroundColor Yellow }

Write-Step "Shot-by-Shot Nuitka build (native compilation)"

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
Write-Step "Installing dependencies + Nuitka (large download - torch etc.)..."
& $VenvPython -m pip install --upgrade pip
& $VenvPip install -r (Join-Path $Root "requirements-cpu.txt")
& $VenvPip install -r (Join-Path $Root "requirements.txt")
& $VenvPip install -r (Join-Path $Root "requirements-desktop.txt")
& $VenvPip install nuitka
if ($LASTEXITCODE -ne 0) { Write-Warn "pip install failed."; exit 1 }

# ---------------------------------------------------------------- common args
# Data dirs bundled at the same relative paths (used by frozen-path code).
$DataArgs = @(
    "--include-data-dir=$Root\templates=templates",
    "--include-data-dir=$Root\static=static",
    "--include-data-dir=$Root\models\sensevoice=models\sensevoice",
    # Few-shot GT AD sentences read by processing/llm_summarizer at import time
    # (resolved as <processing>/../stage2/gt_ad_train). Only the two the app
    # actually reads are shipped - madeval_train.csv is used solely by the
    # unwired stage2/main_*.py research scripts, so it stays out of the bundle.
    "--include-data-files=$Root\stage2\gt_ad_train\cmdad_train.csv=stage2\gt_ad_train\cmdad_train.csv",
    "--include-data-files=$Root\stage2\gt_ad_train\tvad_train.csv=stage2\gt_ad_train\tvad_train.csv"
)
# The packaged app is ONNX-only (funasr_onnx + onnxruntime): the torch-backed
# transcription fallback and the DINOv2 film-grammar helpers are never called,
# so the whole torch stack stays out of the build. This is also what keeps the
# compile time in minutes instead of hours.
$NoFollowArgs = @(
    "--nofollow-import-to=matplotlib",
    "--nofollow-import-to=IPython",
    "--nofollow-import-to=jupyter",
    "--nofollow-import-to=notebook",
    "--nofollow-import-to=tkinter",
    "--nofollow-import-to=PyQt5",
    "--nofollow-import-to=PyQt6",
    "--nofollow-import-to=PySide2",
    "--nofollow-import-to=PySide6",
    "--nofollow-import-to=torch",
    "--nofollow-import-to=torchvision",
    "--nofollow-import-to=torchaudio",
    "--nofollow-import-to=funasr",
    "--nofollow-import-to=modelscope",
    "--nofollow-import-to=transformers",
    "--nofollow-import-to=tensorboard"
)
# funasr_onnx is imported lazily inside functions; jieba ships a dict.txt that
# it loads at runtime, so its data files must be included explicitly.
$IncludeArgs = @(
    "--include-package=processing",
    "--include-package=funasr_onnx",
    "--include-package-data=jieba"
)
$CommonArgs = @(
    "--standalone",
    "--output-dir=$Root\dist-nuitka",
    "--windows-icon-from-ico=$Root\packaging\shot.ico",
    "--assume-yes-for-downloads",
    # Huge generated C modules (mpmath/libmp, google.genai.types, ...) killed the
    # default parallel cl.exe jobs with OOM ("Compiler terminating ... Abort
    # complete", exit 3221225786) when 8 of them ran at once. Keep the lean
    # compiler settings from --low-memory, but run 2 compile jobs instead of its
    # default of 1 - an explicit --jobs overrides that default (Nuitka's
    # Options.getJobLimit), so both options combine.
    "--low-memory",
    "--jobs=2"
) + $NoFollowArgs + $IncludeArgs + $DataArgs

Write-Step "Compiling desktop app with Nuitka (native, no readable Python)..."
Push-Location $Root
& $VenvPython -m nuitka @CommonArgs --windows-console-mode=disable `
    --output-filename=ShotByShotDesktop.exe desktop.py
if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Warn "Nuitka build (desktop) failed."; exit 1 }
Write-Ok "Built: $(Join-Path $Root 'dist-nuitka\desktop.dist\ShotByShotDesktop.exe')"

Write-Step "Compiling web app with Nuitka..."
& $VenvPython -m nuitka @CommonArgs --windows-console-mode=force `
    --output-filename=ShotByShotWeb.exe webapp_entry.py
if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Warn "Nuitka build (web) failed."; exit 1 }
Write-Ok "Built: $(Join-Path $Root 'dist-nuitka\webapp_entry.dist\ShotByShotWeb.exe')"
Pop-Location

Write-Step "Done. Both .dist folders are self-contained and need no Python."
Write-Warn "Zip them for distribution, or point ShotByShot.iss at the Nuitka output."
