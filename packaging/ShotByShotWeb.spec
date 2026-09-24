# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for Shot-by-Shot: builds the ONE web server executable
# into a one-folder bundle:
#
#   - BuddyADWeb.exe : local Flask server (webapp_entry.py). Running it
#                      directly opens the browser; the standalone BuddyAD.exe
#                      shell spawns it with SBS_NO_BROWSER=1 so no browser
#                      window appears.
#
# Optional env vars:
#   SBS_WEB_ENTRY  path to a PyArmor-obfuscated webapp_entry.py
#
# Usage: pyinstaller packaging\ShotByShotWeb.spec

import os

try:
    from PyInstaller.utils.hooks import collect_all
except ImportError:
    collect_all = None

PROJECT_ROOT = os.path.dirname(os.path.abspath(SPECPATH))

WEB_ENTRY = os.environ.get("SBS_WEB_ENTRY") or os.path.join(PROJECT_ROOT, "webapp_entry.py")
WEB_PATH = [os.path.dirname(WEB_ENTRY), PROJECT_ROOT]

# Third-party deps of the obfuscated modules: PyArmor-encrypted bytecode is
# opaque to PyInstaller's import analysis, so every import made from
# webapp_entry.py / app.py / processing/* must be listed here.
COMMON_HIDDEN = [
    "processing",
    "processing._api_common",
    "processing.shot_detector",
    "processing.sensevoice_transcriber",
    "processing.dialogue_gap_detector",
    "processing.vlm_describer",
    "processing.llm_summarizer",
    "processing.film_grammar",
    "processing.character_recognizer",
    "processing.vtt_writer",
    "processing.history_engine",
    "processing.time_ranges",
    "scenedetect",
    "cv2",
    "pandas",
    "numpy",
    "requests",
    "PIL",
    "sklearn.cluster",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "google.genai",
    "google.genai.types",
    # promptloader lives in stage1/ and stage2/ (same name, different
    # content); processing modules now import them as real packages.
    "stage1",
    "stage1.promptloader",
    "stage2",
    "stage2.promptloader",
    # funasr / modelscope are excluded (ONNX-only runtime); keep only the
    # funasr-onnx backend that actually runs.
    "funasr_onnx",
    "funasr_onnx.sensevoice_bin",
    "funasr_onnx.utils",
    "funasr_onnx.vad_bin",
    "jieba",
    "onnxruntime",
]

# onnxruntime ships many providers/data files that PyInstaller does not pick
# up otherwise; collect them wholesale into the bundle.
_ONNX_RT = collect_all("onnxruntime") if collect_all else ([], [], [])
_ONNX_DATAS, _ONNX_BINARIES, _ONNX_HIDDEN = _ONNX_RT
FA_ONNX_HIDDEN = []
if collect_all:
    try:
        _FA = collect_all("funasr_onnx")
        FA_ONNX_HIDDEN = list(_FA[2])
        _ONNX_DATAS = list(_ONNX_DATAS) + list(_FA[0])
        _ONNX_BINARIES = list(_ONNX_BINARIES) + list(_FA[1])
    except Exception:
        pass

# jieba is imported at module level by funasr_onnx.utils.utils (used by the
# ONNX transcription path). It lazily loads dict.txt + analyse/finalseg/
# posseg/lac_small data files at runtime, so they must be bundled too - a
# plain hiddenimport only ships the .py files and jieba fails to initialise.
JIEBA_HIDDEN = []
if collect_all:
    try:
        _JB = collect_all("jieba")
        JIEBA_HIDDEN = list(_JB[2])
        _ONNX_DATAS = list(_ONNX_DATAS) + list(_JB[0])
        _ONNX_BINARIES = list(_ONNX_BINARIES) + list(_JB[1])
    except Exception:
        pass

# PyArmor-encrypted modules are opaque to PyInstaller's static import scan, so
# every stdlib submodule they use must be listed explicitly too - otherwise the
# frozen app dies at startup with e.g. "No module named 'logging.handlers'".
STDLIB_HIDDEN = [
    "ast",
    "base64",
    "copy",
    "ctypes",
    "datetime",
    "functools",
    "io",
    "json",
    "logging",
    "logging.handlers",
    "math",
    "os",
    "pathlib",
    "random",
    "re",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "typing",
    "urllib.request",
    "uuid",
    "warnings",
    "wave",
    "webbrowser",
    "winreg",
    "PIL.Image",
    "werkzeug.utils",
]

WEB_HIDDEN = COMMON_HIDDEN + STDLIB_HIDDEN + _ONNX_HIDDEN + FA_ONNX_HIDDEN + JIEBA_HIDDEN + [
    # obfuscated webapp_entry.py's `from app import app` is invisible to
    # static analysis - pull the Flask glue module in explicitly
    "app",
    "flask",
    "flask_cors",
]

EXCLUDES = [
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    # The desktop (pywebview) variant is no longer built; drop its backends.
    "webview",
    "webview.platforms.cef",
    "webview.platforms.gtk",
    "webview.platforms.qt",
    "webview.platforms.mshtml",
    "webview.platforms.cocoa",
    "webview.platforms.edgtw",
    # Research-only / optional across the dependency tree; nothing imports
    # these at runtime and dropping them keeps the PYZ leaner.
    "sympy",
    "pygments",
    "pytest",
    # The packaged app runs ONNX-only, and processing/film_grammar.py no longer
    # imports torch at module scope (the DINOv2 shot-scale / thread helpers are
    # lazily loaded and never called by the pipeline - shot scales come from the
    # UI). Dropping the whole PyTorch stack is by far the biggest size win.
    "torch",
    "torchvision",
    "torchaudio",
    "decord",
    # funasr/torch-legacy extras pulled in through static analysis of the
    # torch transcription fallback. None of these are imported at runtime.
    "funasr",
    "modelscope",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "hf_xet",
    "tiktoken",
    "aliyunsdkcore",
    "wandb",
    "tensorboard",
]

# Few-shot ground-truth AD examples used by processing/llm_summarizer at
# runtime (resolved relative to the processing module as ../stage2/gt_ad_train).
# Only cmdad (movies) and tvad (TV series / stage) are read by the app;
# madeval_train.csv is used solely by the unwired stage2/main_*.py research
# scripts, so it is intentionally not bundled.
GT_TRAIN_SRC = os.path.join(PROJECT_ROOT, "stage2", "gt_ad_train")
GT_TRAIN_DATA = [
    (os.path.join(GT_TRAIN_SRC, "cmdad_train.csv"), os.path.join("stage2", "gt_ad_train")),
    (os.path.join(GT_TRAIN_SRC, "tvad_train.csv"), os.path.join("stage2", "gt_ad_train")),
]

DATA = [
    (os.path.join(PROJECT_ROOT, "templates"), "templates"),
    (os.path.join(PROJECT_ROOT, "static"), "static"),
    (os.path.join(PROJECT_ROOT, "models", "sensevoice"), os.path.join("models", "sensevoice")),
] + GT_TRAIN_DATA

# Bundled static ffmpeg (fetched by build.ps1 into tools\ffmpeg; see
# processing/sensevoice_transcriber._ffmpeg_bin for runtime resolution).
_FFMPEG_DIR = os.path.join(PROJECT_ROOT, "tools", "ffmpeg")
_FFMPEG_DATA = []
for _exe in ("ffmpeg.exe", "ffprobe.exe"):
    _p = os.path.join(_FFMPEG_DIR, _exe)
    if os.path.isfile(_p):
        _FFMPEG_DATA.append((_p, os.path.join("tools", "ffmpeg")))
if not _FFMPEG_DATA:
    raise SystemExit(
        "tools\\ffmpeg\\ffmpeg.exe missing - run build.ps1 to fetch the "
        "bundled ffmpeg binary before packaging."
    )

a = Analysis(
    [WEB_ENTRY],
    pathex=WEB_PATH,
    binaries=(_ONNX_BINARIES or []),
    datas=(DATA + _FFMPEG_DATA + _ONNX_DATAS),
    hiddenimports=WEB_HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe_web = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BuddyADWeb",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # visible console = easy way to stop the server
    icon=os.path.join(PROJECT_ROOT, "packaging", "shot.ico"),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_web,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BuddyADPortable",
)