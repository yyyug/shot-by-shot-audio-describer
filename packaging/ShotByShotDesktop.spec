# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for Shot-by-Shot: builds BOTH executables into one
# one-folder bundle that shares a single _internal directory:
#
#   - ShotByShotDesktop.exe : pywebview window app (desktop.py)
#   - ShotByShotWeb.exe     : local Flask server + auto browser open
#                             (webapp_entry.py, console kept visible so the
#                              user can stop the server by closing it)
#
# Optional env vars:
#   SBS_ENTRY      path to a PyArmor-obfuscated desktop.py (protected build)
#   SBS_WEB_ENTRY  path to a PyArmor-obfuscated webapp_entry.py
#
# Usage: pyinstaller packaging\ShotByShotDesktop.spec

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(SPECPATH))

DESKTOP_ENTRY = os.environ.get("SBS_ENTRY") or os.path.join(PROJECT_ROOT, "desktop.py")
WEB_ENTRY = os.environ.get("SBS_WEB_ENTRY") or os.path.join(PROJECT_ROOT, "webapp_entry.py")

DESKTOP_PATH = [os.path.dirname(DESKTOP_ENTRY), PROJECT_ROOT]
WEB_PATH = [os.path.dirname(WEB_ENTRY), PROJECT_ROOT]

# Third-party deps of the obfuscated modules: PyArmor-encrypted bytecode is
# opaque to PyInstaller's import analysis, so every import made from
# desktop.py / webapp_entry.py / processing/* must be listed here.
COMMON_HIDDEN = [
    "processing",
    "processing.shot_detector",
    "processing.sensevoice_transcriber",
    "processing.dialogue_gap_detector",
    "processing.vlm_describer",
    "processing.llm_summarizer",
    "processing.film_grammar",
    "processing.character_recognizer",
    "processing.vtt_writer",
    "scenedetect",
    "cv2",
    "pandas",
    "numpy",
    "requests",
    "PIL",
    "torch",
    "torchvision",
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
    # funasr / modelscope dynamic imports
    "funasr",
    "funasr.models",
    "funasr.utils.postprocess_utils",
    "modelscope",
    "modelscope.pipelines",
]

DESKTOP_HIDDEN = COMMON_HIDDEN + [
    # pywebview backends (edgtw / mshtml / cef) get picked up dynamically
    "webview",
    "webview.platforms",
    "webview.platforms.edgtw",
]

WEB_HIDDEN = COMMON_HIDDEN + [
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
]

DATA = [
    (os.path.join(PROJECT_ROOT, "templates"), "templates"),
    (os.path.join(PROJECT_ROOT, "static"), "static"),
    (os.path.join(PROJECT_ROOT, "models", "sensevoice"), os.path.join("models", "sensevoice")),
]

a = Analysis(
    [DESKTOP_ENTRY],
    pathex=DESKTOP_PATH,
    binaries=[],
    datas=DATA,
    hiddenimports=DESKTOP_HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

b = Analysis(
    [WEB_ENTRY],
    pathex=WEB_PATH,
    binaries=[],
    datas=[],  # templates/static/models are already collected via `a`
    hiddenimports=WEB_HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

# Both analyses resolve the same shared modules (processing/*, torch, ...),
# so merge their TOCs while dropping duplicate entries - PyInstaller errors
# out on repeated names inside a single PYZ / COLLECT.
def _merge_unique(primary_toc, secondary_toc):
    seen = set(item[0] for item in primary_toc)
    merged = list(primary_toc)
    for item in secondary_toc:
        if item[0] not in seen:
            seen.add(item[0])
            merged.append(item)
    return merged


pure = _merge_unique(a.pure, b.pure)
binaries = _merge_unique(a.binaries, b.binaries)
datas = _merge_unique(a.datas, b.datas)

pyz = PYZ(pure)

exe_desktop = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ShotByShotDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join(PROJECT_ROOT, "packaging", "shot.ico"),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

exe_web = EXE(
    pyz,
    b.scripts,
    [],
    exclude_binaries=True,
    name="ShotByShotWeb",
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
    exe_desktop,
    exe_web,
    binaries,
    datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ShotByShotDesktop",
)
