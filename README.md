# Shot-by-Shot: 影片逐鏡頭分析工具 / Video Shot-by-Shot Audio Describer

[English](#english) | [中文](#中文)

---

## English

### Overview

This tool processes video files to generate shot-by-shot analysis with:
- Shot boundary detection (PySceneDetect)
- Audio transcription + dialogue gap detection (SenseVoice / FunASR)
- **Character bank** — face detection & clustering across shots
- **Context extension** — past/future shot context for AD placement
- AI-powered video descriptions — **Multi-backend**: Gemini, Qwen (DashScope), DeepSeek, or any OpenAI-compatible endpoint
- Audio description summarization (Stage 2 LLM)

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- SenseVoice (FunASR) for transcription, optional

**Backend-specific:**
- **Gemini**: API key + `pip install google-generativeai`
- **Qwen / DeepSeek**: API key for the respective endpoint
- **OpenAI-compatible**: any base URL + model name (e.g. OpenRouter, a local vLLM server)

### Distribution / Installer (recommended for end users)

A ready-to-use installer is included. It sets up everything **in an isolated
virtual environment** so your system Python is never touched, uses the
**pre-downloaded SenseVoice models** bundled in `models/sensevoice/` (no download
on first run), and creates Start Menu shortcuts.

| File | Purpose |
|------|---------|
| `setup.bat` | Double-click installer (venv + deps + shortcuts) |
| `app-start.bat` | Launch the web interface |
| `start-desktop.bat` | Launch the desktop app (pywebview) |
| `uninstall.bat` | Remove shortcuts, venv, models, and optionally the app folder |

Steps:

```text
1. Copy the whole folder (including models/) to the target machine.
2. Double-click setup.bat (requires Python 3.8+ already installed).
3. Start Menu > "Shot-by-Shot (Web)" to use it.
4. Start Menu > "Shot-by-Shot (Uninstall)" to remove it.
```

The virtual environment lives in `.venv/` and SenseVoice models in
`models/sensevoice/`. Uninstalling only removes these + shortcuts; your system
Python and other projects are unaffected.

### Packaging a single installer (no Python needed on target machines)

End users who do not have Python can get a self-contained Windows desktop app.

1. **Pre-download the models** (already done in this repo — `models/sensevoice/`).
2. Run `build.ps1` (PowerShell):
   - Builds the **desktop app** with **PyInstaller** into `dist/ShotByShotDesktop/`
     (includes templates, static files, and the ~1GB SenseVoice models).
   - If **Inno Setup 6** (free, https://jrsoftware.org/isinfo.php) is installed,
     also compiles `dist/ShotByShot-Setup.exe` — a single-file **UI installer**
     with shortcuts and an uninstall entry in Windows "Apps & Features".

```text
powershell -ExecutionPolicy Bypass -File build.ps1
# or skip the Inno Setup step:
powershell -ExecutionPolicy Bypass -File build.ps1 -SkipInno
```

Notes:
- The bundle is large (~1.5–2GB) because the int8-quantized SenseVoice model
  (~235MB, `model_quant.onnx`) is bundled for offline use — this is the trade-off
  for a single offline installer with no download on first run.
- The packaged desktop app stores outputs and the log in
  `%LOCALAPPDATA%\ShotByShot\` (writable even under Program Files).
- Only `models/sensevoice/` is bundled. Older unrelated files in `models/`
  (`whisper/`, `shot_scale_ckpt.pth`) are **not** used by the current pipeline
  (shot detection is PySceneDetect, not a model file) and are excluded.
- If PyInstaller misses any dynamic imports (funasr/modelscope), add them to
  `hiddenimports` in `packaging/ShotByShotDesktop.spec` and rebuild.
- `dist/ShotByShotDesktop/` can also be zipped and distributed as a portable app.

### Build environment packages (`.build-venv`)

`build.ps1 -Obfuscate` creates `.build-venv` (standard `python -m venv`, CPython
3.11) and installs exactly these groups — nothing touches your system Python:

| Source | Packages (versions from current build) |
|--------|------------------------------------------|
| `requirements-cpu.txt` | torch 2.13.0+cpu, torchaudio 2.11.0+cpu, torchvision 0.28.0+cpu (PyTorch CPU index) |
| `requirements.txt` | flask 3.1.3, flask-cors 6.0.5, scenedetect 0.7.1, pandas 3.0.5, numpy 2.4.6, requests 2.34.2, werkzeug 3.1.8, opencv-python 4.14.0 (**pinned `<5`**), scikit-learn 1.9.0, funasr 1.4.2, google-genai 2.19.0, num2words 0.5.14 |
| `requirements-desktop.txt` | pywebview 6.2.1 |
| build tools | pyinstaller 6.22.1, pyarmor 9.2.6 (trial — buy a license for commercial distribution), pytest 9.1.1 |

Hard-won pins / gotchas:
- **opencv-python must stay `<5`**: OpenCV 5 removed `CascadeClassifier`
  (`processing/character_recognizer.py`). The module now loads the cascade
  lazily, but keep 4.x so face detection actually works.
- **google-genai** and **num2words** are runtime deps of
  `processing/vlm_describer.py`, `processing/llm_summarizer.py` and
  `stage2/promptloader.py`.
- **PyTorch is not bundled** (and not needed at runtime):
  `processing/film_grammar.py` now imports `torch`/`torchvision` lazily, and the
  DINOv2 shot-scale / thread-structure helpers they serve are never called by
  the pipeline (shot scales come straight from the UI). The spec excludes the
  whole PyTorch stack, shrinking the portable build from a ~90k-file tree to
  ~1,900 files / ~1 GB. torch stays in `requirements-cpu.txt` for dev and tests.
- With `-Obfuscate`, PyArmor encrypts `desktop.py`, `webapp_entry.py`, `app.py`
  and `processing/*`. Their imports become invisible to PyInstaller's static
  analysis, so **every** import of those modules — including stdlib submodules
  such as `logging.handlers` and `urllib.request` — must be listed in
  `hiddenimports` (`packaging/ShotByShotDesktop.spec`, see `STDLIB_HIDDEN`).
  Miss one and the frozen app exits immediately with `ModuleNotFoundError`.

### Troubleshooting the packaged desktop app

- **Log file**: `%LOCALAPPDATA%\ShotByShot\shot_by_shot.log`. All uncaught
  Python exceptions are written here (excepthook). If a crash produces **no**
  log entry, it died at native level (WebView2/COM), not in Python code.
- **First launch on a new machine** may be slow or crash once: Windows Defender
  scans the bundle and WebView2 initializes its user profile on first
  run. Subsequent launches are normal.
- **WebView2 Runtime required** (pre-installed on Windows 11). The Setup.exe
  installs it automatically when missing; portable-zip users should download it
  from https://developer.microsoft.com/microsoft-edge/webview2/
- The app checks for WebView2 at startup and shows a clear message instead of
  crashing when it is absent.

### Optional: protect your Python source (Nuitka)

PyInstaller bundles `.pyc` bytecode that anyone can extract and decompile.
If you must protect your code, use **Nuitka** (`build-nuitka.ps1`): it
compiles your `.py` files to native C / machine code, so the shipped app
contains no readable Python. Third-party packages (torch, opencv, funasr)
are already-compiled `.pyd`/`.dll` files and are copied as-is — only **your**
code gets compiled to native.

```text
powershell -ExecutionPolicy Bypass -File build-nuitka.ps1
```

Trade-offs: first compile is slow (30–90 min, compiles torch's Python layer)
and needs MSVC Build Tools installed. Alternative: **PyArmor** encrypts the
bytecode instead of compiling (faster, still strong against casual decompiling).

### Installation (manual / developer)

```bash
git clone https://github.com/Jyxarthur/shot-by-shot.git
cd shot-by-shot
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
# optional desktop app:
pip install -r requirements-desktop.txt
python scripts\download_sensevoice.py   # pre-download SenseVoice models
```

### Usage

#### Web Interface

```bash
python app.py
```

Open http://localhost:5000 in your browser.

**Web Interface Options:**
| Option | Description |
|--------|-------------|
| **LLM Backend** | Gemini, Qwen, DeepSeek, or OpenAI-compatible |
| **API Key** | Key for the selected backend (auto-toggles) |
| **Enable Transcription** | SenseVoice transcription and dialogue-gap-based AD interval detection |
| **Language** | Language code (en, zh, etc.) or auto-detect |
| **Character Bank** | Face detection + clustering per shot |
| **Context Extension** | Extend AD intervals with surrounding shot context |
| **Video Type** | Movie or TV Series |
| **Skip Stage 2** | Disable AD summarization |
| **Output Format** | Basic, Full (with descriptions), or Original project format |

#### Command Line

```bash
# Full pipeline
python run_processing.py video.mp4 -o output.csv --api-key YOUR_API_KEY

# Skip steps
python run_processing.py video.mp4 --skip-whisper --api-key YOUR_API_KEY
python run_processing.py video.mp4 --skip-vlm
python run_processing.py video.mp4 --skip-whisper --skip-vlm --skip-stage2
python run_processing.py video.mp4 --context   # include surrounding shots in VLM prompts
```

#### Kaggle Notebooks

| Notebook | Description |
|----------|-------------|
| `kaggle_notebook.ipynb` | Basic pipeline (shot detection + VLM + Stage 2) |
| `notebook-ad-2.ipynb` | **Character bank edition** — face detection, dual backend (Qwen Unsloth / Gemini), Traditional Chinese prompts, customizable character names |

### Output Files

| File | Content |
|------|---------|
| `shot_by_shot_output.csv` | Main merged output (shots + subtitles + descriptions) |
| `stage1_descriptions.csv` | Detailed VLM descriptions |
| `stage2_audio_descriptions.csv` | Concise AD sentences |
| `character_bank.csv` | Per-shot character assignments (if character bank enabled) |

### Architecture

```
app.py  ──  Flask web server
├── processing/shot_detector.py         ✅ Wired
├── processing/sensevoice_transcriber.py ✅ Wired (web + desktop)
├── processing/dialogue_gap_detector.py ✅ Wired (AD interval from dialogue gaps)
├── processing/character_recognizer.py  ✅ Wired (web + notebook)
├── processing/context_extender.py      ✅ Wired (web)
├── processing/film_grammar.py          ✅ Wired (prompt variants)
├── processing/vlm_describer.py         ✅ Wired (Qwen / DeepSeek / OpenAI-compatible)
├── processing/llm_summarizer.py        ✅ Wired (Qwen / DeepSeek / OpenAI-compatible)
├── processing/csv_merger.py            ✅ Wired
├── stage1/promptloader.py              ✅ Wired (via vlm_describer)
├── stage2/promptloader.py              ✅ Wired (via llm_summarizer)
├── processing/action_scorer.py         ❌ Unwired (evaluation metric)
├── processing/shot_labeler.py          ❌ Unwired (visual annotation)
└── stage1/, stage2/, preprocess/       ❌ Unwired (research CLI scripts)
```

---

## 中文

### 概述

此工具用於處理影片檔案，生成逐鏡頭分析，包括：
- 鏡頭邊界檢測（PySceneDetect）
- 音訊轉錄及對話空隙偵測（SenseVoice / FunASR）
- **角色庫** — 全鏡頭人臉檢測與聚類
- **上下文擴展** — 前後鏡頭上下文用於口述影像配置
- AI 影片描述 — **多後端**：Gemini、Qwen（DashScope）、DeepSeek，或任何 OpenAI-compatible 端點
- 口述影像摘要（Stage 2 LLM）

### 系統要求

- Python 3.8+
- FFmpeg（用於影片處理）
- SenseVoice（FunASR，用於轉錄，可選）

**各後端要求：**
- **Gemini**：API 金鑰 + `pip install google-generativeai`
- **Qwen / DeepSeek**：對應端點的 API 金鑰
- **OpenAI-compatible**：任意 Base URL + model 名稱（例如 OpenRouter、本地 vLLM）

### 分發 / 安裝（推薦給一般使用者）

內建一鍵安裝程式。所有套件安裝在**隔離的虛擬環境**中，不會污染系統 Python；
使用隨附的**已預載 SenseVoice 模型**（位於 `models/sensevoice/`，第一次使用不需下載），並建立開始功能表捷徑。

| 檔案 | 用途 |
|------|------|
| `setup.bat` | 雙擊即安裝（venv + 依賴 + 捷徑） |
| `app-start.bat` | 啟動網頁介面 |
| `start-desktop.bat` | 啟動桌面版（pywebview） |
| `uninstall.bat` | 移除捷徑、venv、模型（可選擇是否刪除整個資料夾） |

步驟：

```text
1. 將整個資料夾（含 models/）複製到目標電腦。
2. 雙擊 setup.bat（需已安裝 Python 3.8+）。
3. 開始功能表 >「Shot-by-Shot (Web)」即可使用。
4. 開始功能表 >「Shot-by-Shot (Uninstall)」即可解除安裝。
```

虛擬環境位於 `.venv/`、SenseVoice 模型位於 `models/sensevoice/`。
解除安裝只會刪除這些檔案與捷徑，不影響系統 Python 或其他專案。

### 打包成單一安裝檔（目標電腦不需安裝 Python）

沒有 Python 的使用者也能拿到可獨立執行的 Windows **桌面版**應用程式。

1. **預先下載模型**（本 repo 已含 — `models/sensevoice/`）。
2. 執行 `build.ps1`（PowerShell）：
   - 用 **PyInstaller** 打包桌面版到 `dist/ShotByShotDesktop/`（內含 templates、static、
     約 1GB 的 SenseVoice 模型）。
   - 若已安裝 **Inno Setup 6**（免費，https://jrsoftware.org/isinfo.php），
     會一併編譯出 `dist/ShotByShot-Setup.exe` — 單一檔案的 **UI 安裝程式**，
     含捷徑與 Windows「應用程式與功能」中的解除安裝項目。

```text
powershell -ExecutionPolicy Bypass -File build.ps1
# 或跳過 Inno Setup 步驟：
powershell -ExecutionPolicy Bypass -File build.ps1 -SkipInno
```

注意事項：
- 安裝檔約 1.5–2GB，因為 int8 量化版 SenseVoice 模型（約 235MB，`model_quant.onnx`）隨附在內，以便離線使用。
- 打包版桌面應用程式的輸出與 log 放在 `%LOCALAPPDATA%\ShotByShot\`（Program Files 下也可寫入）。
- 只打包 `models/sensevoice/`。`models/` 下其他舊檔案（`whisper/`、`shot_scale_ckpt.pth`）
  **目前管線未使用**（鏡頭偵測用 PySceneDetect，不需模型檔），故排除。
- 若 PyInstaller 漏掉 funasr/modelscope 的動態 import，請在 `packaging/ShotByShotDesktop.spec` 的
  `hiddenimports` 補上後重新打包。
- `dist/ShotByShotDesktop/` 也可直接壓縮成 zip 當作可攜版分發。

### 建置環境套件（`.build-venv`）

`build.ps1 -Obfuscate` 會建立 `.build-venv`（標準 `python -m venv`，CPython 3.11）
並安裝以下套件——不會動到系統 Python：

| 來源 | 套件（目前建置版本） |
|------|----------------------|
| `requirements-cpu.txt` | torch 2.13.0+cpu、torchaudio 2.11.0+cpu、torchvision 0.28.0+cpu（PyTorch CPU index） |
| `requirements.txt` | flask 3.1.3、flask-cors 6.0.5、scenedetect 0.7.1、pandas 3.0.5、numpy 2.4.6、requests 2.34.2、werkzeug 3.1.8、opencv-python 4.14.0（**鎖定 `<5`**）、scikit-learn 1.9.0、funasr 1.4.2、google-genai 2.19.0、num2words 0.5.14 |
| `requirements-desktop.txt` | pywebview 6.2.1 |
| 建置工具 | pyinstaller 6.22.1、pyarmor 9.2.6（試用版——商用分發需購買授權）、pytest 9.1.1 |

重要釘選與陷阱：
- **opencv-python 必須 `<5`**：OpenCV 5 移除了 `CascadeClassifier`
  （`processing/character_recognizer.py` 使用）。模組已改為惰性載入不會炸，
  但要讓人臉偵測正常運作請維持 4.x。
- **google-genai** 與 **num2words** 是 `processing/vlm_describer.py`、
  `processing/llm_summarizer.py`、`stage2/promptloader.py` 的執行期依賴。
- **不再打包 PyTorch**（執行期也不需要）：`processing/film_grammar.py` 已改成
  惰性 import `torch`/`torchvision`，而它們支撐的 DINOv2 鏡頭景別／thread 預測
  目前管線根本不會呼叫（景別直接由 UI 提供）。spec 已排除整包 PyTorch，
  可攜版因此從約 9 萬個檔案縮到約 1,900 檔／約 1GB。torch 仍保留在
  `requirements-cpu.txt` 供開發與測試。
- 加 `-Obfuscate` 時，PyArmor 會加密 `desktop.py`、`webapp_entry.py`、`app.py`
  與 `processing/*`。PyInstaller 的靜態分析看不到這些模組的 import，
  因此**所有**用到的模組——包含標準庫子模組如 `logging.handlers`、
  `urllib.request`——都必須列在 `packaging/ShotByShotDesktop.spec` 的
  `hiddenimports`（見 `STDLIB_HIDDEN`）。漏掉一個，打包版會立刻以
  `ModuleNotFoundError` 結束。

### 打包版桌面應用疑難排解

- **Log 位置**：`%LOCALAPPDATA%\ShotByShot\shot_by_shot.log`。所有未捕捉的
  Python 例外都會寫進這裡（excepthook）。若崩潰時 log **沒有**新內容，
  代表是原生層級崩潰（WebView2/COM），不是 Python 程式碼問題。
- **新機器第一次啟動**可能較慢或崩潰一次：Windows Defender 要掃描整包檔案，
  WebView2 也要首次建立使用者設定檔；第二次之後即正常。
- **需要 Microsoft Edge WebView2 Runtime**（Windows 11 內建）。Setup.exe 偵測到
  缺少時會自動安裝；可攜版 zip 使用者請自行到
  https://developer.microsoft.com/microsoft-edge/webview2/ 下載安裝。
- 應用程式啟動時會預檢 WebView2，缺少時顯示明確訊息而非神秘崩潰。

### 選配：保護你的 Python 原始碼（Nuitka）

PyInstaller 包的是 `.pyc` 位元碼，別人可以用工具解壓並還原出原始碼。
若要保護程式碼，可用 **Nuitka**（`build-nuitka.ps1`）：它把你的 `.py` **編譯成原生 C/機器碼**，
發佈的程式內不含可讀的 Python 原始碼。第三方套件（torch、opencv、funasr）本身已是編譯過的
`.pyd`/`.dll`，會原樣複製 — 只有**你自己的程式碼**會被編譯成原生碼。

```text
powershell -ExecutionPolicy Bypass -File build-nuitka.ps1
```

代價：第一次編譯較慢（30–90 分鐘，需編譯 torch 的 Python 層），且需安裝 MSVC Build Tools。
另一個選擇：**PyArmor** 用加密取代編譯（較快，對一般反編譯仍有強防護）。

### 安裝（手動 / 開發者）

```bash
git clone https://github.com/Jyxarthur/shot-by-shot.git
cd shot-by-shot
python -m venv .venv
.venv\Scripts\activate        # Windows  （Linux/macOS：source .venv/bin/activate）
pip install -r requirements.txt
# 桌面版（可選）：
pip install -r requirements-desktop.txt
python scripts\download_sensevoice.py   # 預先下載 SenseVoice 模型
```

### 使用方法

#### 網頁介面

```bash
python app.py
```

在瀏覽器中開啟 http://localhost:5000

**網頁介面選項：**
| 選項 | 說明 |
|------|------|
| **LLM Backend** | Gemini、Qwen、DeepSeek 或 OpenAI-compatible |
| **API Key** | 對應後端的金鑰（自動切換） |
| **啟用轉錄** | SenseVoice 轉錄與對話空隙為基礎的 AD 區間偵測 |
| **Language** | 語言代碼（en, zh 等）或自動偵測 |
| **Character Bank** | 每鏡頭人臉檢測與聚類 |
| **Context Extension** | 用周圍鏡頭上下文擴展口述影像區間 |
| **Video Type** | 電影或電視劇 |
| **Skip Stage 2** | 停用口述影像摘要 |
| **Output Format** | Basic、Full（含描述）或 Original 專案格式 |

#### 命令列

```bash
# 完整流程
python run_processing.py video.mp4 -o output.csv --api-key YOUR_API_KEY

# 跳過部分步驟
python run_processing.py video.mp4 --skip-whisper --api-key YOUR_API_KEY
python run_processing.py video.mp4 --skip-vlm
python run_processing.py video.mp4 --skip-whisper --skip-vlm --skip-stage2
```

#### Kaggle Notebooks

| Notebook | 說明 |
|----------|------|
| `kaggle_notebook.ipynb` | 基礎流程（鏡頭偵測 + VLM + Stage 2） |
| `notebook-ad-2.ipynb` | **角色庫版本** — 人臉檢測、雙後端（Qwen Unsloth / Gemini）、繁體中文提示、可自訂角色名稱 |

### 輸出檔案

| 檔案 | 內容 |
|------|------|
| `shot_by_shot_output.csv` | 主要合併輸出（鏡頭 + 字幕 + 描述） |
| `stage1_descriptions.csv` | 詳細 VLM 描述 |
| `stage2_audio_descriptions.csv` | 簡潔口述影像句子 |
| `character_bank.csv` | 每鏡頭角色分配（若啟用角色庫） |
