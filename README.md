# Shot-by-Shot: 影片逐鏡頭分析工具 / Video Shot-by-Shot Audio Describer

[English](#english) | [中文](#中文)

---

## English

### Overview

This tool processes video files to generate shot-by-shot analysis with:
- Shot boundary detection (PySceneDetect)
- Audio transcription + dialogue gap detection (Faster-Whisper / whisper.cpp)
- **Character bank** — face detection & clustering across shots
- **Context extension** — past/future shot context for AD placement
- AI-powered video descriptions — **Multi-backend**: Gemini 2.5 Flash, OpenRouter (Qwen 2.5 VL 7B), or local Qwen (Unsloth bnb-4bit)
- Audio description summarization (Stage 2 LLM)

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- whisper.cpp (for transcription, optional)

**Backend-specific:**
- **OpenRouter**: API key
- **Gemini**: API key + `pip install google-generativeai`
- **Local Qwen**: GPU (T4 x2 recommended) + `pip install unsloth bitsandbytes`

### Installation

```bash
git clone https://github.com/Jyxarthur/shot-by-shot.git
cd shot-by-shot
pip install -r requirements.txt

# Optional backends:
pip install google-generativeai          # Gemini
pip install unsloth unsloth_zoo bitsandbytes  # Local Qwen
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
| **LLM Backend** | OpenRouter or Gemini |
| **API Key** | OpenRouter or Gemini key (auto-toggles) |
| **Enable Whisper** | Transcription and dialogue gap detection |
| **Language** | Language code (en, zh, etc.) or auto-detect |
| **Character Bank** | Face detection + clustering per shot |
| **Context Extension** | Extend AD intervals with surrounding shot context |
| **Video Type** | Movie or TV Series |
| **Skip Stage 2** | Disable AD summarization |
| **Output Format** | Basic, Full (with descriptions), or Original project format |

#### Command Line

```bash
# Full pipeline
python run_processing.py video.mp4 -o output.csv --openrouter-key YOUR_API_KEY

# Skip steps
python run_processing.py video.mp4 --skip-whisper --openrouter-key YOUR_API_KEY
python run_processing.py video.mp4 --skip-vlm
python run_processing.py video.mp4 --skip-whisper --skip-vlm --skip-stage2
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
├── processing/whisper_transcriber.py   ✅ Wired
├── processing/character_recognizer.py  ✅ Wired (web + notebook)
├── processing/context_extender.py      ✅ Wired (web)
├── processing/film_grammar.py          ✅ Wired (prompt variants)
├── processing/vlm_describer.py         ✅ Wired (OpenRouter)
├── processing/llm_summarizer.py        ✅ Wired (OpenRouter)
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
- 音訊轉錄及對話空隙偵測（Faster-Whisper / whisper.cpp）
- **角色庫** — 全鏡頭人臉檢測與聚類
- **上下文擴展** — 前後鏡頭上下文用於口述影像配置
- AI 影片描述 — **多後端**：Gemini 2.5 Flash、OpenRouter (Qwen 2.5 VL 7B)、或本地 Qwen (Unsloth bnb-4bit)
- 口述影像摘要（Stage 2 LLM）

### 系統要求

- Python 3.8+
- FFmpeg（用於影片處理）
- whisper.cpp（用於轉錄，可選）

**各後端要求：**
- **OpenRouter**：API 金鑰
- **Gemini**：API 金鑰 + `pip install google-generativeai`
- **本地 Qwen**：GPU（建議 T4 x2）+ `pip install unsloth bitsandbytes`

### 安裝

```bash
git clone https://github.com/Jyxarthur/shot-by-shot.git
cd shot-by-shot
pip install -r requirements.txt

# 可選後端：
pip install google-generativeai              # Gemini
pip install unsloth unsloth_zoo bitsandbytes  # 本地 Qwen
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
| **LLM Backend** | OpenRouter 或 Gemini |
| **API Key** | OpenRouter 或 Gemini 金鑰（自動切換） |
| **Enable Whisper** | 轉錄和對話空隙偵測 |
| **Language** | 語言代碼（en, zh 等）或自動偵測 |
| **Character Bank** | 每鏡頭人臉檢測與聚類 |
| **Context Extension** | 用周圍鏡頭上下文擴展口述影像區間 |
| **Video Type** | 電影或電視劇 |
| **Skip Stage 2** | 停用口述影像摘要 |
| **Output Format** | Basic、Full（含描述）或 Original 專案格式 |

#### 命令列

```bash
# 完整流程
python run_processing.py video.mp4 -o output.csv --openrouter-key YOUR_API_KEY

# 跳過部分步驟
python run_processing.py video.mp4 --skip-whisper --openrouter-key YOUR_API_KEY
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
