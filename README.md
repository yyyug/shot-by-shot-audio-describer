# Shot-by-Shot: 影片逐鏡頭分析工具 / Video Shot-by-Shot Analysis Tool

[English](#english) | [中文](#中文)

---

## English

### Overview

This tool processes video files to generate shot-by-shot analysis with:
- Shot boundary detection
- Audio transcription (subtitles)
- Optional AI-powered video descriptions

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- whisper.cpp (for transcription, optional)
- OpenRouter API key (for AI descriptions, optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/Jyxarthur/shot-by-shot.git
cd shot-by-shot

# Install Python dependencies
pip install -r requirements.txt

# Install whisper.cpp (optional)
# Follow instructions at: https://github.com/ggerganov/whisper.cpp
```

### Usage

#### Web Interface

```bash
python app.py
```

Open http://localhost:5000 in your browser.

#### Command Line

```bash
# Basic usage (shot detection + transcription)
python run_processing.py video.mp4 -o output.csv

# With VLM descriptions
python run_processing.py video.mp4 -o output.csv --openrouter-key YOUR_API_KEY

# Specify whisper.cpp path
python run_processing.py video.mp4 -o output.csv --whisper-path /path/to/whisper-cpp

# Full options
python run_processing.py video.mp4 \
    -o output.csv \
    --whisper-path /path/to/whisper-cpp \
    --model-path models/ggml-medium.bin \
    --language en \
    --openrouter-key YOUR_API_KEY \
    --format full
```

#### Kaggle

1. Upload `kaggle_notebook.ipynb` to Kaggle
2. Add your video as a dataset input
3. Set `OPENROUTER_API_KEY` in Cell 3
4. Run all cells
5. Download output CSV

### Output Formats

| Format | Columns |
|--------|---------|
| basic | shot_id, start_time, end_time, subtitle |
| full | shot_id, start_time, end_time, subtitle, video_description |
| original | anno_idx, imdbid, start, end, text_gen |

### Configuration

| Option | Description | Default |
|--------|-------------|---------|
| --whisper-path | Path to whisper-cpp binary | whisper-cpp |
| --model-path | Path to whisper model file | None (uses default) |
| --language | Language code (en, zh, etc.) | None (auto-detect) |
| --openrouter-key | OpenRouter API key | None (skip VLM) |
| --format | Output format | basic |

---

## 中文

### 概述

此工具用於處理影片檔案，生成逐鏡頭分析，包括：
- 鏡頭邊界檢測
- 音訊轉錄（字幕）
- 可選的 AI 影片描述

### 系統要求

- Python 3.8+
- FFmpeg（用於影片處理）
- whisper.cpp（用於轉錄，可選）
- OpenRouter API 金鑰（用於 AI 描述，可選）

### 安裝

```bash
# 複製儲存庫
git clone https://github.com/Jyxarthur/shot-by-shot.git
cd shot-by-shot

# 安裝 Python 依賴
pip install -r requirements.txt

# 安裝 whisper.cpp（可選）
# 請參考：https://github.com/ggerganov/whisper.cpp
```

### 使用方法

#### 網頁介面

```bash
python app.py
```

在瀏覽器中開啟 http://localhost:5000

#### 命令列

```bash
# 基本用法（鏡頭檢測 + 轉錄）
python run_processing.py video.mp4 -o output.csv

# 加入 AI 描述
python run_processing.py video.mp4 -o output.csv --openrouter-key YOUR_API_KEY

# 指定 whisper.cpp 路徑
python run_processing.py video.mp4 -o output.csv --whisper-path /path/to/whisper-cpp

# 完整選項
python run_processing.py video.mp4 \
    -o output.csv \
    --whisper-path /path/to/whisper-cpp \
    --model-path models/ggml-medium.bin \
    --language en \
    --openrouter-key YOUR_API_KEY \
    --format full
```

#### Kaggle

1. 將 `kaggle_notebook.ipynb` 上傳到 Kaggle
2. 將您的影片作為資料集輸入
3. 在第 3 格設定 `OPENROUTER_API_KEY`
4. 執行所有儲存格
5. 下載輸出 CSV

### 輸出格式

| 格式 | 欄位 |
|------|------|
| basic | shot_id, start_time, end_time, subtitle |
| full | shot_id, start_time, end_time, subtitle, video_description |
| original | anno_idx, imdbid, start, end, text_gen |

### 設定選項

| 選項 | 說明 | 預設值 |
|------|------|--------|
| --whisper-path | whisper-cpp 執行檔路徑 | whisper-cpp |
| --model-path | whisper 模型檔案路徑 | None（使用預設） |
| --language | 語言代碼（en, zh 等） | None（自動偵測） |
| --openrouter-key | OpenRouter API 金鑰 | None（跳過 AI 描述） |
| --format | 輸出格式 | basic |
