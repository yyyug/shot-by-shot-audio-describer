"""
SenseVoice (FunASR) transcription module.

Uses the QwenAudio/SenseVoice model (SenseVoiceSmall) through FunASR with
FSMN-VAD for voice activity detection. Returns segments in the same format
as the previous whisper.cpp transcriber:

    [{"text": str, "start_time": float_seconds, "end_time": float_seconds}, ...]

Time output is converted from milliseconds to float seconds so it matches
the rest of the pipeline (shot detection, dialogue gap detection, CSVs).
"""
import os
import sys
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "iic/SenseVoiceSmall"
DEFAULT_VAD_MODEL = "fsmn-vad"
_MODEL_DIRS_JSON = "model_dirs.json"


def _find_local_models_dir():
    """Return the local SenseVoice models folder, or None if not found.

    Looks under <root>/models/sensevoice in source trees and in PyInstaller-
    frozen builds (models are shipped next to the executable).
    """
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        meipass = getattr(sys, "_MEIPASS", "")
        candidates = [
            os.path.join(exe_dir, "models", "sensevoice"),
            os.path.join(meipass, "models", "sensevoice"),
            os.path.join(exe_dir, "models"),
            os.path.join(meipass, "models"),
        ]
    else:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(package_dir)
        candidates = [
            os.path.join(repo_root, "models", "sensevoice"),
            os.path.join(repo_root, "models"),
            os.path.join(os.getcwd(), "models", "sensevoice"),
        ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _resolve_model_dirs():
    """Resolve (model_dir, vad_model) from local pre-download, env, or defaults.

    Priority:
      1. <repo>/models/model_dirs.json  (created by scripts/download_sensevoice.py)
      2. SENSEVOICE_MODEL_DIR env var for the ASR model
      3. Built-in defaults (downloaded on first run via FunASR/ModelScope)
    """
    local = _find_local_models_dir()
    if local:
        json_path = os.path.join(local, _MODEL_DIRS_JSON)
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                model_dir = info.get("model_dir") or DEFAULT_MODEL_DIR
                vad_model = info.get("vad_model") or DEFAULT_VAD_MODEL
                # Manifest paths are stored relative to the models/ folder so the
                # whole bundle is portable across machines.
                for key in ("model_dir", "vad_model"):
                    path = info.get(key)
                    if path and not os.path.isdir(path) and os.path.isdir(os.path.join(local, path)):
                        info[key] = os.path.join(local, path)
                model_dir = info.get("model_dir") or DEFAULT_MODEL_DIR
                vad_model = info.get("vad_model") or DEFAULT_VAD_MODEL
                if os.path.isdir(model_dir):
                    logger.info("Using pre-downloaded SenseVoice models: %s", model_dir)
                    return model_dir, vad_model
            except Exception:
                pass
    model_dir = os.environ.get("SENSEVOICE_MODEL_DIR", DEFAULT_MODEL_DIR)
    vad_model = os.environ.get("SENSEVOICE_VAD_MODEL", DEFAULT_VAD_MODEL)
    return model_dir, vad_model


def transcribe_video(
    video_path: str,
    model_dir: str = None,
    device: str = None,
    language: str = "auto",
    use_itn: bool = True,
    merge_length_s: float = 15.0,
    batch_size_s: float = 60.0,
    callback=None
) -> list:
    """
    Transcribe video audio using SenseVoice (FunASR).

    Args:
        video_path: Path to the video file
        model_dir: SenseVoice model name/path (default from pre-download or env)
        device: "cpu", "cuda:0", ... (default from SENSEVOICE_DEVICE env, else "cpu")
        language: "auto", "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn: Include punctuation and inverse text normalization
        merge_length_s: Merge VAD segments up to this length (seconds)
        batch_size_s: Dynamic batching duration (seconds)
        callback: Optional callback function(progress_float, message_str)

    Returns:
        List of dicts with keys: text, start_time, end_time (float seconds)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    resolved_model_dir, resolved_vad_model = _resolve_model_dirs()
    model_dir = model_dir or resolved_model_dir
    vad_model = resolved_vad_model
    device = device or os.environ.get("SENSEVOICE_DEVICE", "cpu")

    if callback:
        callback(0.1, "Loading SenseVoice model...")

    try:
        from funasr import AutoModel
    except ImportError:
        raise RuntimeError("funasr not installed. Run: pip install 'funasr>=1.3.29'")

    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        vad_model=vad_model,
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        quantize=os.environ.get("SENSEVOICE_QUANTIZE", "1") not in ("0", "false", "False"),
    )

    if callback:
        callback(0.4, "Transcribing audio...")

    res = model.generate(
        input=video_path,
        cache={},
        language=language,
        use_itn=use_itn,
        batch_size_s=batch_size_s,
        merge_vad=True,
        merge_length_s=merge_length_s,
    )

    segments = _parse_result(res, video_path)

    if callback:
        callback(1.0, f"Transcribed {len(segments)} segments")

    return segments


def _parse_result(res, video_path=None):
    """Convert FunASR SenseVoice output into standard segments (seconds)."""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError:
        rich_transcription_postprocess = lambda text: text

    if not res or not isinstance(res, list) or len(res) == 0:
        return []

    first = res[0]
    if not isinstance(first, dict):
        return []

    segments = []

    # Per-sentence timestamps (milliseconds) via sentence_info (FunASR >= 1.3.29)
    sentence_info = first.get("sentence_info") or []
    for sent in sentence_info:
        start_ms = sent.get("start")
        end_ms = sent.get("end")
        if start_ms is None or end_ms is None:
            continue
        text = rich_transcription_postprocess(sent.get("text", ""))
        segments.append({
            "text": text,
            "start_time": round(float(start_ms) / 1000.0, 3),
            "end_time": round(float(end_ms) / 1000.0, 3),
        })

    if segments:
        return segments

    # Fallback: single segment spanning the whole input
    text = rich_transcription_postprocess(first.get("text", ""))
    if not text:
        return []
    duration = _video_duration(video_path) if video_path else 0.0
    segments.append({"text": text, "start_time": 0.0, "end_time": round(duration, 3)})
    return segments


def _video_duration(video_path):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count / fps if fps > 0 else 0.0
    except Exception:
        return 0.0
