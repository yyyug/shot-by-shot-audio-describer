import json
import os
import sys
import shutil
import tempfile
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

DEFAULT_VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
DEFAULT_ONNX_MODEL = "iic/SenseVoiceSmall-onnx"
_MODEL_DIRS_JSON = "model_dirs.json"


def _find_local_models_dir():
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
    local = _find_local_models_dir()
    if local:
        json_path = os.path.join(local, _MODEL_DIRS_JSON)
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                onnx_dir = info.get("onnx_dir")
                vad_model = info.get("vad_model") or DEFAULT_VAD_MODEL
                for key in ("onnx_dir", "vad_model"):
                    path = info.get(key)
                    if path and not os.path.isdir(path) and os.path.isdir(os.path.join(local, path)):
                        info[key] = os.path.join(local, path)
                onnx_dir = info.get("onnx_dir")
                vad_model = info.get("vad_model") or DEFAULT_VAD_MODEL
                if onnx_dir and os.path.isdir(onnx_dir):
                    return onnx_dir, vad_model, "onnx"
            except Exception:
                pass
    # Legacy fallback (torch)
    if local:
        json_path = os.path.join(local, _MODEL_DIRS_JSON)
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                model_dir = info.get("model_dir")
                vad_model = info.get("vad_model") or DEFAULT_VAD_MODEL
                for key in ("model_dir", "vad_model"):
                    path = info.get(key)
                    if path and not os.path.isdir(path) and os.path.isdir(os.path.join(local, path)):
                        info[key] = os.path.join(local, path)
                model_dir = info.get("model_dir")
                if model_dir and os.path.isdir(model_dir):
                    return model_dir, vad_model, "torch"
            except Exception:
                pass
    model_dir = os.environ.get("SENSEVOICE_MODEL_DIR")
    onnx_dir = os.environ.get("SENSEVOICE_ONNX_DIR")
    vad_model = os.environ.get("SENSEVOICE_VAD_MODEL", DEFAULT_VAD_MODEL)
    if onnx_dir and os.path.isdir(onnx_dir):
        return onnx_dir, vad_model, "onnx"
    if model_dir and os.path.isdir(model_dir):
        return model_dir, vad_model, "torch"
    return None, vad_model, None


def _audio_duration(video_path):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count / fps if fps > 0 else 0.0
    except Exception:
        return 0.0


def _read_wav_float32(path):
    """Read a mono WAV into float32 samples in [-1, 1] using only stdlib
    (libsndfile/soundfile is unreliable inside the PyInstaller bundle)."""
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError(f"expected mono wav, got {w.getnchannels()} channels")
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data, sr


def _write_wav_float32(path, samples, sr):
    """Write mono int16 WAV from float32 samples; stdlib only."""
    import wave
    import numpy as np
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _ffmpeg_bin():
    """Resolve ffmpeg: bundled copy (frozen app) first, then PATH."""
    import shutil
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        for cand in (
            os.path.join(meipass, "tools", "ffmpeg", "ffmpeg.exe"),
            os.path.join(meipass, "ffmpeg", "ffmpeg.exe"),
        ):
            if os.path.isfile(cand):
                return cand
    return shutil.which("ffmpeg")


def _extract_audio(video_path):
    import subprocess
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. The portable app bundles it - if you are "
            "running from source, install ffmpeg and put it on PATH."
        )
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path, "-ar", "16000", "-ac", "1", tmp_path,
        ], check=True)
        return _read_wav_float32(tmp_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio: {exc}")
    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _load_vad_segments(audio, sr, vad_model, callback=None):
    try:
        from funasr_onnx.vad_bin import Fsmn_vad
    except ImportError as e:
        raise RuntimeError(
            f"funasr-onnx VAD backend not available ({type(e).__name__}: {e}). "
            "Run: pip install funasr-onnx"
        ) from e
    if callback:
        callback(0.15, "Loading VAD model...")
    vad = Fsmn_vad(model_dir=vad_model, device_id="-1", quantize=True)
    if callback:
        callback(0.35, "Detecting speech segments...")
    raw = vad(audio)
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], list):
        return []
    return [(float(s), float(e)) for s, e in raw[0]]


def _strip_sensevoice_tags(text):
    """Remove SenseVoice system tags from output text."""
    import re
    text = re.sub(r"<\|[^\|]+\|>", "", text)
    return text.strip()


def transcribe_video(
    video_path: str,
    model_dir: str = None,
    device: str = None,
    language: str = "auto",
    use_itn: bool = True,
    merge_length_s: float = 15.0,
    batch_size_s: float = 60.0,
    callback=None,
) -> list:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    resolved_model, resolved_vad, mode = _resolve_model_dirs()
    if model_dir is None:
        model_dir = resolved_model
    vad_model = resolved_vad or DEFAULT_VAD_MODEL

    if mode == "onnx" and model_dir:
        return _transcribe_video_onnx(video_path, model_dir, vad_model, language, use_itn, callback)
    if model_dir:
        return _transcribe_video_torch(video_path, model_dir, vad_model, language, use_itn, merge_length_s, batch_size_s, callback)
    raise RuntimeError("No transcription model available.")


def _transcribe_video_onnx(video_path, onnx_dir, vad_model, language, use_itn, callback):
    try:
        from funasr_onnx import SenseVoiceSmall
        from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError as e:
        raise RuntimeError(
            f"funasr-onnx not available ({type(e).__name__}: {e}). "
            "Run: pip install funasr-onnx"
        ) from e

    audio, sr = _extract_audio(video_path)
    vad_segments = _load_vad_segments(audio, sr, vad_model, callback)
    if not vad_segments:
        return []

    if callback:
        callback(0.5, "Running ONNX SenseVoice...")

    model = SenseVoiceSmall(onnx_dir, batch_size=1, quantize=True)
    segments = []
    tmp_dir = tempfile.mkdtemp(prefix="sbs_onnx_")
    try:
        for i, (s_ms, e_ms) in enumerate(vad_segments):
            s = int(s_ms / 1000 * sr)
            e = int(e_ms / 1000 * sr)
            chunk = audio[s:e]
            if len(chunk) < 800:
                continue
            tmp_wav = os.path.join(tmp_dir, f"{i}.wav")
            _write_wav_float32(tmp_wav, chunk, sr)
            text_norm = "withitn" if use_itn else "woitn"
            res = model([tmp_wav], language=language, textnorm=text_norm)
            text = res[0] if res else ""
            text = _strip_sensevoice_tags(text)
            try:
                text = rich_transcription_postprocess(text)
            except Exception:
                pass
            if text:
                segments.append({
                    "text": text,
                    "start_time": round(s_ms / 1000.0, 3),
                    "end_time": round(e_ms / 1000.0, 3),
                })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if callback:
        callback(1.0, f"Transcribed {len(segments)} segments")
    return segments


def _transcribe_video_torch(video_path, model_dir, vad_model, language, use_itn, merge_length_s, batch_size_s, callback):
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
        device=os.environ.get("SENSEVOICE_DEVICE", "cpu"),
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
    segments = _parse_torch_result(res, video_path)
    if callback:
        callback(1.0, f"Transcribed {len(segments)} segments")
    return segments


def _parse_torch_result(res, video_path=None):
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
    text = rich_transcription_postprocess(first.get("text", ""))
    if not text:
        return []
    duration = _audio_duration(video_path) if video_path else 0.0
    segments.append({"text": text, "start_time": 0.0, "end_time": round(duration, 3)})
    return segments
