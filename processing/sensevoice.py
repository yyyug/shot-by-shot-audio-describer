from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import threading

from .ffmpeg_utils import ensure_ffmpeg_in_path
from .models import Segment, TranscriptionResult


SENSEVOICE_LANGUAGE_OPTIONS: list[dict[str, str]] = [
    {"code": "auto", "name": "Auto detect"},
    {"code": "zh", "name": "Chinese"},
    {"code": "en", "name": "English"},
    {"code": "yue", "name": "Cantonese"},
    {"code": "ja", "name": "Japanese"},
    {"code": "ko", "name": "Korean"},
]


class SenseVoiceError(RuntimeError):
    pass


class SenseVoiceAdapter:
    _runtime_cache: dict[tuple[str, str, str], tuple[Any, str]] = {}
    _runtime_cache_lock = threading.Lock()

    @staticmethod
    def list_languages() -> list[dict[str, str]]:
        return list(SENSEVOICE_LANGUAGE_OPTIONS)

    @staticmethod
    def default_vad_model_path(model_id: str) -> str:
        model_path = Path(model_id)
        bundled_path = (
            model_path.parent / "iic_speech_fsmn_vad_zh-cn-16k-common-pytorch"
            if model_path.is_absolute()
            else Path("iic_speech_fsmn_vad_zh-cn-16k-common-pytorch")
        )
        cached_path = (
            Path.home()
            / ".cache"
            / "modelscope"
            / "hub"
            / "models"
            / "iic"
            / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
        )
        if bundled_path.exists():
            return str(bundled_path)
        if cached_path.exists():
            return str(cached_path)
        return str(bundled_path)

    def __init__(
        self,
        model_id: str,
        model_source: str = "local",
        ffmpeg_path: str = "",
        vad_model_path: str = "",
    ):
        self.model_id = model_id
        self.model_source = model_source
        self.ffmpeg_path = ffmpeg_path
        configured_vad_path = Path(vad_model_path) if vad_model_path else None
        if configured_vad_path and configured_vad_path.exists():
            self.vad_model_path = str(configured_vad_path)
        else:
            self.vad_model_path = self.default_vad_model_path(model_id)
        self._runtime = None
        self._runtime_kind = None

    def _runtime_cache_key(self) -> tuple[str, str, str]:
        return (
            str(Path(self.model_id)),
            self.model_source,
            str(Path(self.vad_model_path)),
        )

    def _load_runtime(self) -> None:
        if self._runtime is not None:
            return

        ffmpeg_status = ensure_ffmpeg_in_path(self.ffmpeg_path)
        if not ffmpeg_status.ready:
            source_hint = {
                "settings": "Configured ffmpeg path does not exist.",
                "env": "Configured ffmpeg environment path does not exist.",
                "missing": "No bundled, configured, or PATH ffmpeg executable was found.",
            }.get(ffmpeg_status.source, "ffmpeg is unavailable.")
            raise SenseVoiceError(
                f"Missing required executable: ffmpeg. {source_hint} "
                "Install ffmpeg, keep the bundled Voxyz ffmpeg files in place, or set a valid ffmpeg executable path in Settings."
            )

        local_model_path = Path(self.model_id)
        if self.model_source != "local":
            raise SenseVoiceError(
                "Voxyz is configured to use bundled local SenseVoice assets only."
            )
        if not local_model_path.exists():
            raise SenseVoiceError(
                f"Bundled SenseVoice model path does not exist: {self.model_id}"
            )
        local_vad_path = Path(self.vad_model_path)
        if not local_vad_path.exists():
            raise SenseVoiceError(
                f"Bundled VAD model path does not exist: {self.vad_model_path}"
            )
        runtime_cache_key = self._runtime_cache_key()
        with self._runtime_cache_lock:
            cached_runtime = self._runtime_cache.get(runtime_cache_key)
        if cached_runtime is not None:
            self._runtime, self._runtime_kind = cached_runtime
            return

        try:
            from funasr import AutoModel  # type: ignore
            from funasr.utils.postprocess_utils import (  # type: ignore
                rich_transcription_postprocess,
            )

            model = AutoModel(
                model=self.model_id,
                trust_remote_code=True,
                vad_model=str(local_vad_path),
                vad_kwargs={"max_single_segment_time": 30_000},
            )
            self._runtime = (model, rich_transcription_postprocess)
            self._runtime_kind = "funasr"
            with self._runtime_cache_lock:
                self._runtime_cache[runtime_cache_key] = (
                    self._runtime,
                    self._runtime_kind,
                )
            return
        except Exception:
            pass

        try:
            from funasr_onnx import SenseVoiceSmall  # type: ignore
            from funasr_onnx.utils.postprocess_utils import (  # type: ignore
                rich_transcription_postprocess,
            )

            model = SenseVoiceSmall(self.model_id, batch_size=10, quantize=True)
            self._runtime = (model, rich_transcription_postprocess)
            self._runtime_kind = "funasr_onnx"
            with self._runtime_cache_lock:
                self._runtime_cache[runtime_cache_key] = (
                    self._runtime,
                    self._runtime_kind,
                )
            return
        except Exception as exc:
            raise SenseVoiceError(
                "SenseVoice runtime is unavailable. Install `funasr` or "
                "`funasr-onnx` and ensure the SenseVoice model can be resolved."
            ) from exc

    def warmup(self) -> None:
        self._load_runtime()

    @staticmethod
    def get_runtime_status() -> dict[str, Any]:
        status = {
            "funasr": False,
            "funasr_onnx": False,
            "available": False,
        }
        try:
            __import__("funasr")
            status["funasr"] = True
        except Exception:
            pass

        try:
            __import__("funasr_onnx")
            status["funasr_onnx"] = True
        except Exception:
            pass

        status["available"] = bool(status["funasr"] or status["funasr_onnx"])
        return status

    @staticmethod
    def validate_model_config(model_id: str, model_source: str, ffmpeg_path: str = "") -> dict[str, Any]:
        runtime = SenseVoiceAdapter.get_runtime_status()
        messages: list[str] = []
        valid = runtime["available"]
        local_model_path = Path(model_id)
        local_vad_path = Path(SenseVoiceAdapter.default_vad_model_path(model_id))
        ffmpeg_status = ensure_ffmpeg_in_path(ffmpeg_path)
        ffmpeg_available = ffmpeg_status.ready

        if not runtime["available"]:
            messages.append("No SenseVoice runtime found. Install `funasr` or `funasr-onnx`.")
        if not ffmpeg_available:
            valid = False
            if ffmpeg_status.source == "settings":
                messages.append("Configured ffmpeg path does not exist.")
            elif ffmpeg_status.source == "env":
                messages.append("Configured ffmpeg environment path does not exist.")
            else:
                messages.append("`ffmpeg` was not found. SenseVoice audio loading on Windows requires ffmpeg.")
        else:
            messages.append(f"`ffmpeg` is available at {ffmpeg_status.path}.")
            messages.append(f"`ffmpeg` source: {ffmpeg_status.source}.")
            messages.append(
                "`ffmpeg` bin directory is on runtime PATH."
                if ffmpeg_status.path_in_environment
                else "`ffmpeg` bin directory is not on runtime PATH."
            )

        if model_source != "local":
            valid = False
            messages.append("Voxyz is configured to use bundled local SenseVoice assets only.")
        if not local_model_path.exists():
            valid = False
            messages.append("Bundled local SenseVoice model path does not exist.")
        else:
            messages.append("Bundled local SenseVoice model path exists.")
        if not local_vad_path.exists():
            valid = False
            messages.append("Bundled local VAD model path does not exist.")
        else:
            messages.append("Bundled local VAD model path exists.")

        if runtime["funasr"]:
            messages.append("`funasr` runtime is available.")
        if runtime["funasr_onnx"]:
            messages.append("`funasr_onnx` runtime is available.")

        return {
            "valid": valid,
            "messages": messages,
            "runtime": runtime,
            "model_id": model_id,
            "model_source": model_source,
            "ffmpeg_path": ffmpeg_status.path,
            "ffmpeg_source": ffmpeg_status.source,
            "ffmpeg_ready": ffmpeg_status.ready,
        }

    def transcribe_file(
        self,
        file_path: str,
        *,
        language: str,
        use_itn: bool,
        vad_enabled: bool,
        merge_vad: bool,
    ) -> TranscriptionResult:
        self._load_runtime()
        path = Path(file_path)
        if not path.exists():
            raise SenseVoiceError(f"Audio file does not exist: {file_path}")
        ffmpeg_status = ensure_ffmpeg_in_path(self.ffmpeg_path)
        if not ffmpeg_status.ready:
            raise SenseVoiceError(
                "Missing required executable: ffmpeg. Install ffmpeg and ensure it is available on PATH, "
                "keep the bundled Voxyz ffmpeg files in place, or set the ffmpeg executable path in Settings, then retry transcription."
            )

        model, postprocess = self._runtime

        if self._runtime_kind == "funasr":
            try:
                result = model.generate(
                    input=str(path),
                    cache={},
                    language=language,
                    use_itn=use_itn,
                    batch_size_s=60,
                    merge_vad=merge_vad if vad_enabled else False,
                    merge_length_s=15,
                )
            except FileNotFoundError as exc:
                missing_name = getattr(exc, "filename", None) or "ffmpeg"
                raise SenseVoiceError(
                    f"Missing required executable: {missing_name}. Install ffmpeg and ensure it is available on PATH, then retry transcription."
                ) from exc
            payload = result[0]
            text = self._sanitize_transcript_text(postprocess(payload["text"]))
            segments = self._parse_segments(payload)
            return TranscriptionResult(
                text=text,
                segments=segments,
                language=payload.get("text_language", language),
                emotion=None,
                event=None,
                raw=payload,
            )

        if self._runtime_kind == "funasr_onnx":
            try:
                results = model([str(path)], language=language, use_itn=use_itn)
            except FileNotFoundError as exc:
                missing_name = getattr(exc, "filename", None) or "ffmpeg"
                raise SenseVoiceError(
                    f"Missing required executable: {missing_name}. Install ffmpeg and ensure it is available on PATH, then retry transcription."
                ) from exc
            text = self._sanitize_transcript_text(postprocess(results[0]))
            return TranscriptionResult(
                text=text,
                segments=[],
                language=language,
                raw={"text": results[0]},
            )

        raise SenseVoiceError("Unsupported SenseVoice runtime.")

    def _parse_segments(self, payload: dict[str, Any]) -> list[Segment]:
        raw_segments = payload.get("timestamp") or payload.get("segments") or []
        segments: list[Segment] = []
        for entry in raw_segments:
            if isinstance(entry, dict):
                start = float(entry.get("start", 0.0))
                end = float(entry.get("end", start))
                text = self._sanitize_transcript_text(str(entry.get("text", "")).strip())
                speaker = entry.get("speaker") or entry.get("spk")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                start = float(entry[0])
                end = float(entry[1])
                text = self._sanitize_transcript_text(str(entry[2]).strip())
                speaker = entry[3] if len(entry) >= 4 else None
            else:
                continue
            if text:
                segments.append(
                    Segment(
                        start=start,
                        end=end,
                        text=text,
                        speaker=str(speaker).strip() if speaker else None,
                    )
                )
        return segments

    @staticmethod
    def _sanitize_transcript_text(text: str) -> str:
        cleaned = re.sub(r"\{\\[^}]*\}", "", text)
        cleaned = cleaned.replace("\\N", "\n").replace("\\n", "\n")
        cleaned = re.sub(r"[😊😡😔🎼😀👏]", "", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
