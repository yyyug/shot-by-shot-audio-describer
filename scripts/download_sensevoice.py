"""
Pre-download SenseVoice models so users don't need to download on first run.

Downloads:
  - iic/SenseVoiceSmall                     (ASR model)
  - iic/speech_fsmn_vad_zh-cn-16k-common-pytorch  (FSMN-VAD voice activity detection)

Models are cached inside <repo root>/models and a model_dirs.json manifest is
written so processing/sensevoice_transcriber.py loads them without network.

Usage:
    python scripts/download_sensevoice.py [--model-dir PATH] [--vad-dir PATH] [--force]
"""
import argparse
import json
import os
import sys
import tempfile
import wave


APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(APP_ROOT, "models", "sensevoice")
MANIFEST = os.path.join(MODELS_DIR, "model_dirs.json")

ASR_MODEL_ID = "iic/SenseVoiceSmall"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


def log(msg):
    print(f"[download_sensevoice] {msg}")


def snapshot_download(model_id, cache_dir, force):
    """Download a ModelScope model into cache_dir. Returns local dir path."""
    log(f"Downloading {model_id} ...")
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise RuntimeError("modelscope not installed. Run: pip install 'funasr>=1.3.29'")

    kwargs = {"cache_dir": cache_dir}
    if force:
        kwargs["force_download"] = True
    path = snapshot_download(model_id, **kwargs)
    log(f"  -> {path}")
    return path


def smoke_test(model_dir, vad_dir):
    """Run a tiny inference on a 1-second silent wav to verify the models load."""
    log("Running smoke test (1s silent audio)...")
    import numpy as np

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
        tmp.close()

        from funasr import AutoModel

        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            vad_model=vad_dir,
            vad_kwargs={"max_single_segment_time": 30000},
            device="cpu",
        )
        res = model.generate(input=tmp.name, cache={}, language="auto", use_itn=True)
        log(f"Smoke test OK (result: {str(res)[:120]})")
    finally:
        os.unlink(tmp.name)


def main():
    parser = argparse.ArgumentParser(description="Pre-download SenseVoice models")
    parser.add_argument("--model-dir", default=ASR_MODEL_ID, help="ASR model id or local path")
    parser.add_argument("--vad-dir", default=VAD_MODEL_ID, help="VAD model id or local path")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip inference check")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_dir = args.model_dir if os.path.isdir(args.model_dir) else snapshot_download(args.model_dir, MODELS_DIR, args.force)
    vad_dir = args.vad_dir if os.path.isdir(args.vad_dir) else snapshot_download(args.vad_dir, MODELS_DIR, args.force)

    # Store paths relative to the models folder so the bundle is portable
    # across machines (the whole folder can be shipped to end users).
    def to_relative(path):
        try:
            rel = os.path.relpath(path, MODELS_DIR)
            return rel if not rel.startswith("..") else path
        except ValueError:
            return path

    manifest = {"model_dir": to_relative(model_dir), "vad_model": to_relative(vad_dir)}
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"Wrote manifest: {MANIFEST}")

    if not args.skip_smoke_test:
        try:
            smoke_test(model_dir, vad_dir)
        except Exception as e:
            log(f"Smoke test FAILED: {e}")
            sys.exit(1)

    log("Done. SenseVoice models are ready for offline use.")
    print(MANIFEST)


if __name__ == "__main__":
    main()
