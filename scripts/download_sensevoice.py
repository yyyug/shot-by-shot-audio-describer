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


def to_fp16_inplace(model_dir):
    """Halve the on-disk model.pt by converting its weights to float16.

    SenseVoiceSmall is ~900MB in fp32, which dominates the packaged app size.
    Converting the stored state dict to fp16 shrinks it to ~450MB; FunASR
    loads it transparently (and the transcriber can still apply int8 PTQ on
    top at load time via quantize=True).

    A .fp32.bak backup is kept next to model.pt until the end of the script;
    callers MUST remove it before packaging, otherwise the backup is bundled.
    """
    import os

    model_pt = os.path.join(model_dir, "model.pt")
    backup_pt = model_pt + ".fp32.bak"
    if not os.path.exists(model_pt):
        return False

    import torch

    sd = torch.load(model_pt, map_location="cpu")
    is_already_fp16 = all(
        (not (torch.is_tensor(v) and v.dtype.is_floating_point))
        or (torch.is_tensor(v) and v.dtype == torch.float16)
        for v in sd.values() if torch.is_tensor(v)
    ) if isinstance(sd, dict) else False
    if is_already_fp16:
        log("model.pt is already fp16; skipping conversion")
        return True

    if not os.path.exists(backup_pt):
        import shutil
        shutil.copy(model_pt, backup_pt)
    log(f"Converting {model_pt} to fp16 (backup at {os.path.basename(backup_pt)})...")

    def walk(obj):
        if torch.is_tensor(obj):
            return obj.half() if obj.dtype.is_floating_point else obj
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj

    torch.save(walk(sd), model_pt)
    log(f"Converted -> {os.path.getsize(model_pt) / 1e6:.1f} MB")
    return True


def smoke_test(model_dir, vad_dir, quantize=True):
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
            quantize=quantize,
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
    parser.add_argument("--no-fp16", action="store_true", help="Keep the ASR weights in fp32 (larger package)")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_dir = args.model_dir if os.path.isdir(args.model_dir) else snapshot_download(args.model_dir, MODELS_DIR, args.force)
    vad_dir = args.vad_dir if os.path.isdir(args.vad_dir) else snapshot_download(args.vad_dir, MODELS_DIR, args.force)

    # Halve the packaged size by storing the ASR model in fp16 (optional; the
    # transcriber applies int8 PTQ at load time regardless of stored dtype).
    if not args.no_fp16:
        to_fp16_inplace(model_dir)
    # Never ship the fp32 backup: removing it keeps the bundle lean.
    backup_pt = os.path.join(model_dir, "model.pt.fp32.bak")
    if os.path.exists(backup_pt):
        os.remove(backup_pt)
        log("Removed fp32 backup from the model dir")

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
