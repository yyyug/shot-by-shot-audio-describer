"""
Pre-download SenseVoice models so users don't need to download on first run.

ONNX mode (default) downloads:
  - iic/SenseVoiceSmall-onnx                    (INT8 quantized ONNX ASR model)
  - iic/speech_fsmn_vad_zh-cn-16k-common-pytorch (FSMN-VAD voice activity detection,
                                                 used in torch to segment speech)
  - iic/SenseVoiceSmall (torch snapshot, only to borrow chn_jpn_yue_eng_ko_spectok
                                                 .bpe.model required by funasr-onnx)

The ONNX ASR weights are ~230MB versus ~900MB for the torch fp32 model, so the
packaged app is much smaller while retaining per-segment timestamps (VAD boundaries).

Models are cached inside <repo root>/models and a model_dirs.json manifest is
written so processing/sensevoice_transcriber.py loads them without network.

Usage:
    python scripts/download_sensevoice.py [--onnx|--torch] [--model-dir PATH] [--vad-dir PATH] [--force]
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
ASR_ONNX_MODEL_ID = "iic/SenseVoiceSmall-onnx"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
BPE_MODEL_FILE = "chn_jpn_yue_eng_ko_spectok.bpe.model"


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


def to_relative(path):
    rel = os.path.relpath(path, MODELS_DIR)
    return rel if not rel.startswith("..") else path


def export_vad_onnx(vad_dir):
    """Ensure a quantized ONNX VAD model exists for funasr_onnx.Fsmn_vad.

    funasr_onnx loads the FSMN-VAD through ``model_quant.onnx``; the
    downloaded snapshot only carries the torch checkpoint, so we export it
    once (offline, using the legacy torch.onnx exporter). Idempotent.
    """
    model_quant = os.path.join(vad_dir, "model_quant.onnx")
    if os.path.exists(model_quant):
        log(f"VAD ONNX already present: {model_quant}")
        return
    if not os.path.exists(os.path.join(vad_dir, "config.yaml")) or not os.path.exists(
        os.path.join(vad_dir, "am.mvn")
    ):
        raise RuntimeError(f"VAD model dir missing config.yaml/am.mvn: {vad_dir}")

    import torch

    _orig_export = torch.onnx.export

    def _patched_export(*args, **kwargs):
        kwargs["dynamo"] = False
        return _orig_export(*args, **kwargs)

    torch.onnx.export = _patched_export

    from funasr_onnx.vad_bin import Fsmn_vad

    log(f"Exporting quantized VAD ONNX into {vad_dir} ...")
    Fsmn_vad(model_dir=vad_dir, device_id="-1", quantize=True)
    raw_onnx = os.path.join(vad_dir, "model.onnx")
    if os.path.exists(model_quant) and os.path.exists(raw_onnx):
        os.unlink(raw_onnx)
    log("VAD ONNX ready")


def onnx_smoke_test(onnx_dir, vad_dir):
    """Run funasr-onnx SenseVoiceSmall + VAD once to verify the INT8 models load."""
    log("Running ONNX smoke test (1s silent audio)...")
    import numpy as np

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
        tmp.close()

        from funasr_onnx import SenseVoiceSmall

        model = SenseVoiceSmall(onnx_dir, batch_size=1, quantize=True)
        res = model([tmp.name], language="auto", textnorm="woitn")
        log(f"ONNX smoke test OK (result: {str(res)[:120]})")

        if vad_dir:
            from funasr_onnx.vad_bin import Fsmn_vad
            vad = Fsmn_vad(model_dir=vad_dir, device_id="-1", quantize=True)
            out = vad(np.zeros(16000, dtype=np.float32))
            log("VAD smoke test OK")
    finally:
        os.unlink(tmp.name)


def torch_smoke_test(model_dir):
    log("Running torch smoke test (1s silent audio)...")
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
            vad_model=VAD_MODEL_ID,
            vad_kwargs={"max_single_segment_time": 30000},
            device="cpu",
        )
        model.generate(input=tmp.name, cache={}, language="auto", use_itn=True)
        log("torch smoke test OK")
    finally:
        os.unlink(tmp.name)


def setup_onnx(args):
    os.makedirs(MODELS_DIR, exist_ok=True)

    onnx_dir = args.onnx_dir if os.path.isdir(args.onnx_dir) else snapshot_download(
        args.onnx_dir or ASR_ONNX_MODEL_ID, MODELS_DIR, args.force)
    vad_dir = args.vad_dir if os.path.isdir(args.vad_dir) else snapshot_download(VAD_MODEL_ID, MODELS_DIR, args.force)

    # funasr-onnx needs the sentencepiece BPE model, which lives in the torch
    # snapshot only. Borrow it, then drop the ~900MB torch checkpoint.
    torch_dir = os.path.join(MODELS_DIR, "models", "iic--SenseVoiceSmall", "snapshots", "master")
    bpe_src = os.path.join(torch_dir, BPE_MODEL_FILE)
    if not os.path.exists(bpe_src):
        torch_dir = snapshot_download(ASR_MODEL_ID, MODELS_DIR, args.force)
        bpe_src = os.path.join(torch_dir, BPE_MODEL_FILE)
    if os.path.exists(bpe_src):
        bpe_dst = os.path.join(onnx_dir, BPE_MODEL_FILE)
        if not os.path.exists(bpe_dst):
            import shutil
            shutil.copy(bpe_src, bpe_dst)
            log(f"Copied {BPE_MODEL_FILE} into the ONNX model dir")
    else:
        raise RuntimeError("Could not find the SenseVoice BPE model file")

    export_vad_onnx(vad_dir)

    manifest = {"onnx_dir": to_relative(onnx_dir), "vad_model": to_relative(vad_dir)}
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"Wrote manifest: {MANIFEST}")

    if not args.skip_smoke_test:
        try:
            onnx_smoke_test(onnx_dir, vad_dir)
        except Exception as e:
            log(f"ONNX smoke test FAILED: {e}")
            sys.exit(1)

    log("Done. ONNX SenseVoice models are ready for offline use.")
    print(MANIFEST)


def setup_torch(args):
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_dir = args.model_dir if os.path.isdir(args.model_dir) else snapshot_download(args.model_dir, MODELS_DIR, args.force)
    vad_dir = args.vad_dir if os.path.isdir(args.vad_dir) else snapshot_download(args.vad_dir, MODELS_DIR, args.force)

    if not args.no_fp16:
        to_fp16_inplace(model_dir)
    backup_pt = os.path.join(model_dir, "model.pt.fp32.bak")
    if os.path.exists(backup_pt):
        os.remove(backup_pt)
        log("Removed fp32 backup from the model dir")

    manifest = {"model_dir": to_relative(model_dir), "vad_model": to_relative(vad_dir)}
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"Wrote manifest: {MANIFEST}")

    if not args.skip_smoke_test:
        try:
            torch_smoke_test(model_dir)
        except Exception as e:
            log(f"Torch smoke test FAILED: {e}")
            sys.exit(1)

    log("Done. SenseVoice torch models are ready for offline use.")
    print(MANIFEST)


def to_fp16_inplace(model_dir):
    """Halve the on-disk model.pt by converting its weights to float16 (torch mode only)."""
    import torch
    import shutil

    model_pt = os.path.join(model_dir, "model.pt")
    backup_pt = model_pt + ".fp32.bak"
    if not os.path.exists(model_pt):
        return False
    sd = torch.load(model_pt, map_location="cpu")
    if not isinstance(sd, dict):
        return False
    is_already_fp16 = all(
        (not torch.is_tensor(v)) or (v.dtype == torch.float16)
        for v in sd.values()
    )
    if is_already_fp16:
        log("model.pt is already fp16; skipping conversion")
        return True
    if not os.path.exists(backup_pt):
        shutil.copy(model_pt, backup_pt)
    log(f"Converting {model_pt} to fp16 (backup at {os.path.basename(backup_pt)})...")

    def walk(obj):
        if torch.is_tensor(obj):
            return obj.half() if obj.dtype.is_floating_point else obj
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [walk(v) for v in obj]
        return obj

    torch.save(walk(sd), model_pt)
    log(f"Converted -> {os.path.getsize(model_pt) / 1e6:.1f} MB")
    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-download SenseVoice models")
    parser.add_argument("--onnx", dest="onnx", action="store_true", help="Use the INT8 ONNX ASR model (default)")
    parser.add_argument("--torch", dest="onnx", action="store_false", help="Use the torch ASR model")
    parser.set_defaults(onnx=True)
    parser.add_argument("--model-dir", default=ASR_MODEL_ID, help="(torch mode) ASR model id or local path")
    parser.add_argument("--onnx-dir", default="", help="(onnx mode) ASR ONNX model id or local path")
    parser.add_argument("--vad-dir", default=VAD_MODEL_ID, help="VAD model id or local path")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip inference check")
    parser.add_argument("--no-fp16", action="store_true", help="(torch mode) Keep the ASR weights in fp32")
    args = parser.parse_args()

    if args.onnx:
        setup_onnx(args)
    else:
        setup_torch(args)


if __name__ == "__main__":
    main()