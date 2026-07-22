"""
Transcription module using SenseVoice (FunASR)
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def transcribe_video(
    video_path: str,
    whisper_cpp_path: str = None,
    model_path: str = None,
    language: str = "auto",
    compute_type: str = "int8",
    use_itn: bool = True,
    callback=None
) -> list:
    """
    Transcribe video audio using SenseVoice (FunASR).
    
    Args:
        video_path: Path to the video file
        callback: Optional callback function(progress_float, message_str)
        
    Returns:
        List of dicts with keys: text, start_time, end_time
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if callback:
        callback(0.0, "Loading SenseVoice model...")
    
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'SenseVoice', 'model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SenseVoice model not found at: {model_path}")

    try:
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError:
        raise RuntimeError("funasr not installed. Please run: uv pip install funasr --system")

    logger.info(f"Loading SenseVoice model from: {model_path}")
    model = AutoModel(
        model=model_path,
        trust_remote_code=True,
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 30000},
    )
    
    if callback:
        callback(0.3, "Model loaded, transcribing...")

    logger.info(f"Transcribing: {video_path}")
    result = model.generate(
        input=video_path,
        cache={},
        language=language,
        use_itn=use_itn,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    
    if callback:
        callback(0.9, "Processing results...")

    payload = result[0]
    text = rich_transcription_postprocess(payload["text"])
    
    segments = []
    raw_segments = payload.get("timestamp") or []
    
    for entry in raw_segments:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            start = float(entry[0]) / 1000
            end = float(entry[1]) / 1000
            seg_text = str(entry[2]).strip()
            if seg_text:
                segments.append({
                    "text": seg_text,
                    "start_time": start,
                    "end_time": end
                })
    
    if callback:
        callback(1.0, f"Transcribed {len(segments)} segments")
    
    return segments
