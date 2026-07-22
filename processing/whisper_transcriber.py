"""
Transcription module using whisper.cpp
"""
import os
import subprocess
import tempfile
import json
import logging

logger = logging.getLogger(__name__)


def transcribe_video(
    video_path: str,
    whisper_cpp_path: str = "whisper-cpp",
    model_path: str = None,
    language: str = "auto",
    compute_type: str = "int8",
    use_itn: bool = True,
    callback=None
) -> list:
    """
    Transcribe video audio using whisper.cpp.
    
    Args:
        video_path: Path to the video file
        whisper_cpp_path: Path to whisper-cpp binary
        model_path: Path to whisper model file
        language: Language code ('auto', 'en', 'zh', etc.)
        compute_type: Compute type ('int8', 'float16', etc.)
        use_itn: Use inverse text normalization
        callback: Optional callback function(progress_float, message_str)
        
    Returns:
        List of dicts with keys: text, start_time, end_time
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if callback:
        callback(0.0, "Starting transcription...")
    
    # Default model path
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'whisper', 'ggml-small.bin')
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = os.path.join(tmpdir, "output")
        
        # Build command
        cmd = [
            whisper_cpp_path,
            "-f", video_path,
            "-o", tmpdir,
            "--output-format", "json",
            "--compute-type", compute_type
        ]
        
        if model_path and os.path.exists(model_path):
            cmd.extend(["-m", model_path])
        
        if language and language != "auto":
            cmd.extend(["-l", language])
        
        logger.info(f"Running whisper.cpp: {' '.join(cmd)}")
        
        if callback:
            callback(0.3, "Transcribing audio...")
        
        # Run whisper.cpp
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"whisper-cpp not found at: {whisper_cpp_path}. "
                "Please install whisper.cpp or provide the correct path."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"whisper.cpp failed: {e.stderr}")
        
        if callback:
            callback(0.9, "Processing results...")
        
        # Parse JSON output
        json_path = output_prefix + ".json"
        if not os.path.exists(json_path):
            raise RuntimeError(f"Output file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to standard format
        segments = []
        for seg in data.get("segments", []):
            segments.append({
                "text": seg["text"].strip(),
                "start_time": seg["start"],
                "end_time": seg["end"]
            })
        
        if callback:
            callback(1.0, f"Transcribed {len(segments)} segments")
        
        return segments
