import os
import subprocess
import tempfile
import json


def transcribe_video(
    video_path: str,
    whisper_cpp_path: str = "whisper-cpp",
    model_path: str = None,
    language: str = None,
    compute_type: str = "int8"
) -> list:
    """
    Transcribe video audio using whisper.cpp.

    Args:
        video_path: Path to the video file
        whisper_cpp_path: Path to whisper-cpp binary
        model_path: Path to whisper model file (e.g., models/ggml-medium.bin)
        language: Language code (e.g., 'en', 'zh') or None for auto-detect
        compute_type: Compute type (default: 'int8')

    Returns:
        List of dicts with keys: text, start_time, end_time
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

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

        if model_path:
            cmd.extend(["-m", model_path])

        if language:
            cmd.extend(["-l", language])

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

        return segments
