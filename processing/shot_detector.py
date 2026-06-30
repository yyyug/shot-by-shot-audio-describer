import os
from scenedetect import detect, AdaptiveDetector

def detect_shots(video_path: str, threshold: float = 27.0) -> list:
    """
    Detect shot boundaries in a video file.
    
    Args:
        video_path: Path to the video file
        threshold: AdaptiveDetector threshold (default 27.0)
    
    Returns:
        List of dicts with keys: shot_id, start_time, end_time
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=threshold))
    
    shots = []
    for idx, (start, end) in enumerate(scene_list):
        shots.append({
            "shot_id": idx + 1,
            "start_time": start.get_seconds(),
            "end_time": end.get_seconds()
        })
    
    return shots
