import os
from scenedetect import detect, AdaptiveDetector, SceneManager, open_video

def detect_shots(video_path: str, threshold: float = 27.0, callback=None) -> list:
    """
    Detect shot boundaries in a video file.
    
    Args:
        video_path: Path to the video file
        threshold: AdaptiveDetector threshold (default 27.0)
        callback: Optional callback function(progress_float, message_str)
    
    Returns:
        List of dicts with keys: shot_id, start_time, end_time
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if callback:
        # Use SceneManager with callback for progress updates
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold))
        
        total_frames = video.duration.get_frames() if hasattr(video, 'duration') else 1000
        
        def on_frame(frame, frame_num):
            # frame_num is a FrameTimecode object, convert to frame number
            current_frame = frame_num.get_frames() if hasattr(frame_num, 'get_frames') else 0
            progress = current_frame / total_frames if total_frames > 0 else 0
            callback(progress, f"Processing frame {current_frame}/{total_frames}")
        
        # Process with progress callback
        scene_manager.detect_scenes(video, callback=on_frame)
        scene_list = scene_manager.get_scene_list()
    else:
        # Simple detection without callback
        scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=threshold))
    
    shots = []
    for idx, (start, end) in enumerate(scene_list):
        shots.append({
            "shot_id": idx + 1,
            "start_time": start.get_seconds(),
            "end_time": end.get_seconds()
        })
    
    return shots
