import os
import pytest
from unittest.mock import patch, MagicMock
from processing.shot_detector import detect_shots


def test_detect_shots_file_not_found():
    """Test that detect_shots raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        detect_shots("nonexistent.mp4")


def test_detect_shots_with_callback():
    """Test that callback is called during shot detection."""
    mock_start = MagicMock()
    mock_start.get_frames.return_value = 100
    mock_start.get_seconds.return_value = 0.0
    mock_end = MagicMock()
    mock_end.get_frames.return_value = 200
    mock_end.get_seconds.return_value = 2.5
    
    mock_video = MagicMock()
    mock_video.duration.get_frames.return_value = 500
    
    callback_calls = []
    def test_callback(progress, message):
        callback_calls.append((progress, message))
    
    mock_scene_manager = MagicMock()
    mock_scene_manager.get_scene_list.return_value = [(mock_start, mock_end)]
    
    with patch("processing.shot_detector.os.path.exists", return_value=True), \
         patch("processing.shot_detector.open_video", return_value=mock_video), \
         patch("processing.shot_detector.SceneManager", return_value=mock_scene_manager):
        result = detect_shots("fake_video.mp4", callback=test_callback)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert mock_scene_manager.detect_scenes.called


def test_detect_shots_multiple_shots():
    """Test detection with multiple shots."""
    shots_data = []
    for i in range(3):
        mock_start = MagicMock()
        mock_start.get_seconds.return_value = float(i * 5)
        mock_end = MagicMock()
        mock_end.get_seconds.return_value = float(i * 5 + 4)
        shots_data.append((mock_start, mock_end))

    with patch("processing.shot_detector.os.path.exists", return_value=True), \
         patch("processing.shot_detector.detect", return_value=shots_data):
        result = detect_shots("multi_shot.mp4")

    assert len(result) == 3
    assert [s["shot_id"] for s in result] == [1, 2, 3]
