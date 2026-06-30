import os
import pytest
from unittest.mock import patch, MagicMock
from processing.shot_detector import detect_shots


def test_detect_shots_returns_list():
    """Test that detect_shots returns a list of shot dicts."""
    mock_start = MagicMock()
    mock_start.get_seconds.return_value = 0.0
    mock_end = MagicMock()
    mock_end.get_seconds.return_value = 2.5

    with patch("processing.shot_detector.os.path.exists", return_value=True), \
         patch("processing.shot_detector.detect", return_value=[(mock_start, mock_end)]):
        result = detect_shots("fake_video.mp4")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["shot_id"] == 1
    assert result[0]["start_time"] == 0.0
    assert result[0]["end_time"] == 2.5


def test_detect_shots_file_not_found():
    """Test that detect_shots raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        detect_shots("nonexistent.mp4")


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
