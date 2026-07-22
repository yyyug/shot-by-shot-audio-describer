import pytest
import os
from processing.whisper_transcriber import transcribe_video


def test_transcribe_video_exists():
    """Verify transcribe_video function is importable and callable."""
    assert callable(transcribe_video)


def test_transcribe_video_raises_on_missing_file():
    """Verify proper error when video file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        transcribe_video("nonexistent.mp4")


def test_transcribe_video_returns_list(tmp_path):
    """Verify return type is list for SenseVoice transcription."""
    from unittest.mock import patch, MagicMock
    
    # Create a fake video file
    video_file = tmp_path / "test.mp4"
    video_file.touch()
    
    # Create a fake model file
    model_file = tmp_path / "model.pt"
    model_file.touch()
    
    # Mock the entire SenseVoice flow
    mock_result = [{
        "text": "Hello world Test sentence",
        "timestamp": [
            [0, 2500, "Hello world"],
            [2500, 5000, "Test sentence"]
        ]
    }]
    
    mock_model = MagicMock()
    mock_model.generate.return_value = mock_result
    
    # Mock the funasr module
    mock_funasr = MagicMock()
    mock_funasr.AutoModel.return_value = mock_model
    
    mock_postprocess = MagicMock(side_effect=lambda x: x)
    
    with patch.dict('sys.modules', {
        'funasr': mock_funasr,
        'funasr.utils.postprocess_utils': MagicMock(rich_transcription_postprocess=mock_postprocess)
    }):
        result = transcribe_video(str(video_file), model_path=str(model_file))
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["text"] == "Hello world"
    assert result[0]["start_time"] == 0.0
    assert result[0]["end_time"] == 2.5
