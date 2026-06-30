import pytest
from processing.whisper_transcriber import transcribe_video


def test_transcribe_video_exists():
    """Verify transcribe_video function is importable and callable."""
    assert callable(transcribe_video)


def test_transcribe_video_raises_on_missing_file():
    """Verify proper error when video file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        transcribe_video("nonexistent.mp4")


def test_transcribe_video_returns_list(tmp_path):
    """Verify return type is list with mock whisper output."""
    import json
    from unittest.mock import patch, MagicMock

    # Create a fake video file
    video_file = tmp_path / "test.mp4"
    video_file.touch()

    # Mock subprocess.run to simulate whisper output
    mock_output = {
        "segments": [
            {"text": "Hello world", "start": 0.0, "end": 2.5},
            {"text": "Test sentence", "start": 2.5, "end": 5.0}
        ]
    }

    def mock_run(cmd, **kwargs):
        # The implementation creates a TemporaryDirectory and passes it as the -o arg
        # Extract the output dir from the command
        o_idx = cmd.index("-o")
        output_dir = cmd[o_idx + 1]
        output_file = os.path.join(output_dir, "output.json")
        with open(output_file, 'w') as f:
            json.dump(mock_output, f)
        return MagicMock(returncode=0, stdout="", stderr="")

    import os
    with patch('processing.whisper_transcriber.subprocess.run', side_effect=mock_run):
        result = transcribe_video(str(video_file), whisper_cpp_path="fake-whisper")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["text"] == "Hello world"
    assert result[0]["start_time"] == 0.0
    assert result[0]["end_time"] == 2.5
