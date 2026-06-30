import pandas as pd
from processing.csv_merger import merge_shots_subtitles, merge_with_descriptions, format_original_style


def test_merge_basic():
    """Test basic shot-subtitle merge with overlapping times."""
    shots = [
        {"shot_id": 1, "start_time": 0.0, "end_time": 5.0},
        {"shot_id": 2, "start_time": 5.0, "end_time": 10.0}
    ]
    subtitles = [
        {"text": "Hello", "start_time": 1.0, "end_time": 3.0},
        {"text": "World", "start_time": 6.0, "end_time": 8.0}
    ]
    result = merge_shots_subtitles(shots, subtitles)
    assert len(result) == 2
    assert "subtitle" in result.columns
    assert result.iloc[0]["subtitle"] == "Hello"
    assert result.iloc[1]["subtitle"] == "World"


def test_merge_empty_shots():
    """Test merge with no shots returns empty DataFrame."""
    result = merge_shots_subtitles([], [])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert list(result.columns) == ["shot_id", "start_time", "end_time", "subtitle"]


def test_merge_no_subtitles():
    """Test merge with shots but no subtitles yields empty subtitle column."""
    shots = [
        {"shot_id": 1, "start_time": 0.0, "end_time": 5.0}
    ]
    result = merge_shots_subtitles(shots, [])
    assert len(result) == 1
    assert result.iloc[0]["subtitle"] == ""


def test_merge_multiple_subtitles_per_shot():
    """Test merge when a single shot contains multiple subtitles."""
    shots = [
        {"shot_id": 1, "start_time": 0.0, "end_time": 10.0}
    ]
    subtitles = [
        {"text": "First", "start_time": 1.0, "end_time": 3.0},
        {"text": "Second", "start_time": 5.0, "end_time": 7.0}
    ]
    result = merge_shots_subtitles(shots, subtitles)
    assert len(result) == 1
    assert result.iloc[0]["subtitle"] == "First Second"


def test_merge_no_overlap():
    """Test merge when subtitles don't overlap any shot."""
    shots = [
        {"shot_id": 1, "start_time": 0.0, "end_time": 2.0}
    ]
    subtitles = [
        {"text": "Late", "start_time": 5.0, "end_time": 8.0}
    ]
    result = merge_shots_subtitles(shots, subtitles)
    assert len(result) == 1
    assert result.iloc[0]["subtitle"] == ""


def test_merge_with_descriptions():
    """Test adding VLM descriptions to merged DataFrame."""
    shots = [
        {"shot_id": 1, "start_time": 0.0, "end_time": 5.0},
        {"shot_id": 2, "start_time": 5.0, "end_time": 10.0}
    ]
    subtitles = [
        {"text": "Hello", "start_time": 1.0, "end_time": 3.0}
    ]
    merged = merge_shots_subtitles(shots, subtitles)
    descriptions = [
        {"shot_id": 1, "description": "A person walking"}
    ]
    result = merge_with_descriptions(merged, descriptions)
    assert "video_description" in result.columns
    assert result.iloc[0]["video_description"] == "A person walking"
    assert result.iloc[1]["video_description"] == ""


def test_merge_with_descriptions_empty():
    """Test merge_with_descriptions with no descriptions."""
    shots = [
        {"shot_id": 1, "start_time": 0.0, "end_time": 5.0}
    ]
    merged = merge_shots_subtitles(shots, [])
    result = merge_with_descriptions(merged, [])
    assert "video_description" in result.columns
    assert result.iloc[0]["video_description"] == ""


def test_format_original_style():
    """Test formatting DataFrame to original project output format."""
    shots = [
        {"shot_id": 1, "start_time": 0.0, "end_time": 5.0},
        {"shot_id": 2, "start_time": 5.0, "end_time": 10.0}
    ]
    subtitles = [
        {"text": "Hello", "start_time": 1.0, "end_time": 3.0}
    ]
    merged = merge_shots_subtitles(shots, subtitles)
    result = format_original_style(merged)
    assert "anno_idx" in result.columns
    assert "imdbid" in result.columns
    assert "start" in result.columns
    assert "end" in result.columns
    assert "text_gen" in result.columns
    assert result.iloc[0]["anno_idx"] == 1
    assert result.iloc[1]["anno_idx"] == 2
    assert result.iloc[0]["imdbid"] == "user_upload"
