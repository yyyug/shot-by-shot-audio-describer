from processing.dialogue_gap_detector import detect_ad_intervals


SHOTS = [
    {"shot_id": 1, "start_time": 0.0, "end_time": 3.0},
    {"shot_id": 2, "start_time": 3.0, "end_time": 6.0},
    {"shot_id": 3, "start_time": 6.0, "end_time": 9.0},
    {"shot_id": 4, "start_time": 9.0, "end_time": 12.0},
]


def test_no_subtitles_returns_empty():
    assert detect_ad_intervals([], SHOTS) == []


def test_gaps_between_segments():
    """Two dialogue segments with a gap should produce one AD interval."""
    subtitles = [
        {"text": "Hi", "start_time": 1.0, "end_time": 2.0},
        {"text": "Bye", "start_time": 5.0, "end_time": 6.0},
    ]
    result = detect_ad_intervals(subtitles, SHOTS, min_gap=1.5)
    assert len(result) == 1
    interval = result[0]
    assert interval["start"] == 2.0
    assert interval["end"] == 5.0
    # gap 2-5s spans shots 1 (end 3.0) and 2 (3.0-6.0)
    assert interval["shot_ids"] == [1, 2]


def test_ad_interval_can_span_multiple_shots():
    """A single AD interval covers all shots overlapping the gap."""
    subtitles = [
        {"text": "First", "start_time": 1.0, "end_time": 3.0},
        {"text": "Second", "start_time": 8.0, "end_time": 10.0},
    ]
    result = detect_ad_intervals(subtitles, SHOTS, min_gap=1.5)
    assert len(result) == 1
    assert result[0]["shot_ids"] == [2, 3]


def test_leading_and_trailing_gaps():
    subtitles = [
        {"text": "Hi", "start_time": 6.0, "end_time": 7.0},
    ]
    result = detect_ad_intervals(subtitles, SHOTS, min_gap=1.5, video_duration=12.0)
    assert len(result) == 2
    assert result[0]["shot_ids"] == [1, 2]
    assert result[1]["shot_ids"] == [4]


def test_min_gap_filters_small_gaps():
    subtitles = [
        {"text": "A", "start_time": 1.0, "end_time": 1.5},
        {"text": "B", "start_time": 2.0, "end_time": 2.5},
    ]
    result = detect_ad_intervals(subtitles, SHOTS, min_gap=1.5)
    assert result == []
