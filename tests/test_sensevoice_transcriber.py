from processing.sensevoice_transcriber import _parse_torch_result as _parse_result


def test_parse_result_empty():
    assert _parse_result(None) == []
    assert _parse_result([]) == []


def test_parse_result_sentence_info():
    """Millisecond timestamps from sentence_info must become float seconds."""
    res = [{
        "text": "ignored-whole-text",
        "sentence_info": [
            {"text": "Hello there", "start": 0, "end": 2500},
            {"text": "Second sentence", "start": 2500, "end": 5000},
        ],
    }]
    result = _parse_result(res)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["text"] == "Hello there"
    assert result[0]["start_time"] == 0.0
    assert result[0]["end_time"] == 2.5
    assert result[1]["start_time"] == 2.5
    assert result[1]["end_time"] == 5.0


def test_parse_result_fallback_single_segment(tmp_path):
    """Without sentence_info, fall back to one segment covering the input."""
    video_file = tmp_path / "fake.mp4"
    video_file.touch()
    res = [{"text": "only text"}]
    result = _parse_result(res, str(video_file))
    assert len(result) == 1
    assert result[0]["text"] == "only text"
    assert result[0]["start_time"] == 0.0
