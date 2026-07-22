import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processing.context_extender import find_overlapping_intervals, extend_context


def _make_shot(start, end):
    return {"start_time": start, "end_time": end, "description": f"shot {start}-{end}"}


class TestFindOverlappingIntervals:
    def test_basic_overlap(self):
        shots = [_make_shot(0, 2), _make_shot(2, 4), _make_shot(4, 6)]
        result = find_overlapping_intervals(1, 3, shots)
        assert result == [0, 1]

    def test_no_overlap(self):
        shots = [_make_shot(0, 1), _make_shot(5, 6)]
        result = find_overlapping_intervals(2, 4, shots)
        assert result == []

    def test_exact_match_point_overlap(self):
        shots = [_make_shot(0, 5), _make_shot(5, 10)]
        result = find_overlapping_intervals(0, 5, shots)
        assert result == [0, 1]

    def test_ad_inside_single_shot(self):
        shots = [_make_shot(0, 10)]
        result = find_overlapping_intervals(3, 7, shots)
        assert result == [0]

    def test_multiple_overlaps(self):
        shots = [_make_shot(0, 3), _make_shot(2, 5), _make_shot(4, 7)]
        result = find_overlapping_intervals(2, 5, shots)
        assert result == [0, 1, 2]


class TestExtendContext:
    def test_empty_shots(self):
        result = extend_context([], 5, 10)
        assert result == []

    def test_single_shot_overlapping(self):
        shots = [_make_shot(0, 5), _make_shot(5, 10), _make_shot(10, 15)]
        result = extend_context(shots, 5, 10)
        assert len(result) > 0
        assert any(s["shot_label"].startswith("m-") for s in result)

    def test_context_labels(self):
        shots = [_make_shot(0, 2), _make_shot(2, 4), _make_shot(4, 6),
                 _make_shot(6, 8), _make_shot(8, 10), _make_shot(10, 12)]
        result = extend_context(shots, 4, 8)
        labels = [s["shot_label"] for s in result]
        assert any(l.startswith("l-") for l in labels)
        assert any(l.startswith("m-") for l in labels)
        assert any(l.startswith("r-") for l in labels)

    def test_past_label_ordering(self):
        shots = [_make_shot(0, 2), _make_shot(2, 4), _make_shot(4, 6),
                 _make_shot(6, 8), _make_shot(8, 10)]
        result = extend_context(shots, 4, 6)
        past_labels = [s["shot_label"] for s in result if s["shot_label"].startswith("l-")]
        if len(past_labels) > 1:
            assert past_labels[0].endswith("-1")
            assert past_labels[1].endswith("-0")

    def test_no_nearby_shots_picks_closest(self):
        shots = [_make_shot(0, 2), _make_shot(20, 22), _make_shot(50, 52)]
        result = extend_context(shots, 10, 15)
        assert len(result) == 1
        assert result[0]["shot_label"] == "m-0"

    def test_min_shot_duration_filter(self):
        shots = [_make_shot(0, 3), _make_shot(3, 4), _make_shot(4, 5),
                 _make_shot(5, 10), _make_shot(10, 12),
                 _make_shot(12, 14)]
        result = extend_context(shots, 5, 10, min_shot_duration=0.5)
        labels = [s["shot_label"] for s in result]
        past_labels = [l for l in labels if l.startswith("l-")]
        future_labels = [l for l in labels if l.startswith("r-")]
        assert len(past_labels) >= 1
        assert len(future_labels) >= 1

    def test_does_not_mutate_original(self):
        shots = [_make_shot(0, 2), _make_shot(2, 4), _make_shot(4, 6)]
        extend_context(shots, 2, 4)
        assert "shot_label" not in shots[0]

    def test_internal_range_affects_past_future(self):
        shots = [_make_shot(0, 2), _make_shot(2, 4), _make_shot(4, 6),
                 _make_shot(6, 8), _make_shot(8, 10)]
        result_tight = extend_context(shots, 4, 6, internal_range=1.0)
        result_wide = extend_context(shots, 4, 6, internal_range=5.0)
        labels_tight = len(result_tight)
        labels_wide = len(result_wide)
        assert labels_wide >= labels_tight
