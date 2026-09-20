import pytest

from processing import time_ranges
from processing.time_ranges import (
    RangeError,
    build_custom_units,
    current_shot_indices,
    format_ranges,
    parse_ranges,
    shots_overlapping,
    validate_mode,
)


SHOTS = [
    {"shot_id": 1, "start_time": 0.0, "end_time": 3.0},
    {"shot_id": 2, "start_time": 3.0, "end_time": 6.0},
    {"shot_id": 3, "start_time": 6.0, "end_time": 9.0},
    {"shot_id": 4, "start_time": 9.0, "end_time": 12.0},
]


def test_single_range():
    assert parse_ranges("00:00:10-00:00:20") == [{"start": 10.0, "end": 20.0}]


def test_multiple_ranges_are_sorted_chronologically():
    assert parse_ranges("00:01:05-00:01:30,00:00:10-00:00:20") == [
        {"start": 10.0, "end": 20.0},
        {"start": 65.0, "end": 90.0},
    ]


@pytest.mark.parametrize("text", [
    "00:00:10-00:00:20",       # canonical
    "0:0:10-0:0:20",           # unpadded
    "00:00:10 - 00:00:20",     # spaces
    "00：00：10-00：00：20",    # full-width colons inside the timestamps
    "00:00:10至00:00:20",      # Chinese range word
    "00:00:10~00:00:20",       # tilde
])
def test_accepted_spellings(text):
    assert parse_ranges(text) == [{"start": 10.0, "end": 20.0}]


@pytest.mark.parametrize("text", [
    "00:00:10-00:00:20,00:01:05-00:01:30",
    "00:00:10-00:00:20，00:01:05-00:01:30",
    "00:00:10-00:00:20;00:01:05-00:01:30",
    "00:00:10-00:00:20、00:01:05-00:01:30",
    "00:00:10-00:00:20\n00:01:05-00:01:30",
])
def test_accepted_list_separators(text):
    assert parse_ranges(text) == [
        {"start": 10.0, "end": 20.0},
        {"start": 65.0, "end": 90.0},
    ]


def test_shorter_forms_are_padded_out():
    assert parse_ranges("00:10-00:20") == [{"start": 10.0, "end": 20.0}]
    assert parse_ranges("10-20") == [{"start": 10.0, "end": 20.0}]


def test_out_of_range_components_carry_over():
    # 75 seconds is 1m15s rather than an error.
    assert parse_ranges("00:00:75-00:02:00") == [{"start": 75.0, "end": 120.0}]


def test_sub_second_input_is_rounded():
    assert parse_ranges("00:00:01.6-00:00:05.4") == [{"start": 2.0, "end": 5.0}]


def test_format_ranges_round_trips():
    text = "00:00:10-00:00:20,00:01:05-00:01:30"
    assert format_ranges(parse_ranges(text)) == text


def test_empty_input_is_rejected():
    with pytest.raises(RangeError, match="at least one"):
        parse_ranges("   ,  ")


@pytest.mark.parametrize("text", ["00:00:10", "00:00:10-", "-00:00:20", "a-b", "00:00:10-00:00:2x"])
def test_malformed_input_is_rejected(text):
    with pytest.raises(RangeError):
        parse_ranges(text)


def test_start_must_precede_end():
    with pytest.raises(RangeError, match="earlier"):
        parse_ranges("00:00:20-00:00:10")
    with pytest.raises(RangeError, match="earlier"):
        parse_ranges("00:00:10-00:00:10")


def test_overlapping_ranges_are_rejected():
    with pytest.raises(RangeError, match="overlap"):
        parse_ranges("00:00:10-00:00:30,00:00:20-00:00:40")


def test_touching_ranges_are_not_overlapping():
    assert parse_ranges("00:00:10-00:00:20,00:00:20-00:00:30") == [
        {"start": 10.0, "end": 20.0},
        {"start": 20.0, "end": 30.0},
    ]


def test_range_past_the_end_of_the_video_is_rejected():
    assert parse_ranges("00:00:05-00:00:10", duration=12.0)
    with pytest.raises(RangeError, match="after the video"):
        parse_ranges("00:00:05-00:00:20", duration=12.0)


def test_validate_mode():
    assert validate_mode(None, "") == ("full", [])
    assert validate_mode("full", "garbage") == ("full", [])
    assert validate_mode("EXTRA", "00:00:10-00:00:20") == (
        "extra", [{"start": 10.0, "end": 20.0}])
    assert validate_mode("only", "00:00:10-00:00:20")[0] == "only"
    with pytest.raises(RangeError, match="Unknown range mode"):
        validate_mode("sideways", "00:00:10-00:00:20")


def test_validate_mode_requires_ranges_for_range_modes():
    with pytest.raises(RangeError, match="at least one"):
        validate_mode("only", "")


def test_shots_overlapping_is_half_open():
    # Touching a boundary does not claim the shot on the other side of it.
    assert shots_overlapping(SHOTS, 1.0, 3.0) == [1]
    assert shots_overlapping(SHOTS, 1.0, 4.0) == [1, 2]
    assert shots_overlapping(SHOTS, 2.0, 7.0) == [1, 2, 3]


def test_build_custom_units_spans_exactly_what_was_asked():
    units = build_custom_units(parse_ranges("00:00:02-00:00:04"), SHOTS)
    assert units == [{
        "unit_id": "C1",
        "start": 2.0,
        "end": 4.0,
        "shot_ids": [1, 2],
        "mode": "custom",
    }]


def test_build_custom_units_numbers_them_in_order():
    units = build_custom_units(
        parse_ranges("00:00:01-00:00:02,00:00:05-00:00:06"), SHOTS)
    assert [u["unit_id"] for u in units] == ["C1", "C2"]


def test_current_shot_indices_are_zero_based():
    unit = {"start": 3.0, "end": 6.0, "shot_ids": [2]}
    assert current_shot_indices(unit, SHOTS) == [1]


def test_current_shot_indices_fall_back_to_the_nearest_shot():
    # A range beyond the last shot would otherwise render "[]" in the prompt.
    unit = {"start": 20.0, "end": 25.0, "shot_ids": []}
    assert current_shot_indices(unit, SHOTS) == [3]


def test_current_shot_indices_without_any_shots():
    assert current_shot_indices({"start": 0, "end": 5, "shot_ids": []}, []) == []
