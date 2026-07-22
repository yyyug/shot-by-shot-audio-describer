import numpy as np
import pytest
from unittest.mock import patch
from processing.shot_labeler import (
    add_shot_label,
    add_character_labels,
    label_frames,
)


def _blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# add_shot_label
# ---------------------------------------------------------------------------

def test_add_shot_label_modifies_frame():
    """Output frame should differ from the original."""
    frame = _blank_frame()
    result = add_shot_label(frame, 1)
    assert result.shape == frame.shape
    assert not np.array_equal(result, frame)


def test_add_shot_label_does_not_mutate_input():
    frame = _blank_frame()
    original = frame.copy()
    add_shot_label(frame, 1)
    assert np.array_equal(frame, original)


def test_add_shot_label_text_region_nonzero():
    """Pixels in the top-left corner should be non-zero after labeling."""
    frame = _blank_frame()
    result = add_shot_label(frame, 5)
    # The label is drawn near (10,10) region
    region = result[5:40, 5:200]
    assert region.sum() > 0


def test_add_shot_label_pil_fallback():
    """When font_path is given but PIL unavailable, falls back to cv2."""
    frame = _blank_frame()
    with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None, "PIL.ImageDraw": None, "PIL.ImageFont": None}):
        result = add_shot_label(frame, 1, font_path="nonexistent.ttf", font_size=30)
    assert result.shape == frame.shape


def test_add_shot_label_with_pil():
    """When PIL is available and font_path is given, uses PIL rendering."""
    frame = _blank_frame()

    mock_font = type("MockFont", (), {"getbbox": lambda self, t: (0, 0, 100, 20)})()

    mock_draw = type("MockDraw", (), {
        "textbbox": lambda self, pos, text, font=None: (0, 0, 100, 20),
        "rectangle": lambda self, *a, **kw: None,
        "text": lambda self, *a, **kw: None,
    })()

    mock_image = type("MockImage", (), {
        "fromarray": classmethod(lambda cls, arr: cls()),
        "Draw": classmethod(lambda cls, img: mock_draw),
    })()

    mock_font_mod = type("MockFontModule", (), {
        "truetype": classmethod(lambda cls, path, size: mock_font),
    })()

    with patch.dict("sys.modules", {
        "PIL": mock_image,
        "PIL.Image": mock_image,
        "PIL.ImageDraw": type("M", (), {"Draw": classmethod(lambda cls, img: mock_draw)})(),
        "PIL.ImageFont": mock_font_mod,
    }):
        result = add_shot_label(frame, 1, font_path="arial.ttf", font_size=30)

    assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# add_character_labels
# ---------------------------------------------------------------------------

def test_add_character_labels_draws_boxes():
    frame = _blank_frame()
    bboxes = [(100, 100, 80, 80)]
    ids = [0]
    colors = [(0, 255, 0)]
    result = add_character_labels(frame, bboxes, ids, colors)
    # Region around the box should have color
    assert result[100, 100, 1] > 0  # green channel


def test_add_character_labels_no_mutation():
    frame = _blank_frame()
    original = frame.copy()
    add_character_labels(frame, [(10, 10, 50, 50)], [0], [(0, 0, 255)])
    assert np.array_equal(frame, original)


def test_add_character_labels_multiple_faces():
    frame = _blank_frame()
    bboxes = [(10, 10, 40, 40), (200, 200, 50, 50)]
    ids = [0, 1]
    colors = [(0, 255, 0), (255, 0, 0)]
    result = add_character_labels(frame, bboxes, ids, colors)
    assert result.shape == frame.shape


def test_add_character_labels_empty():
    frame = _blank_frame()
    result = add_character_labels(frame, [], [], [])
    assert np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# label_frames
# ---------------------------------------------------------------------------

def test_label_frames_empty():
    assert label_frames([], 1) == []


def test_label_frames_returns_same_count():
    frames = [_blank_frame(), _blank_frame()]
    result = label_frames(frames, 3)
    assert len(result) == 2


def test_label_frames_with_character_info():
    frames = [_blank_frame()]
    character_info = {
        "bboxes": [[(50, 50, 60, 60)]],
        "character_ids": [[0]],
        "colors": [(0, 255, 0)],
    }
    result = label_frames(frames, 1, character_info=character_info)
    assert len(result) == 1
    assert result[0].shape == (480, 640, 3)


def test_label_frames_no_mutation():
    frames = [_blank_frame()]
    originals = [f.copy() for f in frames]
    label_frames(frames, 1)
    for f, o in zip(frames, originals):
        assert np.array_equal(f, o)


def test_label_frames_character_info_index_out_of_range():
    """Frames beyond the character_info list still get the shot label."""
    frames = [_blank_frame(), _blank_frame()]
    character_info = {
        "bboxes": [[(10, 10, 20, 20)]],
        "character_ids": [[0]],
        "colors": [(0, 255, 0)],
    }
    result = label_frames(frames, 2, character_info=character_info)
    assert len(result) == 2
    # Both should have shot label; second should have no character label
    assert not np.array_equal(result[0], result[1])
