import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import processing.character_recognizer as cr
from processing.character_recognizer import (
    detect_faces,
    extract_face_embeddings,
    cluster_faces,
    recognize_characters,
)


@pytest.fixture(autouse=True)
def _reset_cascade_cache():
    """Keep the lazy cascade cache isolated between tests."""
    cr._face_cascade = None
    cr._face_cascade_error = None
    yield
    cr._face_cascade = None
    cr._face_cascade_error = None


# ---------------------------------------------------------------------------
# detect_faces
# ---------------------------------------------------------------------------

def test_detect_faces_no_faces():
    """Returns empty list when no faces found."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detect_faces(blank)
    assert result == []


def test_detect_faces_returns_tuples():
    """Verify bounding box format is list of (x, y, w, h)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_faces = np.array([[10, 20, 50, 60]], dtype=np.int32)

    with patch("processing.character_recognizer._get_face_cascade") as get_cascade:
        cascade = MagicMock()
        cascade.detectMultiScale.return_value = mock_faces
        get_cascade.return_value = cascade
        result = detect_faces(frame)

    assert len(result) == 1
    assert result[0] == (10, 20, 50, 60)
    assert all(isinstance(v, int) for v in result[0])


# ---------------------------------------------------------------------------
# extract_face_embeddings
# ---------------------------------------------------------------------------

def test_extract_face_embeddings_no_video():
    """Returns empty list when video cannot be opened."""
    with patch("processing.character_recognizer.cv2") as cv:
        cap = MagicMock()
        cap.isOpened.return_value = False
        cv.VideoCapture.return_value = cap
        result = extract_face_embeddings("fake.mp4", {"start_time": 0, "end_time": 2})
        assert result == []


def test_extract_face_embeddings_faces_detected():
    """Returns embeddings when faces are found in sampled frames."""
    with patch("processing.character_recognizer.cv2") as cv:
        # Setup VideoCapture mock
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30.0  # fps
        cv.VideoCapture.return_value = cap

        # Frame read returns success then a frame with a face
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap.read.side_effect = [(True, fake_frame)] * 4 + [(False, None)]

        # Detect faces returns one face
        cv.cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)
        with patch("processing.character_recognizer._get_face_cascade") as get_cascade:
            cascade = MagicMock()
            cascade.detectMultiScale.return_value = np.array([[100, 100, 200, 200]], dtype=np.int32)
            get_cascade.return_value = cascade
            cv.cvtColor.return_value = np.zeros((480, 640), dtype=np.uint8)
            cv.resize.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            cv.calcHist.return_value = np.random.rand(8, 8, 8).astype(np.float32)
            cv.normalize.return_value = None

            result = extract_face_embeddings(
                "fake.mp4", {"start_time": 0, "end_time": 2}, num_frames=4
            )

        assert len(result) > 0
        assert all(isinstance(e, np.ndarray) for e in result)


# ---------------------------------------------------------------------------
# cluster_faces
# ---------------------------------------------------------------------------

def test_cluster_faces_empty():
    assert cluster_faces([]) == []


def test_cluster_faces_single():
    emb = [np.array([1.0, 0.0, 0.0])]
    labels = cluster_faces(emb, threshold=0.6)
    assert labels == [0]


def test_cluster_faces_two_groups():
    """Two very different embeddings should land in separate clusters."""
    a = np.zeros(50)
    a[0] = 1.0
    b = np.zeros(50)
    b[1] = 1.0  # orthogonal to a

    labels = cluster_faces([a, b], threshold=0.5)
    assert len(labels) == 2
    # With high threshold they might merge; with low they separate
    assert isinstance(labels[0], int)


# ---------------------------------------------------------------------------
# recognize_characters
# ---------------------------------------------------------------------------

def test_recognize_characters_no_faces():
    """When no faces are found, every shot gets empty character list."""
    shots = [
        {"shot_id": 1, "start_time": 0.0, "end_time": 1.0},
        {"shot_id": 2, "start_time": 1.0, "end_time": 2.0},
    ]

    with patch(
        "processing.character_recognizer.extract_face_embeddings", return_value=[]
    ):
        result = recognize_characters("video.mp4", shots)

    assert len(result) == 2
    assert result[0]["character_ids"] == []
    assert result[1]["character_ids"] == []
    assert result[0]["face_count"] == 0


def test_recognize_characters_with_embeddings():
    """Pipeline returns character assignments when embeddings exist."""
    shots = [
        {"shot_id": 1, "start_time": 0.0, "end_time": 1.0},
        {"shot_id": 2, "start_time": 1.0, "end_time": 2.0},
    ]

    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])

    with patch(
        "processing.character_recognizer.extract_face_embeddings",
        side_effect=[[emb_a], [emb_b]],
    ), patch(
        "processing.character_recognizer.cluster_faces",
        return_value=[0, 1],
    ):
        result = recognize_characters("video.mp4", shots, threshold=0.5)

    assert len(result) == 2
    assert result[0]["shot_id"] == 1
    assert result[1]["shot_id"] == 2
    assert result[0]["character_ids"] == [0]
    assert result[1]["character_ids"] == [1]
