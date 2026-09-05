"""
Character recognizer - face detection and clustering for shot-by-shot analysis.
"""
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple


_face_cascade = None
_face_cascade_error = None


def _get_face_cascade():
    """Lazily load the Haar cascade so importing this module never fails,
    even on OpenCV builds without CascadeClassifier (e.g. OpenCV 5)."""
    global _face_cascade, _face_cascade_error
    if _face_cascade is not None:
        return _face_cascade
    if _face_cascade_error is not None:
        raise RuntimeError("Face detection unavailable") from _face_cascade_error
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade: {cascade_path}")
    except Exception as e:
        _face_cascade_error = e
        raise
    return _face_cascade


def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a single frame using OpenCV Haar cascades.

    Args:
        frame: BGR image as numpy array

    Returns:
        List of (x, y, w, h) bounding boxes
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def _compute_histogram(face_roi: np.ndarray) -> np.ndarray:
    """Compute normalized colour histogram of a face region for simple embedding."""
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def extract_face_embeddings(
    video_path: str,
    shot: Dict,
    num_frames: int = 16,
) -> List[np.ndarray]:
    """
    Extract face feature vectors from a shot's frames.

    Samples ``num_frames`` evenly across the shot, detects faces, crops
    the largest face per frame, and returns a histogram-based embedding
    for each detected face.

    Args:
        video_path: Path to the video file.
        shot: Dict with ``start_time`` and ``end_time`` (seconds).
        num_frames: Number of frames to sample.

    Returns:
        List of 1-D numpy embedding arrays (may be empty if no faces found).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(shot["start_time"] * fps)
    end_frame = int(shot["end_time"] * fps)
    total_frames = end_frame - start_frame
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(start_frame, end_frame - 1, min(num_frames, total_frames), dtype=int)

    embeddings: List[np.ndarray] = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret:
            continue

        faces = detect_faces(frame)
        if not faces:
            continue

        # Pick the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        roi_resized = cv2.resize(roi, (100, 100))
        embeddings.append(_compute_histogram(roi_resized))

    cap.release()
    return embeddings


def cluster_faces(
    embeddings: List[np.ndarray],
    threshold: float = 0.6,
) -> List[int]:
    """
    Cluster face embeddings to identify unique characters.

    Uses AgglomerativeClustering with cosine distance. Returns a label
    for each embedding indicating which character cluster it belongs to.

    Args:
        embeddings: List of 1-D numpy arrays.
        threshold: Cosine distance threshold for merging clusters.
                   Lower = stricter (fewer clusters).

    Returns:
        List of integer cluster labels, one per embedding.
    """
    if not embeddings:
        return []

    X = np.array(embeddings)

    # Single embedding => single character
    if X.shape[0] == 1:
        return [0]

    sim = cosine_similarity(X)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    n_clusters = min(X.shape[0], max(1, int(1.0 / max(threshold, 1e-6))))
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist)
    return labels.tolist()


def recognize_characters(
    video_path: str,
    shots: List[Dict],
    num_frames: int = 16,
    threshold: float = 0.6,
) -> List[Dict]:
    """
    Full pipeline: detect faces across shots, cluster them, and assign
    character IDs.

    Args:
        video_path: Path to the video file.
        shots: List of shot dicts (``shot_id``, ``start_time``, ``end_time``).
        num_frames: Frames to sample per shot.
        threshold: Clustering distance threshold.

    Returns:
        List of dicts with keys:
            ``shot_id``, ``character_ids`` (sorted unique int labels),
            ``face_count`` (number of face embeddings extracted).
    """
    all_embeddings: List[np.ndarray] = []
    shot_embedding_counts: List[int] = []

    for shot in shots:
        embs = extract_face_embeddings(video_path, shot, num_frames=num_frames)
        shot_embedding_counts.append(len(embs))
        all_embeddings.extend(embs)

    if not all_embeddings:
        return [
            {"shot_id": s["shot_id"], "character_ids": [], "face_count": 0}
            for s in shots
        ]

    global_labels = cluster_faces(all_embeddings, threshold=threshold)

    # Map global labels back to per-shot slices
    results: List[Dict] = []
    offset = 0
    for shot, count in zip(shots, shot_embedding_counts):
        labels = global_labels[offset : offset + count]
        offset += count
        results.append({
            "shot_id": shot["shot_id"],
            "character_ids": sorted(set(labels)),
            "face_count": count,
        })

    return results
