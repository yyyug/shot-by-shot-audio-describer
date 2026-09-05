"""
Dialogue gap detector - builds AD intervals from subtitle/transcript gaps.

Matches the original shot-by-shot repo behavior: one AD interval = one
sentence, placed in a pause (gap) between spoken segments. A single AD
interval can span one or more shots ("current shots").
"""


def detect_ad_intervals(
    subtitles,
    shots,
    min_gap=1.5,
    min_duration=1.0,
    video_duration=None
):
    """
    Find pauses in the transcript and turn them into AD intervals.

    Args:
        subtitles: List of dicts with keys text, start_time, end_time (seconds)
        shots: List of dicts with keys shot_id, start_time, end_time
        min_gap: Minimum silence length (seconds) to become an AD interval
        min_duration: Minimum AD interval duration (seconds) to keep
        video_duration: Total video duration (seconds), used for the trailing gap

    Returns:
        List of dicts with keys: ad_id, start, end, shot_ids
        (shot_ids = shots overlapping the gap, in order)
    """
    if not subtitles:
        return []

    segments = [
        s for s in subtitles
        if s.get("start_time") is not None and s.get("end_time") is not None
    ]
    segments.sort(key=lambda s: s["start_time"])
    if not segments:
        return []

    if video_duration is None:
        video_duration = max(s["end_time"] for s in segments)

    candidates = []

    # Leading gap (before first spoken segment)
    first_start = segments[0]["start_time"]
    if first_start >= min_gap:
        candidates.append((0.0, first_start))

    # Gaps between consecutive spoken segments
    for prev, nxt in zip(segments, segments[1:]):
        gap_start = prev["end_time"]
        gap_end = nxt["start_time"]
        if gap_end - gap_start >= min_gap:
            candidates.append((gap_start, gap_end))

    # Trailing gap (after last spoken segment)
    last_end = segments[-1]["end_time"]
    if video_duration - last_end >= min_gap:
        candidates.append((last_end, video_duration))

    intervals = []
    for ad_id, (gap_start, gap_end) in enumerate(candidates, start=1):
        overlap = [
            s for s in shots
            if max(gap_start, s["start_time"]) <= min(gap_end, s["end_time"])
        ]
        if not overlap:
            continue

        # Clamp the AD interval to the overlapping shots (original repo style)
        start = max(gap_start, min(s["start_time"] for s in overlap))
        end = min(gap_end, max(s["end_time"] for s in overlap))
        if end - start < min_duration:
            continue

        intervals.append({
            "ad_id": ad_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "shot_ids": [s["shot_id"] for s in overlap],
        })

    return intervals
