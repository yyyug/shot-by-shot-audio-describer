"""
WebVTT export for final Audio Description results.

Produces subtitle files loadable in VLC/mpv/browser players so the
generated AD can be previewed against the video directly.
"""
import logging

logger = logging.getLogger(__name__)


def seconds_to_vtt(seconds: float) -> str:
    """Format seconds as WebVTT timestamp HH:MM:SS.mmm."""
    total_ms = int(round(max(0.0, float(seconds)) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_vtt(rows) -> str:
    """Build WebVTT content from dicts with start/end/ad_sentence.

    Units with empty AD text are skipped: a blank cue is useless in a
    player and just clutters the subtitle track.
    """
    lines = ["WEBVTT", ""]
    count = 0
    for row in rows:
        text = str(row.get("ad_sentence") or "").strip()
        if not text:
            continue
        lines.append(f"{seconds_to_vtt(row['start'])} --> {seconds_to_vtt(row['end'])}")
        lines.append(text)
        lines.append("")
        count += 1
    logger.info(f"VTT built with {count} cues")
    return "\n".join(lines)


def write_vtt(path, rows) -> int:
    """Write WebVTT file next to the final CSV. Returns cue count."""
    content = build_vtt(rows)
    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write(content)
    return content.count("-->")
