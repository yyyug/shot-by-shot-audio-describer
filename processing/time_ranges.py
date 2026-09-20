"""User-selected time ranges (``hh:mm:ss-hh:mm:ss``) for targeted descriptions.

The UI offers three modes:

* ``full``  - describe the whole video (the original behaviour)
* ``extra`` - describe the whole video *and* the listed ranges
* ``only``  - describe the listed ranges and nothing else

Ranges are taken literally. A range is never snapped to shot boundaries, and
the dialogue-gap logic is deliberately not applied to one: a range that sits
on top of dialogue is still described, and its length is the length the user
asked for rather than the length of a nearby gap.

Ranges are also allowed to overlap the normal pass's units - ``extra`` keeps
both descriptions - so nothing here de-duplicates against them.
"""

RANGE_MODES = ("full", "extra", "only")

# "," is the documented separator; the rest are accepted so that a paste from a
# Chinese IME or a spreadsheet still works.
_LIST_SEPARATOR = ","
_LIST_ALIASES = ("，", ";", "；", "、", "\n", "\r")

# "-" is the documented separator, "至" covers the Chinese spelling of a range.
_DASH = "-"
_DASH_ALIASES = ("–", "—", "－", "~", "～", "至")


class RangeError(ValueError):
    """Raised with a message intended to be shown to the user as-is."""


def format_timestamp(seconds) -> str:
    """Format seconds as hh:mm:ss."""
    total = int(round(max(0.0, float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_ranges(ranges) -> str:
    """Canonical text for a parsed range list (what the input box shows back)."""
    return _LIST_SEPARATOR.join(
        f"{format_timestamp(r['start'])}-{format_timestamp(r['end'])}" for r in ranges
    )


def _parse_timestamp(token, whole):
    text = token.strip().replace("：", ":")
    if not text:
        raise RangeError(f"'{whole}': missing a start or end time.")
    parts = text.split(":")
    if len(parts) > 3:
        raise RangeError(f"'{text}' is not a valid time - use hh:mm:ss.")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise RangeError(f"'{text}' is not a valid time - use hh:mm:ss.")
    if any(v < 0 for v in values):
        raise RangeError(f"'{text}' is not a valid time - use hh:mm:ss.")
    if len(values) == 3:
        hours, minutes, seconds = values
    elif len(values) == 2:
        hours, minutes, seconds = 0.0, values[0], values[1]
    else:
        hours, minutes, seconds = 0.0, 0.0, values[0]
    # Minutes and seconds past 60 are carried over rather than rejected.
    return hours * 3600 + minutes * 60 + seconds


def parse_ranges(text, duration=None):
    """Parse and validate the user's range text.

    Returns ``[{"start": float, "end": float}]`` sorted by start time; that is
    also the order the units are described and written out in. ``duration``
    (when known) rejects ranges that end past the end of the video.
    """
    raw = str(text or "")
    for alias in _LIST_ALIASES:
        raw = raw.replace(alias, _LIST_SEPARATOR)
    for alias in _DASH_ALIASES:
        raw = raw.replace(alias, _DASH)

    chunks = [c.strip() for c in raw.split(_LIST_SEPARATOR)]
    chunks = [c for c in chunks if c]
    if not chunks:
        raise RangeError("Enter at least one time range as hh:mm:ss-hh:mm:ss.")

    ranges = []
    for chunk in chunks:
        bounds = [b.strip() for b in chunk.split(_DASH)]
        if len(bounds) != 2 or not all(bounds):
            raise RangeError(f"'{chunk}' is not a time range - use hh:mm:ss-hh:mm:ss.")
        start = _parse_timestamp(bounds[0], chunk)
        end = _parse_timestamp(bounds[1], chunk)
        # Sub-second precision is meaningless for a video segment, and the
        # canonical form has no fraction, so round here and never later.
        start = float(int(round(start)))
        end = float(int(round(end)))
        if start >= end:
            raise RangeError(f"'{chunk}': the start time must be earlier than the end time.")
        if duration and end > float(duration) + 0.5:
            raise RangeError(
                f"'{chunk}' ends after the video, which is only {format_timestamp(duration)} long."
            )
        ranges.append({"start": start, "end": end})

    ranges.sort(key=lambda r: r["start"])
    for earlier, later in zip(ranges, ranges[1:]):
        if later["start"] < earlier["end"]:
            raise RangeError(
                f"{format_timestamp(earlier['start'])}-{format_timestamp(earlier['end'])} and "
                f"{format_timestamp(later['start'])}-{format_timestamp(later['end'])} overlap."
            )
    return ranges


def validate_mode(mode, text, duration=None):
    """Normalise a range mode and parse its text.

    Returns ``(mode, ranges)``, where ``ranges`` is empty for the "full" mode.
    Raises :class:`RangeError` for an unknown mode or unusable text.
    """
    mode = str(mode or "full").strip().lower()
    if mode not in RANGE_MODES:
        raise RangeError(f"Unknown range mode '{mode}'.")
    if mode == "full":
        return mode, []
    return mode, parse_ranges(text, duration)


def shots_overlapping(shots, start, end):
    """Shot ids whose span intersects ``[start, end)``.

    Half-open, so a range that merely touches a shot boundary does not also
    claim the shot on the other side of it.
    """
    return [s["shot_id"] for s in shots if s["start_time"] < end and s["end_time"] > start]


def current_shot_indices(unit, shots):
    """0-based shot indices for the Stage-1 prompt.

    ``build_film_grammar_prompt`` renders these as "[Shot 1, Shot 2]". A unit
    that touches none of the detected shots would render "[]", which tells the
    model nothing, so fall back to the shot nearest the middle of the unit.
    """
    ids = unit.get("shot_ids") or []
    if not ids and shots:
        middle = (unit["start"] + unit["end"]) / 2.0
        nearest = min(
            shots, key=lambda s: abs((s["start_time"] + s["end_time"]) / 2.0 - middle)
        )
        ids = [nearest["shot_id"]]
    return [s - 1 for s in ids]


def build_custom_units(ranges, shots, first_index=1):
    """One unit per user range, spanning exactly what the user asked for."""
    units = []
    for i, rng in enumerate(ranges, start=first_index):
        units.append({
            "unit_id": f"C{i}",
            "start": rng["start"],
            "end": rng["end"],
            "shot_ids": shots_overlapping(shots, rng["start"], rng["end"]),
            "mode": "custom",
        })
    return units
