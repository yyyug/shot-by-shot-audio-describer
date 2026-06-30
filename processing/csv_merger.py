import pandas as pd
from typing import List, Dict


def merge_shots_subtitles(
    shots: List[Dict],
    subtitles: List[Dict]
) -> pd.DataFrame:
    """
    Merge shot boundaries with subtitles based on time overlap.

    Args:
        shots: List of dicts with keys: shot_id, start_time, end_time
        subtitles: List of dicts with keys: text, start_time, end_time

    Returns:
        DataFrame with columns: shot_id, start_time, end_time, subtitle
    """
    if not shots:
        return pd.DataFrame(columns=["shot_id", "start_time", "end_time", "subtitle"])

    shots_df = pd.DataFrame(shots)
    subs_df = pd.DataFrame(subtitles) if subtitles else pd.DataFrame(columns=["text", "start_time", "end_time"])

    results = []
    for _, shot in shots_df.iterrows():
        overlapping = subs_df[
            (subs_df["start_time"] < shot["end_time"]) &
            (subs_df["end_time"] > shot["start_time"])
        ]

        if len(overlapping) > 0:
            subtitle_text = " ".join(overlapping["text"].tolist())
        else:
            subtitle_text = ""

        results.append({
            "shot_id": shot["shot_id"],
            "start_time": shot["start_time"],
            "end_time": shot["end_time"],
            "subtitle": subtitle_text
        })

    return pd.DataFrame(results)


def merge_with_descriptions(
    merged_df: pd.DataFrame,
    descriptions: List[Dict]
) -> pd.DataFrame:
    """
    Add VLM descriptions to merged DataFrame.

    Args:
        merged_df: DataFrame from merge_shots_subtitles
        descriptions: List of dicts with keys: shot_id, description

    Returns:
        DataFrame with added video_description column
    """
    if not descriptions:
        merged_df["video_description"] = ""
        return merged_df

    desc_df = pd.DataFrame(descriptions)
    result = merged_df.merge(
        desc_df[["shot_id", "description"]],
        on="shot_id",
        how="left"
    )
    result["description"] = result["description"].fillna("")
    result = result.rename(columns={"description": "video_description"})

    return result


def format_original_style(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format DataFrame to match original project output format.

    Args:
        df: Merged DataFrame

    Returns:
        DataFrame with original project columns
    """
    result = pd.DataFrame()
    result["anno_idx"] = range(1, len(df) + 1)
    result["imdbid"] = "user_upload"
    result["start"] = df["start_time"]
    result["end"] = df["end_time"]
    result["text_gen"] = df.get("video_description", df["subtitle"])

    return result
