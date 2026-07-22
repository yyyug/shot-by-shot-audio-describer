"""
Context extension module - implements context extension logic from original repo
"""
from typing import List, Dict
import numpy as np


def find_overlapping_intervals(ad_start: float, ad_end: float, shots: List[Dict]) -> List[int]:
    """
    Find shot indices that overlap with the AD interval.
    
    Args:
        ad_start: AD interval start time
        ad_end: AD interval end time
        shots: List of dicts with keys: start_time, end_time
        
    Returns:
        List of overlapping shot indices
    """
    overlapping = []
    for i, shot in enumerate(shots):
        if max(ad_start, shot["start_time"]) <= min(ad_end, shot["end_time"]):
            overlapping.append(i)
    return overlapping


def extend_context(
    shots: List[Dict],
    ad_start: float,
    ad_end: float,
    internal_range: float = 3.0,
    external_range: float = 8.0,
    min_shot_duration: float = 0.4
) -> List[Dict]:
    """
    Extend shot context to include 2 past + current + 2 future shots.
    
    Args:
        shots: List of dicts with keys: shot_id, start_time, end_time
        ad_start: AD interval start time
        ad_end: AD interval end time
        internal_range: Max extension of current shots from AD boundaries
        external_range: Max duration of each past/future shot
        min_shot_duration: Minimum shot duration to include
        
    Returns:
        List of dicts with added shot_label key (l-0, l-1, m-0, r-0, r-1)
    """
    if not shots:
        return []
    
    current_indices = find_overlapping_intervals(ad_start, ad_end, shots)
    
    if not current_indices:
        mid_point = (ad_start + ad_end) / 2
        distances = [abs((s["start_time"] + s["end_time"]) / 2 - mid_point) for s in shots]
        closest_idx = distances.index(min(distances))
        current_indices = [closest_idx]
    
    first_current = current_indices[0]
    last_current = current_indices[-1]
    
    extended_start = max(0, ad_start - internal_range)
    extended_end = ad_end + internal_range
    
    past_indices = []
    for i in range(first_current - 1, max(-1, first_current - 3), -1):
        if i >= 0:
            shot = shots[i]
            duration = shot["end_time"] - shot["start_time"]
            if duration >= min_shot_duration and shot["end_time"] >= extended_start:
                past_indices.insert(0, i)
            else:
                break
    
    future_indices = []
    for i in range(last_current + 1, min(len(shots), last_current + 3)):
        shot = shots[i]
        duration = shot["end_time"] - shot["start_time"]
        if duration >= min_shot_duration and shot["start_time"] <= extended_end:
            future_indices.append(i)
        else:
            break
    
    result = []
    
    for idx in reversed(past_indices):
        shot = shots[idx].copy()
        label_idx = past_indices.index(idx)
        shot["shot_label"] = f"l-{len(past_indices) - 1 - label_idx}"
        result.append(shot)
    
    for idx in current_indices:
        shot = shots[idx].copy()
        label_idx = idx - first_current
        shot["shot_label"] = f"m-{label_idx}"
        result.append(shot)
    
    for idx in future_indices:
        shot = shots[idx].copy()
        label_idx = future_indices.index(idx)
        shot["shot_label"] = f"r-{label_idx}"
        result.append(shot)
    
    return result
