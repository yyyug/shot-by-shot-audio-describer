"""
Shot labeler - annotate video frames with shot numbers and character labels.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple


def add_shot_label(
    frame: np.ndarray,
    shot_idx: int,
    font_path: Optional[str] = None,
    font_size: int = 30,
) -> np.ndarray:
    """
    Add a 'Shot N' label to the top-left corner of a frame.

    Uses cv2.putText when no font_path is given, otherwise uses
    PIL/Pillow for TrueType font rendering.

    Args:
        frame: BGR image as numpy array.
        shot_idx: 1-based shot number to display.
        font_path: Optional path to a .ttf font file.
        font_size: Font size in pixels.

    Returns:
        Annotated frame (new array, original is not modified).
    """
    img = frame.copy()
    text = f"Shot {shot_idx}"

    if font_path:
        try:
            from PIL import Image, ImageDraw, ImageFont

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype(font_path, font_size)

            # Background rectangle for readability
            bbox = draw.textbbox((10, 10), text, font=font)
            padding = 4
            draw.rectangle(
                [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
                fill=(0, 0, 0),
            )
            draw.text((10, 10), text, font=font, fill=(255, 255, 255))

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except ImportError:
            pass  # Fall through to cv2.putText

    # Default: OpenCV built-in font
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = font_size / 30.0
    thickness = max(1, int(scale * 2))

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (8, 8), (8 + tw + 8, 8 + th + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 8 + th + 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img


def add_character_labels(
    frame: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    character_ids: List[int],
    colors: List[Tuple[int, int, int]],
) -> np.ndarray:
    """
    Draw colored bounding boxes and character IDs around detected faces.

    Args:
        frame: BGR image as numpy array.
        bboxes: List of (x, y, w, h) bounding boxes.
        character_ids: Character ID for each bounding box.
        colors: BGR color tuple for each bounding box.

    Returns:
        Annotated frame (new array, original is not modified).
    """
    img = frame.copy()

    for (x, y, w, h), char_id, color in zip(bboxes, character_ids, colors):
        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Draw character ID label above the box
        label = f"C{char_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

        label_y = max(y - 5, th + 4)
        cv2.rectangle(img, (x, label_y - th - 4), (x + tw + 4, label_y + 2), color, -1)
        cv2.putText(img, label, (x + 2, label_y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img


def label_frames(
    frames: List[np.ndarray],
    shot_idx: int,
    character_info: Optional[Dict] = None,
) -> List[np.ndarray]:
    """
    Process a list of frames by adding shot number labels and optionally
    character bounding box labels.

    Args:
        frames: List of BGR images as numpy arrays.
        shot_idx: 1-based shot number.
        character_info: Optional dict with keys:
            - 'bboxes': List of (x, y, w, h) per frame (list of lists).
            - 'character_ids': List of lists of ints per frame.
            - 'colors': List of BGR color tuples per character ID.
            If provided, character labels are drawn on each frame.

    Returns:
        List of annotated frames.
    """
    if not frames:
        return []

    labeled = []
    for i, frame in enumerate(frames):
        img = add_shot_label(frame, shot_idx)

        if character_info:
            bboxes_list = character_info.get("bboxes", [])
            ids_list = character_info.get("character_ids", [])
            palette = character_info.get("colors", [])

            if i < len(bboxes_list) and i < len(ids_list):
                bboxes = bboxes_list[i]
                ids = ids_list[i]
                colors = [palette[c % len(palette)] if palette else (0, 255, 0) for c in ids]
                img = add_character_labels(img, bboxes, ids, colors)

        labeled.append(img)

    return labeled
