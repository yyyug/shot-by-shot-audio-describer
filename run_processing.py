#!/usr/bin/env python
# run_processing.py
import argparse
import os
import sys
import pandas as pd

from processing.shot_detector import detect_shots
from processing.whisper_transcriber import transcribe_video
from processing.vlm_describer import describe_frames
from processing.csv_merger import merge_shots_subtitles, merge_with_descriptions, format_original_style


def extract_frames_base64(video_path: str, shot: dict, num_frames: int = 8) -> list:
    """Extract frames from video shot and encode as base64."""
    import cv2
    import base64
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(shot["start_time"] * fps)
    end_frame = int(shot["end_time"] * fps)

    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    frames_b64 = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frames_b64.append(base64.b64encode(buffer).decode('utf-8'))

    cap.release()
    return frames_b64


def main():
    parser = argparse.ArgumentParser(
        description="Process video to generate shot-by-shot CSV"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", default="output.csv", help="Output CSV path")
    parser.add_argument("--whisper-path", default="whisper-cpp", help="Path to whisper-cpp binary")
    parser.add_argument("--model-path", default=None, help="Path to whisper model")
    parser.add_argument("--language", default=None, help="Language code for transcription")
    parser.add_argument("--openrouter-key", default=None, help="OpenRouter API key")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM description")
    parser.add_argument("--format", choices=["basic", "full", "original"], default="basic",
                       help="Output format: basic, full, or original")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    print(f"Processing: {args.video}")

    # Step 1: Shot detection
    print("Step 1/4: Detecting shots...")
    shots = detect_shots(args.video)
    print(f"  Found {len(shots)} shots")

    # Step 2: Transcription
    print("Step 2/4: Transcribing audio...")
    try:
        subtitles = transcribe_video(args.video, args.whisper_path, args.model_path, args.language)
        print(f"  Found {len(subtitles)} subtitle segments")
    except RuntimeError as e:
        print(f"  Warning: Transcription failed: {e}")
        print("  Continuing without subtitles...")
        subtitles = []

    # Step 3: Merge
    print("Step 3/4: Merging results...")
    merged_df = merge_shots_subtitles(shots, subtitles)

    # Step 4: VLM description (optional)
    if not args.skip_vlm and args.openrouter_key:
        print("Step 4/4: Generating VLM descriptions...")
        import numpy as np
        descriptions = []
        for shot in shots:
            try:
                frames_b64 = extract_frames_base64(args.video, shot)
                desc = describe_frames(frames_b64, args.openrouter_key)
                descriptions.append({"shot_id": shot["shot_id"], "description": desc})
                print(f"  Shot {shot['shot_id']}: OK")
            except Exception as e:
                print(f"  Shot {shot['shot_id']}: Failed - {e}")
                descriptions.append({"shot_id": shot["shot_id"], "description": ""})

        merged_df = merge_with_descriptions(merged_df, descriptions)
    else:
        print("Step 4/4: Skipping VLM description")

    # Format output
    if args.format == "original":
        output_df = format_original_style(merged_df)
    elif args.format == "full":
        output_df = merged_df
    else:  # basic
        output_df = merged_df[["shot_id", "start_time", "end_time", "subtitle"]]

    # Save
    output_df.to_csv(args.output, index=False)
    print(f"\nDone! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
