#!/usr/bin/env python
# run_processing.py
import argparse
import os
import sys
import pandas as pd

from processing.shot_detector import detect_shots
from processing.context_extender import extend_context
from processing.whisper_transcriber import transcribe_video
from processing.film_grammar import get_effective_shot_scale, select_prompt_variant
from processing.vlm_describer import describe_frames, build_film_grammar_prompt
from processing.llm_summarizer import batch_summarize, estimate_word_limit
from processing.csv_merger import merge_shots_subtitles, merge_with_descriptions, format_original_style


def calculate_num_frames(duration_seconds: float) -> int:
    """Calculate optimal num_frames based on shot duration."""
    if duration_seconds <= 10:
        return 16  # Short shot
    elif duration_seconds <= 30:
        return 32  # Medium shot
    elif duration_seconds <= 60:
        return 48  # Long shot
    else:
        return 64  # Very long shot


def extract_frames_base64(video_path: str, shot: dict, num_frames: int = None) -> list:
    """Extract frames from video shot and encode as base64."""
    import cv2
    import base64
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Auto-calculate num_frames if not provided
    if num_frames is None:
        duration = shot["end_time"] - shot["start_time"]
        num_frames = calculate_num_frames(duration)

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
    parser.add_argument("--skip-whisper", action="store_true", help="Skip Whisper transcription")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM description")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip Stage 2 summarization")
    parser.add_argument("--video-type", choices=["movie", "tv_series"], default="movie",
                       help="Video type for verb list selection")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    print(f"Processing: {args.video}")

    # Step 1: Shot detection
    print("Step 1/6: Detecting shots...")
    shots = detect_shots(args.video)
    print(f"  Found {len(shots)} shots")

    # Step 2: Transcription
    if not args.skip_whisper:
        print("Step 2/6: Transcribing audio...")
        try:
            subtitles = transcribe_video(args.video, args.whisper_path, args.model_path, args.language)
            print(f"  Found {len(subtitles)} subtitle segments")
        except RuntimeError as e:
            print(f"  Warning: Transcription failed: {e}")
            print("  Continuing without subtitles...")
            subtitles = []
    else:
        print("Step 2/6: Skipping Whisper transcription")
        subtitles = []

    # Step 3: Merge shots and subtitles
    print("Step 3/6: Merging results...")
    merged_df = merge_shots_subtitles(shots, subtitles)

    # Step 4: VLM description (Stage 1)
    descriptions_dict = {}
    if not args.skip_vlm and args.openrouter_key:
        print("Step 4/6: Generating Stage 1 VLM descriptions...")
        import numpy as np
        
        # Get film grammar info for prompts
        shot_scales = [2] * len(shots)  # Default medium scale
        current_shots = list(range(len(shots)))  # All shots are current
        threads = [[i] for i in range(len(shots))]  # Each shot is its own thread
        
        # Calculate effective scale and prompt variant
        effective_scale = get_effective_shot_scale(shot_scales, current_shots)
        prompt_variant = select_prompt_variant(effective_scale)
        
        for shot in shots:
            try:
                frames_b64 = extract_frames_base64(args.video, shot)
                
                # Build film grammar prompt
                film_grammar = {
                    "video_type": args.video_type,
                    "label_type": "none",
                    "char_text": "",
                    "current_shots": [shot["shot_id"] - 1],
                    "threads": threads,
                    "shot_scales": shot_scales,
                    "prompt_variant": prompt_variant
                }
                
                desc = describe_frames(frames_b64, args.openrouter_key, film_grammar=film_grammar)
                descriptions_dict[shot["shot_id"]] = desc
                print(f"  Shot {shot['shot_id']}: OK")
            except Exception as e:
                print(f"  Shot {shot['shot_id']}: Failed - {e}")
                descriptions_dict[shot["shot_id"]] = ""

        # Add descriptions to merged DataFrame
        descriptions = [{"shot_id": k, "description": v} for k, v in descriptions_dict.items()]
        merged_df = merge_with_descriptions(merged_df, descriptions)
    else:
        print("Step 4/6: Skipping VLM description")

    # Step 5: Stage 2 summarization
    if not args.skip_stage2 and args.openrouter_key and descriptions_dict:
        print("Step 5/6: Generating Stage 2 summaries...")
        
        stage1_results = []
        for shot in shots:
            stage1_results.append({
                "shot_id": shot["shot_id"],
                "start": shot["start_time"],
                "end": shot["end_time"],
                "description": descriptions_dict.get(shot["shot_id"], "")
            })
        
        stage2_results = batch_summarize(
            stage1_results,
            args.openrouter_key,
            video_type=args.video_type
        )
        
        # Save Stage 1 output
        stage1_df = pd.DataFrame(stage1_results)
        stage1_output = args.output.replace(".csv", "_stage1.csv")
        stage1_df.to_csv(stage1_output, index=False)
        print(f"  Stage 1 saved to: {stage1_output}")
        
        # Save Stage 2 output
        stage2_df = pd.DataFrame(stage2_results)
        stage2_output = args.output.replace(".csv", "_stage2.csv")
        stage2_df.to_csv(stage2_output, index=False)
        print(f"  Stage 2 saved to: {stage2_output}")
    else:
        print("Step 5/6: Skipping Stage 2 summarization")

    # Step 6: Save final output (always full format)
    print("Step 6/6: Saving final output...")
    merged_df.to_csv(args.output, index=False)
    print(f"\nDone! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
