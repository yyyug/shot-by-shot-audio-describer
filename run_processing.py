#!/usr/bin/env python
# run_processing.py
import argparse
import os
import sys
import pandas as pd

from processing.shot_detector import detect_shots
from processing.sensevoice_transcriber import transcribe_video
from processing.dialogue_gap_detector import detect_ad_intervals
from processing.film_grammar import get_effective_shot_scale, select_prompt_variant
from processing.vlm_describer import describe_frames, build_film_grammar_prompt
from processing.llm_summarizer import batch_summarize, estimate_word_limit


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


def _get_context_shots(shots, current_shot_ids):
    """Get context shots: 2 before + current shots + 2 after."""
    if not current_shot_ids:
        current_shot_ids = [shots[0]["shot_id"]]
    idxs = [s["shot_id"] for s in shots]
    first_idx = idxs.index(current_shot_ids[0])
    last_idx = idxs.index(current_shot_ids[-1])
    context = []
    for i in range(max(0, first_idx - 2), first_idx):
        context.append(shots[i])
    for i in range(first_idx, last_idx + 1):
        context.append(shots[i])
    for i in range(last_idx + 1, min(len(shots), last_idx + 3)):
        context.append(shots[i])
    return context


def _extract_frames_with_context(video_path: str, current_shot: dict, context_shots: list) -> list:
    """Extract frames from current shot + context shots."""
    import cv2
    import base64
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    all_frames_b64 = []
    for shot in context_shots:
        duration = shot["end_time"] - shot["start_time"]
        num_frames = calculate_num_frames(duration)
        start_frame = int(shot["start_time"] * fps)
        end_frame = int(shot["end_time"] * fps)
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                all_frames_b64.append(base64.b64encode(buffer).decode('utf-8'))
    cap.release()
    return all_frames_b64


def main():
    parser = argparse.ArgumentParser(
        description="Process video to generate shot-by-shot CSV"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", default="output.csv", help="Output CSV path")
    parser.add_argument("--language", default=None, help="Language code for transcription")
    parser.add_argument("--api-key", default=None, help="API key for the selected backend")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip transcription (shot-based mode)")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM description")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip Stage 2 summarization")
    parser.add_argument("--context", action="store_true",
                        help="Include 2 surrounding shots (before/after) in VLM descriptions")
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

    # Step 2: Transcription (SenseVoice)
    subtitles = []
    if not args.skip_whisper:
        print("Step 2/6: Transcribing audio with SenseVoice...")
        try:
            subtitles = transcribe_video(args.video, language=args.language or "auto")
            print(f"  Found {len(subtitles)} subtitle segments")
        except RuntimeError as e:
            print(f"  Warning: Transcription failed: {e}")
            print("  Continuing in shot-based mode...")
            subtitles = []
    else:
        print("Step 2/6: Skipping transcription")

    # Step 3: Build AD units (dialogue gaps when transcribed, else shots)
    print("Step 3/6: Building AD units...")
    units = []
    if subtitles:
        ad_intervals = detect_ad_intervals(subtitles, shots)
        print(f"  Detected {len(ad_intervals)} dialogue-gap AD intervals")
        for interval in ad_intervals:
            units.append({
                "unit_id": interval["ad_id"],
                "start": interval["start"],
                "end": interval["end"],
                "shot_ids": interval["shot_ids"],
            })
    if not units:
        for shot in shots:
            units.append({
                "unit_id": shot["shot_id"],
                "start": shot["start_time"],
                "end": shot["end_time"],
                "shot_ids": [shot["shot_id"]],
            })
    if not units:
        import cv2
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        end_t = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps > 0 else 0
        cap.release()
        print("  No shots detected; using full video as single unit")
        units.append({
            "unit_id": 1,
            "start": 0,
            "end": end_t,
            "shot_ids": [],
        })

    # Step 4: VLM description (Stage 1)
    descriptions_dict = {}
    if not args.skip_vlm and args.api_key:
        print(f"Step 4/6: Generating Stage 1 VLM descriptions ({len(units)} units)...")
        import numpy as np
        
        # Get film grammar info for prompts
        shot_scales = [2] * len(shots)  # Default medium scale
        threads = [[i] for i in range(len(shots))]  # Each shot is its own thread
        
        # Calculate effective scale and prompt variant
        current_shots = list(range(len(shots)))  # All shots are current
        effective_scale = get_effective_shot_scale(shot_scales, current_shots)
        prompt_variant = select_prompt_variant(effective_scale)
        
        for unit in units:
            try:
                frame_shot = {"start_time": unit["start"], "end_time": unit["end"]}
                if args.context:
                    context_shots = _get_context_shots(shots, unit["shot_ids"])
                    frames_b64 = _extract_frames_with_context(args.video, frame_shot, context_shots)
                else:
                    frames_b64 = extract_frames_base64(args.video, frame_shot)
                
                # Build film grammar prompt
                film_grammar = {
                    "video_type": args.video_type,
                    "label_type": "none",
                    "char_text": "",
                    "current_shots": [s - 1 for s in unit["shot_ids"]],
                    "threads": threads,
                    "shot_scales": shot_scales,
                    "prompt_variant": prompt_variant
                }
                
                desc = describe_frames(frames_b64, args.api_key, film_grammar=film_grammar)
                descriptions_dict[unit["unit_id"]] = desc
                print(f"  Unit {unit['unit_id']}: OK")
            except Exception as e:
                print(f"  Unit {unit['unit_id']}: Failed - {e}")
                descriptions_dict[unit["unit_id"]] = ""
    else:
        print("Step 4/6: Skipping VLM description")

    # Step 5: Stage 2 summarization
    ad_sentence_map = {}
    if not args.skip_stage2 and args.api_key and descriptions_dict:
        print("Step 5/6: Generating Stage 2 summaries...")
        
        stage1_results = []
        for unit in units:
            stage1_results.append({
                "shot_id": unit["unit_id"],
                "start": unit["start"],
                "end": unit["end"],
                "description": descriptions_dict.get(unit["unit_id"], "")
            })
        
        stage2_results = batch_summarize(
            stage1_results,
            args.api_key,
            video_type=args.video_type
        )
        ad_sentence_map = {r["shot_id"]: r["ad_sentence"] for r in stage2_results}
        
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

    # Step 6: Save final output (shot_id, start, end, ad_sentence)
    print("Step 6/6: Saving final output...")
    output_df = pd.DataFrame([{
        "shot_id": unit["unit_id"],
        "start": unit["start"],
        "end": unit["end"],
        "ad_sentence": ad_sentence_map.get(unit["unit_id"], "")
    } for unit in units])
    output_df.to_csv(args.output, index=False)
    print(f"\nDone! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
