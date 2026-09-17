"""
Flask web server for Shot-by-Shot Video Processor
"""
import os
import sys
import uuid
import json
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Response, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from processing.shot_detector import detect_shots
from processing.sensevoice_transcriber import transcribe_video
from processing.dialogue_gap_detector import detect_ad_intervals
from processing.vlm_describer import describe_frames
from processing.llm_summarizer import batch_summarize, estimate_word_limit, test_connection
from processing.film_grammar import get_effective_shot_scale, select_prompt_variant
from processing.character_recognizer import detect_faces, extract_face_embeddings, cluster_faces
from processing.vtt_writer import write_vtt
from processing import history_engine as history

# Lazy import to mirror vlm_describer/llm_summarizer (also keeps the frozen
# PyArmor builds from resolving a private module at import time).
def _api_common():
    import processing._api_common as m
    return m

# Resolve resource folders for both source and PyInstaller-frozen layouts.
# Mirrors desktop.py exactly so both versions write outputs to the same place.
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
FROZEN = bool(getattr(sys, "frozen", False))
if FROZEN:
    # Packaged app: keep user data in a writable location (Program Files is read-only)
    DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "BuddyAd")
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
CORS(app)

app.config['UPLOAD_FOLDER'] = os.path.join(DATA_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging parity with the desktop version: processing errors must land in a
# file so remote failures can be diagnosed (webapp_entry.py already points
# basicConfig at the same file when frozen; this also covers `python app.py`).
# The file handler rotates at 5MB (keeps 3 backups) so no single log grows
# out of control; 30-day age cleanup above retires stale files.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(DATA_DIR, 'shot_by_shot_web.log'),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8',
        ),
        logging.StreamHandler(),
    ],
)

# Logs rotate at 5MB per file (3 backups: .1/.2/.3); anything older than 30
# days is deleted on startup so the log directory cannot grow without bound.
def _cleanup_old_logs(log_dir, max_age_days=30, pattern="*.log"):
    try:
        cutoff = datetime.now().timestamp() - max_age_days * 86400
        for name in os.listdir(log_dir):
            if ".log" not in name.lower():
                continue
            path = os.path.join(log_dir, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                pass
    except OSError:
        pass


_cleanup_old_logs(DATA_DIR)

logger = logging.getLogger(__name__)

processing_status = {}
# Tracks the single running task id, if any. Only one task may process at a
# time: the free-tier Gemini quota is per-account-per-day, and concurrent
# tasks exhaust it in minutes (each makes dozens of API calls). Rejecting a
# new upload while one is running is intentional, not a UI restriction.
_active_task_id = None
_active_task_lock = threading.Lock()


def _try_acquire_task(task_id: str) -> bool:
    """Atomically claim the processing slot. Returns False if busy."""
    global _active_task_id
    with _active_task_lock:
        if _active_task_id is not None:
            return False
        _active_task_id = task_id
        return True


def _release_task(task_id: str):
    global _active_task_id
    with _active_task_lock:
        if _active_task_id == task_id:
            _active_task_id = None


def build_character_bank(video_path, shots, num_frames=16, threshold=0.6):
    """Detect faces across shots, cluster into characters."""
    all_embeddings = []
    shot_embedding_counts = []
    for shot in shots:
        embs = extract_face_embeddings(video_path, shot, num_frames=num_frames)
        shot_embedding_counts.append(len(embs))
        all_embeddings.extend(embs)
    if not all_embeddings:
        return {}, {}
    global_labels = cluster_faces(all_embeddings, threshold)
    unique_chars = sorted(set(global_labels))
    char_map = {c: f"Character_{chr(65+c)}" for c in unique_chars}
    char_lookup = {}
    offset = 0
    for shot, count in zip(shots, shot_embedding_counts):
        labels = global_labels[offset:offset+count]
        offset += count
        char_lookup[shot["shot_id"]] = [char_map[l] for l in sorted(set(labels))]
    return char_lookup, char_map


def calculate_num_frames(duration_seconds):
    if duration_seconds <= 10: return 16
    elif duration_seconds <= 30: return 32
    elif duration_seconds <= 60: return 48
    else: return 64


def extract_frames_base64(video_path, shot, num_frames=None, shot_idx=None):
    import cv2, base64
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
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
            if shot_idx is not None:
                label = f"Shot {shot_idx}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 1, 2)
                cv2.rectangle(frame, (5, 5), (10+tw, 10+th+5), (255,255,255), -1)
                cv2.putText(frame, label, (10, 10+th), font, 1, (0,0,0), 2)
            _, buf = cv2.imencode('.jpg', frame)
            frames_b64.append(base64.b64encode(buf).decode('utf-8'))
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


def _extract_frames_with_context(video_path, current_shot, context_shots):
    """Extract frames from current shot + context shots."""
    import cv2, base64
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    all_frames_b64 = []
    parts = []
    for shot in context_shots:
        duration = shot["end_time"] - shot["start_time"]
        num_frames = calculate_num_frames(duration)
        is_current = (
            abs(shot.get("start_time", -1) - current_shot.get("start_time", 0)) < 1e-6
            and abs(shot.get("end_time", -1) - current_shot.get("end_time", 0)) < 1e-6
        )
        parts.append(
            f"{'current' if is_current else 'context'}=#{shot.get('shot_id')}({duration:.1f}s,{num_frames}f)"
        )
        start_frame = int(shot["start_time"] * fps)
        end_frame = int(shot["end_time"] * fps)
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode('.jpg', frame)
                all_frames_b64.append(base64.b64encode(buf).decode('utf-8'))
    cap.release()
    logger.info("extract_frames_with_context: shot=%s duration=%.1fs frames=%d parts=[%s]",
                current_shot.get("start_time"), current_shot["end_time"] - current_shot["start_time"],
                len(all_frames_b64), ", ".join(parts))
    return all_frames_b64


def process_video_task(task_id, video_path, options):
    status = {"status": "processing", "step": "initializing", "progress": 0}
    processing_status[task_id] = status
    if not _try_acquire_task(task_id):
        # Should be unreachable (upload rejects busy), but guard anyway.
        logger.warning(f"Task {task_id}: refused to start - another task is already processing")
        status["status"] = "failed"
        status["error"] = "Another task is already processing. Wait for it to finish."
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        backend = options.get("backend", "gemini")
        api_key = options.get("api_key") or options.get("gemini_key")        
        filename = os.path.basename(video_path)
        if filename.startswith(task_id + "_"):
            filename = filename[len(task_id) + 1:]
        history.create_job(
            task_id, filename, DATA_DIR,
            video_type=options.get("video_type", "movie"),
            gap_detection_enabled=bool(options.get("use_whisper", True)),
            use_context_extender=bool(options.get("use_context_extender", False)),
        )
        # The source video is deliberately NOT kept: the exact LLM payload
        # (per-unit frames + rendered prompts) is persisted under
        # history/<job_id>/ instead, so replays and external agents never need
        # the video again.
        logger.info(f"Task {task_id}: source video not archived (frames will be saved)")
        # Get video duration for progress display
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        status["step"] = "shot_detection"
        status["detail"] = f"Detecting shots... ({video_duration:.0f}s video)"
        status["progress"] = 10

        def shot_progress(progress, message):
            status["detail"] = f"Detecting shots: {message}"
            status["progress"] = 10 + progress * 10

        shots = detect_shots(video_path, callback=shot_progress)
        status["shots_count"] = len(shots)
        history.save_shots(task_id, shots, DATA_DIR)
        logger.info(f"Task {task_id}: {len(shots)} shots detected")
        status["progress"] = 20
        status["detail"] = f"Found {len(shots)} shots"
        status["step"] = "transcription"
        subtitles = []
        transcription_error = None
        if options.get("use_whisper", True):
            def trans_progress(progress, message):
                status["detail"] = f"Transcription: {message}"
                status["progress"] = 20 + progress * 10
            try:
                subtitles = transcribe_video(video_path, language="auto", callback=trans_progress)
                status["detail"] = f"Transcribed {len(subtitles)} segments"
                logger.info(f"Task {task_id}: transcribed {len(subtitles)} segments")
            except Exception as e:
                subtitles = []
                transcription_error = e
                status["detail"] = f"Transcription failed ({type(e).__name__}), falling back to shot-based mode"
                logger.warning(f"Task {task_id}: transcription failed ({type(e).__name__}: {e}), shot-based mode", exc_info=True)
        else:
            status["detail"] = "Skipped"
            logger.info(f"Task {task_id}: transcription disabled by user")
        status["progress"] = 30
        history.save_subtitles(task_id, subtitles, DATA_DIR)

        # Build AD units: dialogue-gap intervals when transcription succeeded,
        # otherwise one unit per shot (existing behavior)
        units = []
        gap_detection_active = False
        if subtitles:
            ad_intervals = detect_ad_intervals(subtitles, shots, video_duration=video_duration)
            gap_detection_active = True
            status["detail"] = f"Detected {len(ad_intervals)} dialogue-gap AD intervals"
            logger.info(f"Task {task_id}: dialogue-gap detection ACTIVE - {len(ad_intervals)} intervals from {len(subtitles)} subtitle segments")
            for interval in ad_intervals:
                units.append({
                    "unit_id": interval["ad_id"],
                    "start": interval["start"],
                    "end": interval["end"],
                    "shot_ids": interval["shot_ids"],
                    "mode": "ad_interval",
                })
        if not units:
            for shot in shots:
                units.append({
                    "unit_id": shot["shot_id"],
                    "start": shot["start_time"],
                    "end": shot["end_time"],
                    "shot_ids": [shot["shot_id"]],
                    "mode": "shot",
                })
        if not units:
            units.append({
                "unit_id": 1,
                "start": 0,
                "end": video_duration,
                "shot_ids": [],
                "mode": "shot",
            })
            logger.info(f"Task {task_id}: no shots detected; using full-video as single unit ({video_duration:.1f}s)")
        if not gap_detection_active:
            reason = "transcription disabled" if not options.get("use_whisper", True) else f"transcription failed ({type(transcription_error).__name__})" if transcription_error else "no subtitles produced"
            logger.warning(f"Task {task_id}: dialogue-gap detection NOT active - {reason}. Using {len(units)} per-shot units instead.")
        status["detail"] = f"{len(units)} AD units to describe ({'dialogue-gap' if gap_detection_active else 'shot-based'})"
        status["progress"] = 40
        logger.info(f"Task {task_id}: built {len(units)} AD units (mode={'dialogue-gap' if gap_detection_active else 'shot-based'})")
        history.save_units(task_id, units, DATA_DIR)
        history.begin_run(task_id, 0, DATA_DIR, kind="initial",
                          gap_detection_active=gap_detection_active)
        history.set_current_run(task_id, 0, DATA_DIR)

        # Persist the exact image payload for every unit up-front. Doing it once
        # here (instead of inside the description loop) means a replay loads
        # these files rather than re-running shot detection, transcription or
        # frame extraction - and the payload survives even if the API fails.
        shot_scales = [2] * len(shots)
        threads = [[j for j in range(len(shots))]]
        frame_counts = {}
        status["step"] = "frames"
        for i, unit in enumerate(units):
            frame_shot = {"start_time": unit["start"], "end_time": unit["end"]}
            if options.get("use_context_extender") and shots:
                context_shots = _get_context_shots(shots, unit["shot_ids"])
                frames_b64 = _extract_frames_with_context(video_path, frame_shot, context_shots)
            else:
                frames_b64 = extract_frames_base64(video_path, frame_shot)
            history.save_unit_frames(task_id, unit["unit_id"], frames_b64, DATA_DIR)
            history.save_thumbs_from_unit_frames(task_id, unit, frames_b64, DATA_DIR)
            frame_counts[str(unit["unit_id"])] = len(frames_b64)
            status["detail"] = f"Saving frames {i+1}/{len(units)}"
            status["progress"] = 40 + int(((i + 1) / len(units)) * 5)
        history.save_frame_manifest(
            task_id, DATA_DIR,
            backend=backend,
            model=options.get("openai_model") or "",
            use_context_extender=bool(options.get("use_context_extender")),
            speech_transcription=bool(subtitles),
            frames_per_unit=frame_counts,
        )
        history.write_agent_jsonl(task_id, DATA_DIR, units)
        logger.info(f"Task {task_id}: saved frames for {len(units)} units "
                    f"({sum(frame_counts.values())} images)")

        # Character detection (optional) - parity with desktop: stub only
        if options.get("use_character_bank"):
            status["step"] = "character_bank"
            status["progress"] = 50

        descriptions_dict = {}
        # Usage trackers are referenced below even when no API key is provided
        # (output-only runs), so they must exist in every path.
        stage1_usage = _api_common().TokenUsage()
        stage2_usage = _api_common().TokenUsage()
        if not api_key:
            logger.warning(f"Task {task_id}: no API key provided - VLM descriptions and Stage 2 will be SKIPPED (output CSVs will be empty)")
        if api_key:
            status["step"] = "stage1_vlm"
            status["detail"] = "Starting VLM descriptions..."
            logger.info(f"Task {task_id}: VLM descriptions begin - {len(units)} units, backend={backend}, context={bool(options.get('use_context_extender'))}")
            stage1_error_categories = {}
            for i, unit in enumerate(units):
                try:
                    logger.info(f"Task {task_id}: unit {i+1}/{len(units)} ({unit.get('mode', 'shot')})")
                    # Replay the exact frames persisted above; the prompt is
                    # rebuilt deterministically from the unit + run options.
                    frames_b64 = history.load_unit_frames(task_id, unit["unit_id"], DATA_DIR)
                    film_grammar = {"video_type": options.get("video_type", "movie"), "label_type": "none",
                                    "char_text": "", "current_shots": [s - 1 for s in unit["shot_ids"]],
                                    "threads": threads, "shot_scales": shot_scales, "prompt_variant": 4,
                                    "custom_opening": options.get("custom_opening")}
                    logger.info(f"Task {task_id}: calling API for unit {i+1} ({len(frames_b64)} frames)")
                    desc = describe_frames(frames_b64, api_key, backend=backend, film_grammar=film_grammar,
                                           openai_url=options.get("openai_url"),
                                           openai_model=options.get("openai_model"),
                                           usage_acc=stage1_usage)
                    descriptions_dict[unit["unit_id"]] = desc
                    history.record_unit(task_id, 0, DATA_DIR, unit, desc)
                    logger.info(f"Task {task_id}: unit {i+1} completed ({len(desc)} chars)")
                    status["detail"] = f"Described unit {i+1}/{len(units)}"
                except Exception as e:
                    descriptions_dict[unit["unit_id"]] = ""
                    history.record_unit(task_id, 0, DATA_DIR, unit, "")
                    status["detail"] = f"Unit {i+1} failed: {str(e)[:50]}"
                    logger.error(f"Task {task_id}: unit {i+1}/{len(units)} failed: {e}", exc_info=True)
                    if e is not None:
                        cat = _api_common().categorize_error(e)
                        stage1_error_categories[cat] = stage1_error_categories.get(cat, 0) + 1
                # Rate limit protection: stay under ~15 RPM
                if i < len(units) - 1:
                    time.sleep(3)
                status["progress"] = 50 + int((i/len(units))*20)
        status["detail"] = f"Described {len(descriptions_dict)} units"
        status["progress"] = 80
        if stage1_usage.calls:
            logger.info(f"Task {task_id}: Stage 1 token usage: {stage1_usage.summary()}")

        token_usage_status = {}
        if stage1_usage.calls:
            token_usage_status["stage1"] = stage1_usage.fields()
        status["token_usage"] = token_usage_status or None

        # Stage-1 results are always persisted when there is anything to keep:
        # some shots may have succeeded even when the API quota ran out, and the
        # user explicitly wants those partial descriptions preserved.
        stage1_results = [{"shot_id": u["unit_id"], "start": u["start"], "end": u["end"],
                           "description": descriptions_dict.get(u["unit_id"], "")} for u in units]
        success_count = sum(1 for d in descriptions_dict.values() if str(d).strip())
        has_any_description = success_count > 0

        if has_any_description:
            pd.DataFrame(stage1_results).to_csv(
                os.path.join(DATA_DIR, f"{timestamp}_DetailsDescription.csv"),
                index=False, encoding="utf-8-sig")
            status["stage1_path"] = f"{timestamp}_DetailsDescription.csv"
            logger.info(f"Task {task_id}: wrote stage 1 details (non-empty descriptions: {success_count}/{len(stage1_results)})")

        ad_sentence_map = {}
        if not options.get("skip_stage2") and api_key and descriptions_dict:
            # Abort stage 2 when the majority of shots could not be described.
            # Only the message wording (and text) mentions quota - the actual
            # cause could be auth/config/network, which we now diagnose from
            # the errors collected during stage 1.
            if len(units) > 0 and success_count / len(units) < 0.5:
                status["status"] = "failed"
                status["partial"] = True
                _ac = _api_common()
                dominant = _ac.dominant_category(stage1_error_categories)
                status["error"] = _ac.build_stage1_blocked_message(
                    dominant, success_count, len(units)
                )
                logger.warning(f"Task {task_id}: aborting stage 2 - only {success_count}/{len(units)} shots succeeded (cause={dominant})")
            else:
                status["step"] = "stage2_summarize"
                status["detail"] = "Summarizing audio descriptions..."
                logger.info(f"Task {task_id}: stage 2 summarization begin")
                try:
                    stage2_results = batch_summarize(stage1_results, api_key, backend=backend,
                                                video_type=options.get("video_type", "movie"),
                                                openai_url=options.get("openai_url"),
                                                openai_model=options.get("openai_model"),
                                                usage_acc=stage2_usage)
                    ad_sentence_map = {r["shot_id"]: r["ad_sentence"] for r in stage2_results}
                    non_empty = sum(1 for v in ad_sentence_map.values() if str(v).strip())
                    logger.info(f"Task {task_id}: stage 2 produced {non_empty}/{len(ad_sentence_map)} non-empty AD sentences")
                    grand = stage1_usage.total() + stage2_usage.total()
                    logger.info(f"Task {task_id}: token usage totals: stage1=[{stage1_usage.summary()}] stage2=[{stage2_usage.summary()}] grand_total={grand}")
                    if stage2_usage.calls:
                        token_usage_status["stage2"] = stage2_usage.fields()
                    token_usage_status["grand_total"] = grand
                    status["token_usage"] = token_usage_status or None
                    # Same naming scheme and folder as the desktop version so the
                    # "Outputs" shortcut shows results from both versions.
                    pd.DataFrame(stage2_results).to_csv(os.path.join(DATA_DIR, f"{timestamp}_AD.csv"), index=False, encoding="utf-8-sig")
                    status["stage2_path"] = f"{timestamp}_AD.csv"
                    status["status"] = "completed"
                    status["progress"] = 100
                    output_df = pd.DataFrame([{
                        "shot_id": u["unit_id"],
                        "start": u["start"],
                        "end": u["end"],
                        "ad_sentence": ad_sentence_map.get(u["unit_id"], "")
                    } for u in units])
                    output_df.to_csv(os.path.join(DATA_DIR, f"{timestamp}-final.csv"), index=False, encoding="utf-8-sig")
                    status["output_path"] = f"{timestamp}-final.csv"

                    # WebVTT sidecar so the AD can be previewed in any player
                    vtt_filename = f"{timestamp}-final.vtt"
                    write_vtt(os.path.join(DATA_DIR, vtt_filename), output_df.to_dict("records"))
                    status["vtt_path"] = vtt_filename

                    # History snapshot for this (initial) run. Root copies are
                    # already written above, so no mirroring is needed here.
                    history.save_stage2(task_id, 0, DATA_DIR, ad_sentence_map)
                    history.write_run_files(
                        task_id, 0, DATA_DIR, stage1_results, stage2_results,
                        ad_sentence_map, units, mirror_root=False)
                    history.update_run(task_id, 0, DATA_DIR,
                                       status="completed", stage1_count=success_count,
                                       stage2_count=len(ad_sentence_map))

                    logger.info(f"Task {task_id}: completed - outputs {timestamp}_DetailsDescription.csv / {timestamp}_AD.csv / {timestamp}-final.csv / {timestamp}-final.vtt")
                except Exception as stage2_err:
                    # Stage 2 failed (e.g. API quota exhausted mid-run, or an
                    # auth/config/network problem). Diagnose the actual cause
                    # instead of always blaming free-tier quota. Stage-1
                    # results are already saved, so keep them downloadable and
                    # mark the run as failed with a clear message.
                    _ac = _api_common()
                    status["status"] = "failed"
                    category = _ac.categorize_error(stage2_err)
                    status["error"] = _ac.build_stage2_failed_message(
                        category, _ac.quote_error_detail(stage2_err)
                    )
                    status["partial"] = True
                    history.update_run(task_id, 0, DATA_DIR,
                                       status="failed", stage1_count=success_count)
                    logger.error(f"Task {task_id}: stage 2 failed (cause={category}) but stage-1 results kept: {stage2_err}")
        else:
            # No API key or stage 2 explicitly skipped: only stage-1 CSV exists.
            status["status"] = "completed"
            status["progress"] = 100
            history.write_run_files(task_id, 0, DATA_DIR, stage1_results, None,
                                    {}, units, mirror_root=False)
            history.update_run(task_id, 0, DATA_DIR,
                               status="completed", stage1_count=success_count)
            logger.info(f"Task {task_id}: completed (stage 2 not run) - outputs {timestamp}_DetailsDescription.csv")
    except Exception as e:
        status["status"] = "failed"
        status["error"] = str(e)
        logger.critical(f"Task {task_id}: FAILED at step={status.get('step')} progress={status.get('progress')}: {e}", exc_info=True)
    finally:
        # The upload is only needed while this run executes: the replayable
        # payload now lives under history/<job_id>/frames, so drop the video.
        try:
            upload_root = os.path.abspath(app.config['UPLOAD_FOLDER'])
            if os.path.isfile(video_path) and os.path.abspath(video_path).startswith(upload_root + os.sep):
                os.remove(video_path)
                logger.info(f"Task {task_id}: removed uploaded source video")
        except OSError as e:
            logger.warning(f"Task {task_id}: could not remove uploaded video: {e}")
        _release_task(task_id)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data_dir')
def api_data_dir():
    return jsonify({"path": DATA_DIR})

@app.route('/test_api', methods=['POST'])
def test_api_route():
    data = request.get_json(silent=True) or {}
    ok, message = test_connection(
        data.get("api_key"),
        data.get("backend", "gemini-3.7-flash"),
        data.get("openai_url"),
        data.get("openai_model"),
    )
    logger.info(f"API key test: backend={data.get('backend')} url={data.get('openai_url')} model={data.get('openai_model')} ok={ok} - {message}")
    return jsonify({"ok": ok, "message": message})


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Reject while another task is processing. The free-tier Gemini quota is
    # per-account-per-day and concurrent tasks drain it in minutes (root cause
    # of the empty-CSV incidents). Refusing here is the intended behaviour.
    with _active_task_lock:
        if _active_task_id is not None:
            return jsonify({"error": "A task is already processing. Wait for it to complete before uploading another video."}), 409

    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    file.save(video_path)
    options = {"backend": request.form.get("backend", "gemini"),
               "api_key": request.form.get("api_key") or request.form.get("gemini_key"),
               "openai_url": request.form.get("openai_url"),
               "openai_model": request.form.get("openai_model"),
               "video_type": request.form.get("video_type", "movie"),
               "custom_opening": (request.form.get("custom_opening") or "").strip() or None,
               "use_whisper": request.form.get("use_whisper") != "false",
               "use_context_extender": request.form.get("use_context_extender") == "true",
               "skip_stage2": request.form.get("skip_stage2") == "true",
               "use_character_bank": request.form.get("use_character_bank") == "true"}
    threading.Thread(target=process_video_task, args=(task_id, video_path, options), daemon=True).start()
    return jsonify({"task_id": task_id, "status": "started"})

@app.route('/progress/<task_id>')
def progress(task_id):
    def generate():
        while True:
            status = processing_status.get(task_id, {"status": "unknown"})
            yield f"data: {json.dumps(status, default=str)}\n\n"
            if status.get("status") in ["completed", "failed"]:
                break
            import time; time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<task_id>')
def download(task_id):
    status = processing_status.get(task_id)
    if not status:
        return jsonify({"error": "Not ready"}), 404
    # Allow downloading any stage whose file was actually written to disk,
    # regardless of whether the run ended as completed or failed. The stage
    # *_path fields are only set when the corresponding file exists, so this
    # keeps every on-disk result reachable. The main CSV (output_path) is still
    # restricted to a completed run since it is only meaningful alongside a
    # finished stage 2.
    completed = status.get("status") == "completed"
    stage = request.args.get("stage")
    if stage == "1" and status.get("stage1_path"):
        filename = status["stage1_path"]
    elif stage == "2" and status.get("stage2_path"):
        filename = status["stage2_path"]
    elif stage == "vtt" and status.get("vtt_path"):
        filename = status["vtt_path"]
    elif completed and status.get("output_path"):
        filename = status["output_path"]
    else:
        return jsonify({"error": "No downloadable file for this stage"}), 404
    return send_file(os.path.join(DATA_DIR, filename),
                     as_attachment=True, download_name=filename)


# ---------------------------------------------------------------------------
# History ("重生執行")
# ---------------------------------------------------------------------------

_DETAIL_KINDS = {"stage1": "DetailsDescription.csv",
                 "ad": "AD.csv",
                 "final": "final.csv",
                 "vtt": "final.vtt"}


def _job_summary(job):
    job_id = job["job_id"]
    shots = history.load_shots(job_id, DATA_DIR)
    units = history.load_units(job_id, DATA_DIR)
    run_meta = history.load_run_meta(job_id, job.get("current_run", 0), DATA_DIR)
    manifest = history.load_frame_manifest(job_id, DATA_DIR)
    return {
        "job_id": job_id,
        "filename": job.get("filename", ""),
        "created_at": job.get("created_at", ""),
        "shots_count": len(shots),
        "units_count": len(units),
        "current_run": job.get("current_run", 0),
        "run_status": run_meta.get("status", "running"),
        "runs_count": len(job.get("runs", [])),
        "frames_count": sum(int(v) for v in manifest.get("frames_per_unit", {}).values()),
        "size_bytes": history.job_size_bytes(job_id, DATA_DIR),
        "source_available": bool(history.resolve_source_video(job_id, DATA_DIR)),
    }


def _history_detail(job_id):
    """Full job detail for the UI: shots, units (+current descriptions)."""
    job = history.load_job(job_id, DATA_DIR)
    if not job:
        return None
    video = history.resolve_source_video(job_id, DATA_DIR)
    history.ensure_thumbs(job_id, DATA_DIR, video_path=video)
    shots = history.load_shots(job_id, DATA_DIR)
    units = history.load_units(job_id, DATA_DIR)
    run = job.get("current_run", 0)
    stage1 = history.load_stage1(job_id, run, DATA_DIR)
    stage2 = history.load_stage2(job_id, run, DATA_DIR)

    shot_units = {}
    for u in units:
        for sid in u.get("shot_ids", []):
            shot_units.setdefault(sid, []).append(u["unit_id"])

    units_out = []
    for u in units:
        rec = dict(u)
        rec["description"] = stage1.get(str(u["unit_id"]), {}).get("description", "")
        rec["ad_sentence"] = stage2.get(str(u["unit_id"]), "")
        units_out.append(rec)

    shots_out = []
    for s in shots:
        rec = dict(s)
        rec["unit_ids"] = shot_units.get(s["shot_id"], [])
        shots_out.append(rec)

    runs = []
    for r in job.get("runs", []):
        runs.append(history.load_run_meta(job_id, r, DATA_DIR))

    return {
        "job": {"created_at": job.get("created_at", ""),
                "filename": job.get("filename", ""),
                "video_type": job.get("video_type", ""),
                "current_run": run,
                "gap_detection_enabled": job.get("gap_detection_enabled"),
                "use_context_extender": job.get("use_context_extender")},
        "shots": shots_out,
        "units": units_out,
        "runs": runs,
        "source_available": bool(video),
        "frames_available": any(history.has_unit_frames(job_id, u["unit_id"], DATA_DIR)
                                for u in units),
        "thumb_base": f"/history/{job_id}/thumb/",
        "download_base": f"/history/{job_id}/file",
    }


@app.route('/history')
def history_index():
    return jsonify({"jobs": [_job_summary(j) for j in history.list_jobs(DATA_DIR)]})


@app.route('/history/<job_id>')
def history_detail(job_id):
    detail = _history_detail(job_id)
    if not detail:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(detail)


@app.route('/history/<job_id>/thumb/<int:shot_id>')
def history_thumb(job_id, shot_id):
    p = history.thumb_path(job_id, shot_id, DATA_DIR)
    if not os.path.isfile(p):
        return jsonify({"error": "not found"}), 404
    return send_file(p, mimetype="image/jpeg")


@app.route('/history/<job_id>/file')
def history_file(job_id):
    run = request.args.get("run", type=int)
    kind = request.args.get("kind")
    if run is None or kind not in _DETAIL_KINDS:
        return jsonify({"error": "bad request"}), 400
    path = history.run_file(job_id, run, DATA_DIR, _DETAIL_KINDS[kind])
    if not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    job = history.load_job(job_id, DATA_DIR) or {}
    stem = os.path.splitext(job.get("filename", "run"))[0] or "run"
    return send_file(path, as_attachment=True,
                     download_name=f"{stem}_run{run}_{_DETAIL_KINDS[kind]}")


@app.route('/history/<job_id>', methods=['DELETE'])
def history_delete(job_id):
    job_dir = history.job_dir(job_id, DATA_DIR)
    if not os.path.isdir(job_dir):
        return jsonify({"error": "not found"}), 404
    import shutil
    shutil.rmtree(job_dir, ignore_errors=True)
    logger.info(f"Deleted history job {job_id}")
    return jsonify({"ok": True})


def _reprocess_web_thread(task_id, job_id, selected_unit_ids, options):
    status = {"status": "processing", "step": "initializing", "progress": 0}
    processing_status[task_id] = status
    extractors = {
        "base": extract_frames_base64,
        "context_shots": _get_context_shots,
        "context": _extract_frames_with_context,
    }
    history.reprocess_task(
        task_id, job_id, selected_unit_ids, options, DATA_DIR,
        status, logger, extractors=extractors, emit=None,
        acquire=_try_acquire_task, release=_release_task)


@app.route('/history/<job_id>/reprocess', methods=['POST'])
def reprocess_route(job_id):
    job = history.load_job(job_id, DATA_DIR)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    data = request.get_json(silent=True) or {}
    shot_ids = [int(x) for x in (data.get("shot_ids") or [])]
    if not shot_ids:
        return jsonify({"error": "Please select at least one shot"}), 400
    selected = set(shot_ids)
    selected_units = set()
    for u in history.load_units(job_id, DATA_DIR):
        if any(s in selected for s in u.get("shot_ids", [])):
            selected_units.add(u["unit_id"])
    if not selected_units:
        return jsonify({"error": "Selection maps to no AD units"}), 400

    with _active_task_lock:
        if _active_task_id is not None:
            return jsonify({"error": "A task is already processing. Wait for it to complete before starting another."}), 409

    task_id = str(uuid.uuid4())
    options = {
        "backend": data.get("backend", "gemini"),
        "api_key": data.get("api_key") or data.get("gemini_key"),
        "openai_url": data.get("openai_url"),
        "openai_model": data.get("openai_model"),
        "video_type": data.get("video_type", job.get("video_type", "movie")),
        "custom_opening": (data.get("custom_opening") or "").strip() or None,
        "use_context_extender": bool(data.get("use_context_extender",
                                               job.get("use_context_extender", False))),
        "skip_stage2": bool(data.get("skip_stage2", False)),
    }
    threading.Thread(target=_reprocess_web_thread,
                     args=(task_id, job_id, sorted(selected_units), options),
                     daemon=True).start()
    logger.info(f"Job {job_id}: reprocess started task {task_id} "
                f"(units={sorted(selected_units)}, backend={options['backend']})")
    return jsonify({"task_id": task_id, "status": "started"})


if __name__ == '__main__':
    app.run(debug=False, port=5000)
