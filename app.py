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

# Resolve resource folders for both source and PyInstaller-frozen layouts.
# Mirrors desktop.py exactly so both versions write outputs to the same place.
BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
FROZEN = bool(getattr(sys, "frozen", False))
if FROZEN:
    # Packaged app: keep user data in a writable location (Program Files is read-only)
    DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "ShotByShot")
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'shot_by_shot_web.log'), encoding='utf-8'),
        logging.StreamHandler(),
    ],
)

# Log files older than 30 days are no longer needed; delete them on startup so
# the log directory does not grow without bound.
def _cleanup_old_logs(log_dir, max_age_days=30, pattern="*.log"):
    try:
        cutoff = datetime.now().timestamp() - max_age_days * 86400
        for name in os.listdir(log_dir):
            if not name.lower().endswith(".log"):
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
                _, buf = cv2.imencode('.jpg', frame)
                all_frames_b64.append(base64.b64encode(buf).decode('utf-8'))
    cap.release()
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
        api_key = options.get("api_key") or options.get("gemini_key") or options.get("openrouter_key")        
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
        logger.info(f"Task {task_id}: {len(shots)} shots detected")
        status["progress"] = 20
        status["detail"] = f"Found {len(shots)} shots"
        status["step"] = "transcription"
        subtitles = []
        if options.get("use_whisper", True):
            def trans_progress(progress, message):
                status["detail"] = f"Transcription: {message}"
                status["progress"] = 20 + progress * 10
            try:
                subtitles = transcribe_video(video_path, language="auto", callback=trans_progress)
                status["detail"] = f"Transcribed {len(subtitles)} segments"
                logger.info(f"Task {task_id}: transcribed {len(subtitles)} segments")
            except Exception:
                subtitles = []
                status["detail"] = "Transcription failed, falling back to shot-based mode"
                logger.warning(f"Task {task_id}: transcription failed, shot-based mode", exc_info=True)
        else:
            status["detail"] = "Skipped"
        status["progress"] = 30

        # Build AD units: dialogue-gap intervals when transcription is enabled,
        # otherwise one unit per shot (existing behavior)
        units = []
        if subtitles:
            ad_intervals = detect_ad_intervals(subtitles, shots, video_duration=video_duration)
            status["detail"] = f"Detected {len(ad_intervals)} dialogue-gap AD intervals"
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
        status["detail"] = f"{len(units)} AD units to describe"
        status["progress"] = 40
        logger.info(f"Task {task_id}: built {len(units)} AD units")

        # Character detection (optional) - parity with desktop: stub only
        if options.get("use_character_bank"):
            status["step"] = "character_bank"
            status["progress"] = 50

        descriptions_dict = {}
        if not api_key:
            logger.warning(f"Task {task_id}: no API key provided - VLM descriptions and Stage 2 will be SKIPPED (output CSVs will be empty)")
        if api_key:
            status["step"] = "stage1_vlm"
            status["detail"] = "Starting VLM descriptions..."
            logger.info(f"Task {task_id}: VLM descriptions begin - {len(units)} units, backend={backend}, context={bool(options.get('use_context_extender'))}")
            shot_scales = [2] * len(shots)
            threads = [[j for j in range(len(shots))]]
            for i, unit in enumerate(units):
                try:
                    logger.info(f"Task {task_id}: unit {i+1}/{len(units)} ({unit.get('mode', 'shot')})")
                    frame_shot = {"start_time": unit["start"], "end_time": unit["end"]}
                    if options.get("use_context_extender"):
                        context_shots = _get_context_shots(shots, unit["shot_ids"])
                        frames_b64 = _extract_frames_with_context(video_path, frame_shot, context_shots)
                    else:
                        frames_b64 = extract_frames_base64(video_path, frame_shot)
                    current_shots = [s - 1 for s in unit["shot_ids"]]
                    film_grammar = {"video_type": options.get("video_type", "movie"), "label_type": "none",
                                    "char_text": "", "current_shots": current_shots, "threads": threads,
                                    "shot_scales": shot_scales, "prompt_variant": 4,
                                    "custom_opening": options.get("custom_opening")}
                    logger.info(f"Task {task_id}: calling API for unit {i+1} ({len(frames_b64)} frames)")
                    desc = describe_frames(frames_b64, api_key, backend=backend, film_grammar=film_grammar,
                                           openai_url=options.get("openai_url"),
                                           openai_model=options.get("openai_model"))
                    descriptions_dict[unit["unit_id"]] = desc
                    logger.info(f"Task {task_id}: unit {i+1} completed ({len(desc)} chars)")
                    status["detail"] = f"Described unit {i+1}/{len(units)}"
                except Exception as e:
                    descriptions_dict[unit["unit_id"]] = ""
                    status["detail"] = f"Unit {i+1} failed: {str(e)[:50]}"
                    logger.error(f"Task {task_id}: unit {i+1}/{len(units)} failed: {e}", exc_info=True)
                # Rate limit protection: stay under ~15 RPM
                if i < len(units) - 1:
                    time.sleep(3)
                status["progress"] = 50 + int((i/len(units))*20)
        status["detail"] = f"Described {len(descriptions_dict)} units"
        status["progress"] = 80

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
            # Abort stage 2 when the majority of shots could not be described
            if len(units) > 0 and success_count / len(units) < 0.5:
                status["status"] = "failed"
                status["partial"] = True
                status["error"] = (
                    f"API quota limit reached: only {success_count}/{len(units)} shots could be "
                    f"described (full audio descriptions need ~{2 * len(units)} API calls). "
                    "Stage 2 (AD summarization) was skipped, so final.csv and _AD.csv were NOT "
                    "generated. The partial shot descriptions are saved in _DetailsDescription.csv. "
                    "Free-tier daily quota resets at midnight Pacific Time. Consider a paid tier or "
                    "a higher-quota model (e.g. a 2.x Flash model) to process this video."
                )
                logger.warning(f"Task {task_id}: aborting stage 2 - only {success_count}/{len(units)} shots succeeded (quota likely exhausted)")
            else:
                status["step"] = "stage2_summarize"
                status["detail"] = "Summarizing audio descriptions..."
                logger.info(f"Task {task_id}: stage 2 summarization begin")
                try:
                    stage2_results = batch_summarize(stage1_results, api_key, backend=backend,
                                                    video_type=options.get("video_type", "movie"))
                    ad_sentence_map = {r["shot_id"]: r["ad_sentence"] for r in stage2_results}
                    non_empty = sum(1 for v in ad_sentence_map.values() if str(v).strip())
                    logger.info(f"Task {task_id}: stage 2 produced {non_empty}/{len(ad_sentence_map)} non-empty AD sentences")
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

                    logger.info(f"Task {task_id}: completed - outputs {timestamp}_DetailsDescription.csv / {timestamp}_AD.csv / {timestamp}-final.csv / {timestamp}-final.vtt")
                    # Remove the uploaded copy on success (desktop reads in place; the
                    # upload is only a transfer artifact). Keep it on failure for debugging.
                    try:
                        os.remove(video_path)
                    except OSError:
                        pass
                except Exception as stage2_err:
                    # Stage 2 failed (e.g. API quota exhausted mid-run). Stage-1
                    # results are already saved, so keep them downloadable and
                    # mark the run as failed with a clear message.
                    status["status"] = "failed"
                    status["error"] = (
                        f"Stage 2 (AD summarization) failed: {stage2_err} "
                        "The stage-1 shot descriptions were saved and can be downloaded above. "
                        "Free-tier Gemini daily quota may have been exhausted; it resets at "
                        "midnight Pacific Time."
                    )
                    status["partial"] = True
                    logger.error(f"Task {task_id}: stage 2 failed but stage-1 results kept: {stage2_err}")
        else:
            # No API key or stage 2 explicitly skipped: only stage-1 CSV exists.
            status["status"] = "completed"
            status["progress"] = 100
            logger.info(f"Task {task_id}: completed (stage 2 not run) - outputs {timestamp}_DetailsDescription.csv")
    except Exception as e:
        status["status"] = "failed"
        status["error"] = str(e)
        logger.critical(f"Task {task_id}: FAILED at step={status.get('step')} progress={status.get('progress')}: {e}", exc_info=True)
    finally:
        _release_task(task_id)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_api', methods=['POST'])
def test_api_route():
    data = request.get_json(silent=True) or {}
    ok, message = test_connection(
        data.get("api_key"),
        data.get("backend", "gemini-3.7-flash"),
        data.get("openai_url"),
        data.get("openai_model"),
    )
    logger.info(f"API key test: backend={data.get('backend')} ok={ok} - {message}")
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
               "api_key": request.form.get("api_key") or request.form.get("gemini_key") or request.form.get("openrouter_key"),
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

if __name__ == '__main__':
    app.run(debug=False, port=5000)
