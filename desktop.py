"""
Desktop application using pywebview
"""
import os
import sys
import json
import uuid
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import numpy as np
from datetime import datetime

# Configure logging
FROZEN = bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()
_MEIPASS = getattr(sys, "_MEIPASS", None)
if FROZEN:
    BASE_DIR = _MEIPASS or os.path.dirname(sys.executable)
    DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.dirname(sys.executable)), "BuddyAd")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)

# Logs rotate at 5MB per file (3 backups: .1/.2/.3); anything older than 30
# days is deleted on startup so the log directory cannot grow without bound.
def _cleanup_old_logs(log_dir, max_age_days=30):
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

# RotatingFileHandler: 5MB per file, 3 backups (.1/.2/.3), then 30-day age
# cleanup above retires stale files.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(DATA_DIR, 'shot_by_shot.log'),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8',
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class _NoiseFilter(logging.Filter):
    """Drop pywebview's close-time attribute-recursion spam that would
    otherwise flood the log and bury real diagnostics."""
    def filter(self, record):
        msg = record.getMessage()
        return not msg.startswith("Error while processing window.native")


for _h in logging.getLogger().handlers:
    _h.addFilter(_NoiseFilter())


def _excepthook(exc_type, exc_value, exc_tb):
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _excepthook

_WEBVIEW2_GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


def _webview2_runtime_installed():
    """True if the Edge WebView2 Runtime is registered (machine or per-user)."""
    if sys.platform != "win32":
        return True
    import winreg
    keys = [
        (winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\WOW6432Node\\Microsoft\\EdgeUpdate\\Clients\\" + _WEBVIEW2_GUID),
        (winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\EdgeUpdate\\Clients\\" + _WEBVIEW2_GUID),
        (winreg.HKEY_CURRENT_USER, "Software\\Microsoft\\EdgeUpdate\\Clients\\" + _WEBVIEW2_GUID),
    ]
    for hive, path in keys:
        try:
            with winreg.OpenKey(hive, path):
                return True
        except OSError:
            continue
    return False


def _warn_no_webview2():
    msg = (
        "Microsoft Edge WebView2 Runtime 未安裝，無法啟動介面。\n\n"
        "The Microsoft Edge WebView2 Runtime is required but not installed.\n\n"
        "請安裝後重新執行 / Please install it, then run the app again:\n"
        "https://developer.microsoft.com/microsoft-edge/webview2/"
    )
    logger.critical("WebView2 Runtime not found; aborting startup")
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, msg, "Shot-by-Shot", 0x10)
    except Exception:
        print(msg)


_WINDOW_TITLE = "Shot-by-Shot Audio Describer"


def _start_window_watchdog(timeout=60):
    """Poll for the main window; if it never appears, dump WebView2
    subprocess state to the log so a silent native failure can be
    diagnosed remotely (0 processes = broken runtime; >0 = GPU/render)."""
    stop = threading.Event()

    def watch():
        deadline = time.time() + timeout
        while time.time() < deadline:
            if stop.wait(2):
                return
            try:
                import ctypes
                hwnd = ctypes.windll.user32.FindWindowW(None, _WINDOW_TITLE)
                if hwnd:
                    logger.info(f"Window handle found (hwnd=0x{hwnd:X})")
                    return
            except Exception:
                pass
        logger.warning(f"No window after {timeout}s - dumping WebView2 process state")
        try:
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq msedgewebview2.exe", "/FO", "CSV"],
                capture_output=True, text=True, timeout=15,
            )
            count = out.stdout.count("msedgewebview2")
            logger.warning(f"msedgewebview2.exe running: {count} process(es)")
            logger.warning(f"tasklist output:\n{out.stdout.strip()}")
        except Exception as e:
            logger.error(f"tasklist failed: {e}")

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    return stop

# Add current directory to path
if FROZEN:
    sys.path.insert(0, _MEIPASS or os.path.dirname(sys.executable))
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import webview, provide fallback message if not available
try:
    import webview
except ImportError:
    print("Error: pywebview not installed. Run: uv pip install pywebview --system")
    print("Alternatively, use the web version: python app.py")
    sys.exit(1)

from processing.shot_detector import detect_shots
from processing.sensevoice_transcriber import transcribe_video
from processing.dialogue_gap_detector import detect_ad_intervals
from processing.vlm_describer import describe_frames
from processing.llm_summarizer import batch_summarize, estimate_word_limit, test_connection
from processing.film_grammar import get_effective_shot_scale, select_prompt_variant
from processing.character_recognizer import detect_faces, extract_face_embeddings, cluster_faces
from processing.vtt_writer import write_vtt
from processing import history_engine as history
from processing import time_ranges

# Lazy import to mirror vlm_describer/llm_summarizer (also keeps the frozen
# PyArmor builds from resolving a private module at import time).
def _api_common():
    import processing._api_common as m
    return m


class AppBridge:
    """Bridge between JavaScript frontend and Python backend."""
    
    def __init__(self):
        self.window = None
        self.processing_status = {}
        self._active_task_id = None
    
    def attach_window(self, window):
        self.window = window
    
    def emit_progress(self, task_id, status):
        """Send progress update to frontend."""
        if self.window:
            try:
                self.window.evaluate_js(
                    f"window.dispatchEvent(new CustomEvent('progress', {{detail: {json.dumps(status)}}}))"
                )
            except:
                pass
    
    # File dialog
    def open_file_dialog(self):
        """Open file dialog to select video file. Retries once on transient
        COM/WebView2 failures instead of letting the exception escape."""
        import webview
        file_types = ('Video Files (*.mp4;*.mkv;*.avi;*.mov)',)
        for attempt in (1, 2):
            try:
                logger.info(f"File dialog: attempt {attempt}/2 begin")
                result = self.window.create_file_dialog(
                    webview.FileDialog.OPEN,
                    allow_multiple=False,
                    file_types=file_types
                )
                logger.info(f"File dialog returned: {result!r}")
                if result and len(result) > 0:
                    return result[0]
                return None
            except Exception as e:
                logger.error(f"File dialog failed (attempt {attempt}/2): {e}")
                time.sleep(0.5)
        return None
    
    # API key test
    def test_api(self, backend, api_key, openai_url=None, openai_model=None, model=None):
        """Send a tiny prompt through the selected backend; returns
        {"ok": bool, "message": str} for the UI."""
        logger.info(f"API key test requested: backend={backend} url={openai_url} model={openai_model}/{model}")
        ok, message = test_connection(api_key, backend, openai_url, openai_model, model)
        logger.info(f"API key test result: url={openai_url} model={openai_model} ok={ok} - {message}")
        return {"ok": ok, "message": message}

    # Processing functions
    def process_video(self, video_path, options):
        """Start video processing in background thread.

        Only one task may run at a time: the free-tier Gemini quota is
        per-account-per-day, and concurrent runs exhaust it in minutes. A
        second submission is rejected outright."""
        if self._active_task_id is not None:
            logger.warning("Cannot start - another task is already processing")
            return {"task_id": None, "status": "busy",
                    "error": "A task is already processing. Wait for it to complete before starting another."}

        # "describe only / in addition" needs usable ranges; reject bad input
        # here so the user gets the message instead of a failed task. The
        # end-of-video check waits until the duration is known.
        try:
            time_ranges.validate_mode((options or {}).get("range_mode"),
                                      (options or {}).get("time_ranges"))
        except time_ranges.RangeError as e:
            logger.info(f"Cannot start - {e}")
            return {"task_id": None, "status": "error", "error": str(e)}

        task_id = str(uuid.uuid4())
        self._active_task_id = task_id
        self.processing_status[task_id] = {"status": "processing", "step": "initializing"}

        thread = threading.Thread(
            target=self._process_video_task,
            args=(task_id, video_path, options),
            daemon=True
        )
        thread.start()

        return {"task_id": task_id, "status": "started"}
    
    def _process_video_task(self, task_id, video_path, options):
        """Background video processing."""
        status = self.processing_status[task_id]
        save_to_history = bool((options or {}).get("save_to_history", True))
        
        try:
            # Generate timestamp for file names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # History record: keep only neutral metadata + a local path reference
            # (never the api url / model / api key).
            if save_to_history:
                history.create_job(
                    task_id, os.path.basename(video_path), DATA_DIR,
                    source_path=video_path,
                    video_type=options.get("video_type", "movie"),
                    gap_detection_enabled=bool(options.get("use_whisper", True)),
                    use_context_extender=bool(options.get("use_context_extender", False)),
                )
            else:
                logger.info("History saving disabled by user - outputs written to DATA_DIR only")
            
            # Get video duration for progress display
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = frame_count / fps if fps > 0 else 0
            cap.release()

            # Ranges were validated in process_video; this second pass also
            # catches ones that run past the end of the video.
            range_mode, custom_ranges = time_ranges.validate_mode(
                options.get("range_mode"), options.get("time_ranges"),
                video_duration if video_duration > 0 else None)
            if custom_ranges:
                logger.info(f"Range mode '{range_mode}' - {len(custom_ranges)} range(s) "
                            f"{time_ranges.format_ranges(custom_ranges)}")
            
            # Step 1: Shot detection with callback
            status["step"] = "shot_detection"
            status["detail"] = f"Detecting shots... ({video_duration:.0f}s video)"
            status["progress"] = 10
            self.emit_progress(task_id, status)
            
            def shot_progress(progress, message):
                status["detail"] = f"Detecting shots: {message}"
                status["progress"] = 10 + progress * 10  # 10-20%
                self.emit_progress(task_id, status)
            
            shots = detect_shots(video_path, callback=shot_progress)
            status["shots_count"] = len(shots)
            if save_to_history:
                history.save_shots(task_id, shots, DATA_DIR)
            logger.info(f"{len(shots)} shots detected")
            status["detail"] = f"Found {len(shots)} shots"
            status["progress"] = 20
            self.emit_progress(task_id, status)
            
            # Step 2: Transcription (SenseVoice)
            status["step"] = "transcription"
            status["detail"] = "Starting transcription..."
            self.emit_progress(task_id, status)
            
            subtitles = []
            if options.get("use_whisper", True):
                def trans_progress(progress, message):
                    status["detail"] = f"Transcription: {message}"
                    status["progress"] = 20 + progress * 10  # 20-30%
                    self.emit_progress(task_id, status)
                
                try:
                    subtitles = transcribe_video(video_path, language="auto", callback=trans_progress)
                    status["detail"] = f"Transcribed {len(subtitles)} segments"
                    logger.info(f"Transcribed {len(subtitles)} segments")
                except Exception:
                    subtitles = []
                    status["detail"] = "Transcription failed, using shot-based mode"
                    logger.warning("Transcription failed, using shot-based mode", exc_info=True)
            else:
                status["detail"] = "Skipped"
            status["progress"] = 30
            if save_to_history:
                history.save_subtitles(task_id, subtitles, DATA_DIR)
            self.emit_progress(task_id, status)
            
            # Step 3: Build AD units (dialogue gaps when transcribed, else shots)
            status["step"] = "merging"
            status["detail"] = "Building AD units..."
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
                logger.info("No dialogue-gap intervals; falling back to shot-based units")
                for shot in shots:
                    units.append({
                        "unit_id": shot["shot_id"],
                        "start": shot["start_time"],
                        "end": shot["end_time"],
                        "shot_ids": [shot["shot_id"]],
                        "mode": "shot",
                    })
            if not units:
                logger.info(f"No shots detected; using full video as single unit ({video_duration:.1f}s)")
                units.append({
                    "unit_id": 1,
                    "start": 0,
                    "end": video_duration,
                    "shot_ids": [],
                    "mode": "shot",
                })
            if custom_ranges:
                # A range is described exactly as asked for - never snapped to a
                # shot, and never shortened to a nearby dialogue gap. In "extra"
                # mode both passes are kept, even where they overlap.
                custom_units = time_ranges.build_custom_units(custom_ranges, shots)
                units = custom_units if range_mode == "only" else units + custom_units
                logger.info(f"Range mode '{range_mode}' -> {len(custom_units)} "
                            f"user-range units ({len(units)} total)")
            units_label = "dialogue-gap" if subtitles else "shot-based"
            if range_mode == "only":
                units_label = "user ranges"
            elif range_mode == "extra":
                units_label = f"{units_label} + user ranges"
            status["detail"] = f"{len(units)} AD units to describe ({units_label})"
            status["progress"] = 40
            if save_to_history:
                history.save_units(task_id, units, DATA_DIR)
                history.begin_run(task_id, 0, DATA_DIR,
                                  kind="initial",
                                  gap_detection_active=bool(subtitles))
                history.set_current_run(task_id, 0, DATA_DIR)
            self.emit_progress(task_id, status)
            logger.info(f"Built {len(units)} AD units")

            # Persist the exact image payload + rendered prompt per unit so a
            # replay never has to re-detect shots, re-transcribe or re-grab
            # frames (the source video stays where the user put it and is not
            # copied into the job folder).
            status["step"] = "frames"
            frame_counts = {}
            frames_cache = {}
            for i, unit in enumerate(units):
                frame_shot = {"start_time": unit["start"], "end_time": unit["end"]}
                # Context extension follows shots; a user range is not a shot,
                # so it is deliberately never extended.
                if options.get("use_context_extender") and shots and unit.get("mode") != "custom":
                    context_shots = self._get_context_shots(shots, unit["shot_ids"])
                    frames_b64 = self._extract_frames_with_context(video_path, frame_shot, context_shots)
                else:
                    frames_b64 = self._extract_frames(video_path, frame_shot)
                if save_to_history:
                    history.save_unit_frames(task_id, unit["unit_id"], frames_b64, DATA_DIR)
                    history.save_thumbs_from_unit_frames(task_id, unit, frames_b64, DATA_DIR)
                else:
                    frames_cache[str(unit["unit_id"])] = frames_b64
                frame_counts[str(unit["unit_id"])] = len(frames_b64)
                status["detail"] = f"Saving frames {i+1}/{len(units)}"
                status["progress"] = 40 + int(((i + 1) / len(units)) * 5)
                self.emit_progress(task_id, status)
            if save_to_history:
                history.save_frame_manifest(
                    task_id, DATA_DIR,
                    backend=options.get("backend", "gemini"),
                    model=options.get("model") or options.get("openai_model") or "",
                    use_context_extender=bool(options.get("use_context_extender")),
                    speech_transcription=bool(subtitles),
                    frames_per_unit=frame_counts,
                )
                history.write_agent_jsonl(task_id, DATA_DIR, units)
            logger.info(f"Saved frames for {len(units)} units ({sum(frame_counts.values())} images)")

            # Step 4: Character detection (optional)
            if options.get("use_character_bank"):
                status["step"] = "character_bank"
                self.emit_progress(task_id, status)
                # Character detection logic here
                status["progress"] = 50
                self.emit_progress(task_id, status)
            
            # Step 5: VLM description
            backend = options.get("backend", "gemini")
            api_key = options.get("api_key", "")
            use_context = options.get("use_context_extender", False)
            descriptions_dict = {}
            
            if api_key:
                status["step"] = "describe"
                status["detail"] = f"Starting VLM descriptions... ({len(units)} units to process)"
                logger.info(f"Starting VLM descriptions: {len(units)} units, backend={backend}, context={use_context}")
                self.emit_progress(task_id, status)
                
                descriptions_dict = {}
                stage1_error_categories = {}
                for i, unit in enumerate(units):
                    try:
                        logger.info(f"Processing unit {i+1}/{len(units)}...")
                        
                        # Replay the exact frames (persisted above, or kept in
                        # memory when history saving is disabled); the prompt is
                        # rebuilt deterministically from the unit + run options.
                        frames_b64 = (history.load_unit_frames(task_id, unit["unit_id"], DATA_DIR)
                                      if save_to_history
                                      else frames_cache.get(str(unit["unit_id"]), []))

                        duration = unit["end"] - unit["start"]
                        num_frames = len(frames_b64)
                        status["detail"] = f"Describing unit {i+1}/{len(units)} ({duration:.0f}s, {num_frames} frames)"

                        logger.info(f"Calling API for unit {i+1} ({num_frames} frames)...")

                        film_grammar = {
                            "video_type": options.get("video_type", "movie"),
                            "label_type": "none",
                            "char_text": "",
                            "current_shots": time_ranges.current_shot_indices(unit, shots),
                            "threads": [[j for j in range(len(shots))]],
                            "shot_scales": [2] * len(shots),
                            "prompt_variant": 4,
                            "custom_opening": (options.get("custom_opening") or "").strip() or None,
                            "lang": options.get("lang")
                        }

                        desc = describe_frames(
                            frames_b64, api_key,
                            backend=backend,
                            film_grammar=film_grammar,
                            openai_url=options.get("openai_url"),
                            openai_model=options.get("openai_model"),
                            model=options.get("model")
                        )
                        logger.info(f"Unit {i+1} completed: {len(desc)} chars")
                        descriptions_dict[unit["unit_id"]] = desc
                        if save_to_history:
                            history.record_unit(task_id, 0, DATA_DIR, unit, desc)
                    except Exception as e:
                        logger.error(f"Unit {i+1} failed: {e}", exc_info=True)
                        descriptions_dict[unit["unit_id"]] = ""
                        if save_to_history:
                            history.record_unit(task_id, 0, DATA_DIR, unit, "")
                        status["detail"] = f"Unit {i+1} failed: {str(e)[:50]}"
                        if e is not None:
                            cat = _api_common().categorize_error(e)
                            stage1_error_categories[cat] = stage1_error_categories.get(cat, 0) + 1
                    
                    # Rate limit protection: wait between API calls
                    if i < len(units) - 1:
                        import time
                        time.sleep(3)  # Wait 3 seconds to stay under 15 RPM
                    
                    status["progress"] = 50 + (i / len(units)) * 20
                    self.emit_progress(task_id, status)
            else:
                status["detail"] = "Skipped (no API key)"
                logger.warning("No API key provided - VLM descriptions and Stage 2 will be SKIPPED (output CSVs will be empty)")
            
            status["detail"] = f"Described {len(descriptions_dict)} units"
            status["progress"] = 80
            self.emit_progress(task_id, status)
            
            # Step 6: Stage 2 summarization
            ad_sentence_map = {}

            # Stage-1 results are always persisted when there is anything to
            # keep, so partial successes survive a quota abort.
            stage1_results = []
            for unit in units:
                stage1_results.append({
                    "shot_id": unit["unit_id"],
                    "start": unit["start"],
                    "end": unit["end"],
                    "mode": unit.get("mode"),
                    "description": descriptions_dict.get(unit["unit_id"], "")
                })
            success_count = sum(1 for v in descriptions_dict.values() if str(v).strip())
            import pandas as pd
            output_dir = DATA_DIR
            os.makedirs(output_dir, exist_ok=True)
            if success_count > 0:
                stage1_df = pd.DataFrame(stage1_results)
                stage1_path = os.path.join(output_dir, f"{timestamp}_DetailsDescription.csv")
                stage1_df.to_csv(stage1_path, index=False, encoding="utf-8-sig")
                status["stage1_path"] = stage1_path
                logger.info(f"Stage 1 written: non-empty descriptions {success_count}/{len(stage1_results)}")

            if not options.get("skip_stage2") and api_key and descriptions_dict:
                # Abort stage 2 when the majority of shots could not be
                # described. Rather than blaming quota unconditionally, diagnose
                # the actual cause from the errors collected during stage 1.
                # Keep the partial stage-1 CSV and say so clearly rather than
                # silently emitting an empty final file.
                if len(units) > 0 and success_count / len(units) < 0.5:
                    status["status"] = "failed"
                    status["partial"] = True
                    _ac = _api_common()
                    dominant = _ac.dominant_category(stage1_error_categories)
                    status["error"] = _ac.build_stage1_blocked_message(
                        dominant, success_count, len(units)
                    )
                    if save_to_history:
                        history.update_run(task_id, 0, DATA_DIR,
                                           status="failed", stage1_count=success_count)
                    logger.warning(f"Aborting stage 2 - only {success_count}/{len(units)} shots succeeded (cause={dominant})")
                    self.emit_progress(task_id, status)
                else:
                    status["step"] = "summarize"
                    status["detail"] = "Summarizing audio descriptions..."
                    logger.info("Stage 2 summarization begin")
                    self.emit_progress(task_id, status)

                    stage2_results = batch_summarize(
                        stage1_results, api_key, backend=backend,
                        video_type=options.get("video_type", "movie"),
                        openai_url=options.get("openai_url"),
                        openai_model=options.get("openai_model"),
                        model=options.get("model"),
                        lang=options.get("lang")
                    )
                    ad_sentence_map = {r["shot_id"]: r["ad_sentence"] for r in stage2_results}
                    non_empty = sum(1 for v in ad_sentence_map.values() if str(v).strip())
                    logger.info(f"Stage 2 produced {non_empty}/{len(ad_sentence_map)} non-empty AD sentences")

                    stage2_df = pd.DataFrame(stage2_results)
                    stage2_path = os.path.join(output_dir, f"{timestamp}_AD.csv")
                    stage2_df.to_csv(stage2_path, index=False, encoding="utf-8-sig")
                    status["stage2_path"] = stage2_path

                    status["progress"] = 100
                    status["status"] = "completed"

                    output_df = pd.DataFrame([{
                        "shot_id": unit["unit_id"],
                        "start": unit["start"],
                        "end": unit["end"],
                        "ad_sentence": ad_sentence_map.get(unit["unit_id"], "")
                    } for unit in units])
                    output_path = os.path.join(output_dir, f"{timestamp}-final.csv")
                    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                    status["output_path"] = output_path

                    vtt_path = os.path.join(output_dir, f"{timestamp}-final.vtt")
                    write_vtt(vtt_path, output_df.to_dict("records"))

                    logger.info(f"Completed - outputs {timestamp}_DetailsDescription.csv / {timestamp}_AD.csv / {timestamp}-final.csv / {timestamp}-final.vtt")
                    if save_to_history:
                        history.save_stage2(task_id, 0, DATA_DIR, ad_sentence_map)
                        history.write_run_files(
                            task_id, 0, DATA_DIR, stage1_results, stage2_results,
                            ad_sentence_map, units, mirror_root=False)
                        history.update_run(task_id, 0, DATA_DIR,
                                           status="completed", stage1_count=success_count,
                                           stage2_count=len(ad_sentence_map))

                    self.emit_progress(task_id, status)
            else:
                # No API key or stage 2 explicitly skipped: only stage-1 CSV exists.
                status["status"] = "completed"
                status["progress"] = 100
                if save_to_history:
                    history.write_run_files(task_id, 0, DATA_DIR, stage1_results, None,
                                            {}, units, mirror_root=False)
                    history.update_run(task_id, 0, DATA_DIR,
                                       status="completed", stage1_count=success_count)
                logger.info(f"Completed (stage 2 not run) - outputs {timestamp}_DetailsDescription.csv")
                self.emit_progress(task_id, status)

        except Exception as e:
            status["status"] = "failed"
            # Stage-2 summarization failures bubble here; diagnose the real
            # cause (auth/config/quota/network) instead of a bare string where
            # possible, keeping stage-1 results as the recoverable artifact.
            if status.get("step") == "summarize":
                _ac = _api_common()
                status["error"] = _ac.build_stage2_failed_message(
                    _ac.categorize_error(e), _ac.quote_error_detail(e)
                )
            else:
                status["error"] = str(e)
            logger.critical(
                f"Processing FAILED at step={status.get('step')} progress={status.get('progress')}: {e}",
                exc_info=True,
            )
            try:
                if save_to_history:
                    history.update_run(task_id, 0, DATA_DIR,
                                       status="failed", step=status.get("step"))
            except Exception:
                pass
            self.emit_progress(task_id, status)
        finally:
            if self._active_task_id == task_id:
                self._active_task_id = None
    
    def _get_context_shots(self, shots, current_shot_ids):
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
    
    def _extract_frames_with_context(self, video_path, current_shot, context_shots):
        """Extract frames from current shot + context shots."""
        import cv2
        import base64
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        all_frames_b64 = []
        
        # Extract frames from each context shot
        for shot in context_shots:
            duration = shot["end_time"] - shot["start_time"]
            if duration <= 10:
                num_frames = 16
            elif duration <= 30:
                num_frames = 32
            elif duration <= 60:
                num_frames = 48
            else:
                num_frames = 64
            
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
    
    def _extract_frames(self, video_path, shot, num_frames=8):
        """Extract frames from video shot."""
        import cv2
        import base64
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        duration = shot["end_time"] - shot["start_time"]
        if duration <= 10:
            num_frames = 16
        elif duration <= 30:
            num_frames = 32
        elif duration <= 60:
            num_frames = 48
        else:
            num_frames = 64
        
        start_frame = int(shot["start_time"] * fps)
        end_frame = int(shot["end_time"] * fps)
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        
        frames_b64 = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode('.jpg', frame)
                frames_b64.append(base64.b64encode(buf).decode('utf-8'))
        
        cap.release()
        return frames_b64
    
    def get_progress(self, task_id):
        """Get current progress for a task."""
        return self.processing_status.get(task_id, {"status": "unknown"})
    
    def get_output_path(self, task_id):
        """Get output file path for download."""
        status = self.processing_status.get(task_id, {})
        return status.get("output_path")

    # ------------------------------------------------------------------ history
    def _try_acquire_task(self, task_id):
        if self._active_task_id is not None:
            return False
        self._active_task_id = task_id
        return True

    def _release_task(self, task_id):
        if self._active_task_id == task_id:
            self._active_task_id = None

    def list_history(self):
        jobs = []
        for job in history.list_jobs(DATA_DIR):
            job_id = job["job_id"]
            meta = history.load_run_meta(job_id, job.get("current_run", 0), DATA_DIR)
            manifest = history.load_frame_manifest(job_id, DATA_DIR)
            jobs.append({
                "job_id": job_id,
                "filename": job.get("filename", ""),
                "created_at": job.get("created_at", ""),
                "shots_count": len(history.load_shots(job_id, DATA_DIR)),
                "units_count": len(history.load_units(job_id, DATA_DIR)),
                "current_run": job.get("current_run", 0),
                "run_status": meta.get("status", "running"),
                "runs_count": len(job.get("runs", [])),
                "frames_count": sum(int(v) for v in manifest.get("frames_per_unit", {}).values()),
                "size_bytes": history.job_size_bytes(job_id, DATA_DIR),
                "source_available": bool(history.resolve_source_video(job_id, DATA_DIR)),
            })
        return {"jobs": jobs}

    def history_detail(self, job_id):
        job = history.load_job(job_id, DATA_DIR)
        if not job:
            return {"error": "not found"}
        video = history.resolve_source_video(job_id, DATA_DIR)
        shots_all = history.ensure_thumbs(job_id, DATA_DIR, video_path=video)
        if not shots_all:
            shots_all = history.load_shots(job_id, DATA_DIR)
        units = history.load_units(job_id, DATA_DIR)
        run = job.get("current_run", 0)
        stage1 = history.load_stage1(job_id, run, DATA_DIR)
        stage2 = history.load_stage2(job_id, run, DATA_DIR)

        shot_units = {}
        for u in units:
            for sid in u.get("shot_ids", []):
                shot_units.setdefault(sid, []).append(u["unit_id"])

        shots_out = []
        for s in shots_all:
            rec = dict(s)
            rec["unit_ids"] = shot_units.get(s["shot_id"], [])
            rec["thumb"] = history.thumb_data_url(job_id, s["shot_id"], DATA_DIR)
            shots_out.append(rec)

        units_out = []
        for u in units:
            rec = dict(u)
            rec["description"] = stage1.get(str(u["unit_id"]), {}).get("description", "")
            rec["ad_sentence"] = stage2.get(str(u["unit_id"]), "")
            # A user-picked range is previewed by its own thumbnail, not by the
            # shots it happens to cover.
            if u.get("mode") == "custom":
                rec["thumb"] = history.thumb_data_url(job_id, u["unit_id"], DATA_DIR)
            units_out.append(rec)

        runs = [history.load_run_meta(job_id, r, DATA_DIR)
                for r in job.get("runs", [])]
        return {
            "job": {"created_at": job.get("created_at", ""),
                    "filename": job.get("filename", ""),
                    "video_type": job.get("video_type", ""),
                    "current_run": run,
                    "use_context_extender": job.get("use_context_extender")},
            "shots": shots_out,
            "units": units_out,
            "runs": runs,
            "source_available": bool(video),
            "frames_available": any(history.has_unit_frames(job_id, u["unit_id"], DATA_DIR)
                                    for u in units),
            "desktop": True,
        }

    def reprocess_history(self, job_id, shot_ids, options):
        """Re-describe the units of the selected shots, then re-run stage 2."""
        if self._active_task_id is not None:
            return {"task_id": None, "status": "busy",
                    "error": "A task is already processing. Wait for it to complete before starting another."}
        units = history.load_units(job_id, DATA_DIR)
        # A shot is selected by number; a user-picked range is selected by its
        # own unit id ("C1"), so both forms are accepted here.
        selected = set()
        for x in (shot_ids or []):
            try:
                selected.add(int(x))
            except (TypeError, ValueError):
                selected.add(str(x))
        selected_units = {u["unit_id"] for u in units
                          if u.get("unit_id") in selected
                          or any(s in selected for s in u.get("shot_ids", []))}
        if not selected_units:
            return {"task_id": None, "status": "error",
                    "error": "Selection maps to no AD units"}
        task_id = str(uuid.uuid4())
        self.processing_status[task_id] = {"status": "processing", "step": "initializing"}
        thread = threading.Thread(
            target=self._reprocess_task,
            args=(task_id, job_id, sorted(selected_units, key=str), options),
            daemon=True)
        thread.start()
        return {"task_id": task_id, "status": "started"}

    def _reprocess_task(self, task_id, job_id, selected_unit_ids, options):
        status = self.processing_status[task_id]
        extractors = {
            "base": self._extract_frames,
            "context_shots": self._get_context_shots,
            "context": self._extract_frames_with_context,
        }
        history.reprocess_task(
            task_id, job_id, selected_unit_ids, options, DATA_DIR,
            status, logger, extractors=extractors, emit=self.emit_progress,
            acquire=self._try_acquire_task, release=self._release_task)

    def delete_history(self, job_id):
        import shutil
        job_dir = history.job_dir(job_id, DATA_DIR)
        if not os.path.isdir(job_dir):
            return {"ok": False}
        shutil.rmtree(job_dir, ignore_errors=True)
        return {"ok": True}

    def reveal_run(self, job_id, run):
        """Open a run's folder in Explorer."""
        d = history.run_dir(job_id, run, DATA_DIR)
        if not os.path.isdir(d):
            return False
        os.startfile(d)
        return True

    def get_data_dir(self):
        return {"path": DATA_DIR}

    def reveal_appdata(self):
        """Open the app data folder (root outputs dir) in Explorer."""
        os.makedirs(DATA_DIR, exist_ok=True)
        os.startfile(DATA_DIR)
        return True


def main():
    """Start the desktop application."""
    logger.info(f"Startup begin (frozen={FROZEN}, base_dir={BASE_DIR}, data_dir={DATA_DIR})")

    if not _webview2_runtime_installed():
        _warn_no_webview2()
        sys.exit(1)

    bridge = AppBridge()
    
    # Get the frontend HTML path
    html_path = os.path.join(BASE_DIR, "templates", "desktop.html")
    
    if not os.path.exists(html_path):
        # Fallback to index.html
        html_path = os.path.join(BASE_DIR, "templates", "index.html")
    
    # Read HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Load translations.js inline: the HTML is handed to pywebview as a raw
    # string whose base URL does not resolve /static/ assets. Keep the runtime
    # identifier consistent with the web version so both share one i18n source.
    i18n_path = os.path.join(BASE_DIR, "static", "translations.js")
    if os.path.exists(i18n_path):
        with open(i18n_path, 'r', encoding='utf-8') as f:
            i18n_js = f.read()
        # Insert before the page's own <script> so I18N is defined first.
        html_content = html_content.replace(
            "<script>", f"<script>\n{i18n_js}\n</script>\n<script>", 1)
    
    # Create window with HTML content
    window = webview.create_window(
        title="Shot-by-Shot Audio Describer",
        html=html_content,
        js_api=bridge,
        width=1200,
        height=800,
        min_size=(800, 600),
        on_top=True,
    )
    
    bridge.attach_window(window)

    # Deterministic WebView2 user-data folder: pre-created, writable, and
    # survives across runs so first-run profile provisioning only happens once.
    webview_storage = os.path.join(DATA_DIR, "WebView2")
    os.makedirs(webview_storage, exist_ok=True)

    # Optional per-machine workaround: set SBS_BROWSER_ARGS=--disable-gpu
    # (e.g. on VMs/RDP) before launching; passed via the official WebView2
    # environment variable.
    browser_args = os.environ.get("SBS_BROWSER_ARGS", "").strip()
    if browser_args:
        merged = (os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "") + " " + browser_args).strip()
        os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = merged
        logger.info(f"Additional browser args: {merged}")

    watchdog_stop = _start_window_watchdog()
    logger.info("Starting webview loop (EdgeChromium)")
    webview.start(debug=False, private_mode=False, storage_path=webview_storage)
    watchdog_stop.set()
    logger.info("Webview loop ended")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Try using the web version instead: python app.py")
