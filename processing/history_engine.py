"""Job history + reprocess engine (shared by the Flask app and the pywebview
desktop app).

Layout under DATA_DIR (root outputs dir; %LOCALAPPDATA%\\BuddyAd when frozen):

    history/<job_id>/
        job.json            # neutral metadata ONLY (no api url / model / key)
        shots.json          # detected shots
        subtitles.json      # transcription segments
        units.json          # AD units (dialogue-gap intervals, shots, or user ranges)
        frames/unit_<id>/NNNN.jpg  # exact image payload sent per unit
        frames_manifest.json       # provenance (rules / model / counts)
        units.jsonl         # one JSON request per unit, for external agents
        thumbs/<shot_id>.jpg  # one representative thumbnail per shot
        thumbs/<unit_id>.jpg  # ... and one per user-picked range ("C1", ...)
        runs/<n>/
            meta.json       # run timestamp / status / success counts
            stage1.json     # {unit_id: {start,end,shot_ids,description}}
            stage2.json     # {unit_id: ad_sentence}
            DetailsDescription.csv / AD.csv / final.csv / final.vtt

The source video is deliberately NOT archived - the per-unit frames above are
what a replay (or an external agent) needs. Prompts are not stored either; they
are rebuilt deterministically from the unit data.

A reprocess ("重生執行") only re-describes the units that contain the selected
shots, reuses the descriptions of every other unit from the current run, then
re-runs stage 2 over the complete description set and snapshots everything into
a NEW run directory (previous runs are kept; the newest is "current").
"""
import os
import json
import shutil
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# persistence helpers
# ---------------------------------------------------------------------------

def history_dir(data_dir):
    return os.path.join(data_dir, "history")


def job_dir(job_id, data_dir):
    return os.path.join(history_dir(data_dir), str(job_id))


def job_file(job_id, data_dir, *parts):
    return os.path.join(job_dir(job_id, data_dir), *parts)


def run_dir(job_id, run, data_dir):
    return os.path.join(job_dir(job_id, data_dir), "runs", str(run))


def run_file(job_id, run, data_dir, name):
    return os.path.join(run_dir(job_id, run, data_dir), name)


def _atomic_write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


# ---------------------------------------------------------------------------
# job lifecycle
# ---------------------------------------------------------------------------

def create_job(job_id, filename, data_dir, **meta):
    """Register a new job. meta must only contain non-sensitive neutral data
    (page/book state); never api url / model / api key."""
    os.makedirs(job_dir(job_id, data_dir), exist_ok=True)
    rec = {
        "job_id": job_id,
        "filename": filename,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_run": 0,
        "runs": [0],
    }
    rec.update(meta)
    _atomic_write(job_file(job_id, data_dir, "job.json"), rec)
    return rec


def load_job(job_id, data_dir):
    return _read_json(job_file(job_id, data_dir, "job.json"))


def list_jobs(data_dir):
    """All job records, newest first."""
    root = history_dir(data_dir)
    if not os.path.isdir(root):
        return []
    jobs = []
    for name in os.listdir(root):
        if name.startswith("."):
            continue
        cand = os.path.join(root, name)
        if os.path.isdir(cand):
            rec = load_job(name, data_dir)
            if rec:
                jobs.append(rec)
    jobs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return jobs


def update_job(job_id, data_dir, **fields):
    rec = load_job(job_id, data_dir)
    if rec is None:
        return None
    rec.update(fields)
    _atomic_write(job_file(job_id, data_dir, "job.json"), rec)
    return rec


def set_current_run(job_id, run, data_dir):
    rec = update_job(job_id, data_dir, current_run=run)
    if rec is not None and run not in rec.get("runs", []):
        rec["runs"] = list(rec.get("runs", [])) + [run]
        _atomic_write(job_file(job_id, data_dir, "job.json"), rec)


def resolve_source_video(job_id, data_dir):
    """Return the on-disk video for a job: archived copy (web) or the local
    path recorded on the desktop app (source_path)."""
    job = load_job(job_id, data_dir)
    if not job:
        return None
    if job.get("source_path"):
        if os.path.isfile(job["source_path"]):
            return job["source_path"]
    exts = [job.get("ext", ""), ".mp4", ".mkv", ".mov", ".avi", ".webm"]
    for ext in exts:
        cand = job_file(job_id, data_dir, "source" + ext)
        if os.path.isfile(cand):
            return cand
    return None


def save_shots(job_id, shots, data_dir):
    _atomic_write(job_file(job_id, data_dir, "shots.json"), {"shots": shots})


def save_subtitles(job_id, subtitles, data_dir):
    _atomic_write(job_file(job_id, data_dir, "subtitles.json"), {"subtitles": subtitles})


def save_units(job_id, units, data_dir):
    _atomic_write(job_file(job_id, data_dir, "units.json"), {"units": units})


def load_shots(job_id, data_dir):
    return _read_json(job_file(job_id, data_dir, "shots.json"), {}).get("shots", [])


def load_units(job_id, data_dir):
    return _read_json(job_file(job_id, data_dir, "units.json"), {}).get("units", [])


# ---------------------------------------------------------------------------
# unit frames + prompts - the replayable LLM payload
#
# The exact images sent to the VLM for each unit are written to disk in order,
# so a replay (and any external agent) never has to re-run shot detection,
# transcription or frame extraction. The source video itself is NOT kept.
# ---------------------------------------------------------------------------

FRAMES_DIR = "frames"
AGENT_JSONL = "units.jsonl"
FRAME_MANIFEST = "frames_manifest.json"
SCHEMA_VERSION = 1


def unit_frames_dir(job_id, unit_id, data_dir):
    return job_file(job_id, data_dir, FRAMES_DIR, f"unit_{unit_id}")


def _unit_frame_names(job_id, unit_id, data_dir):
    d = unit_frames_dir(job_id, unit_id, data_dir)
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if n.lower().endswith(".jpg"))


def has_unit_frames(job_id, unit_id, data_dir):
    return bool(_unit_frame_names(job_id, unit_id, data_dir))


def save_unit_frames(job_id, unit_id, frames_b64, data_dir):
    """Persist one unit's exact image payload, in order, as JPEG files."""
    import base64
    d = unit_frames_dir(job_id, unit_id, data_dir)
    os.makedirs(d, exist_ok=True)
    names = []
    for i, b64 in enumerate(frames_b64):
        name = f"{i + 1:04d}.jpg"
        with open(os.path.join(d, name), "wb") as f:
            f.write(base64.b64decode(b64))
        names.append(name)
    return names


def load_unit_frames(job_id, unit_id, data_dir):
    """Reload a unit's saved images as base64, in the original order."""
    import base64
    d = unit_frames_dir(job_id, unit_id, data_dir)
    out = []
    for name in _unit_frame_names(job_id, unit_id, data_dir):
        with open(os.path.join(d, name), "rb") as f:
            out.append(base64.b64encode(f.read()).decode("ascii"))
    return out


def save_thumbs_from_unit_frames(job_id, unit, frames_b64, data_dir, max_width=320):
    """Write the unit's thumbnail from its already-extracted frames.

    History previews must not depend on the source video (which is no longer
    kept), so one of the unit's frames is reused: the first frame for a
    user-picked range (that is the frame that identifies it), the middle frame
    for a shot. A range is keyed by its own unit id instead of by the shots it
    covers, so it never overwrites their previews.
    """
    custom = unit.get("mode") == "custom"
    key_ids = [unit.get("unit_id")] if custom else (unit.get("shot_ids") or [])
    if not key_ids or not frames_b64:
        return
    import base64
    import cv2
    import numpy as np
    try:
        index = 0 if custom else len(frames_b64) // 2
        arr = np.frombuffer(base64.b64decode(frames_b64[index]), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        h, w = frame.shape[:2]
        if max_width and w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (max_width, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        for sid in key_ids:
            path = thumb_path(job_id, sid, data_dir)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(buf.tobytes())
    except Exception as e:
        logger.warning(f"could not write thumbnails for unit {unit.get('unit_id')}: {e}")


def save_frame_manifest(job_id, data_dir, **meta):
    """Record how the frames were produced so a replay is byte-identical."""
    rec = {
        "schema_version": SCHEMA_VERSION,
        "frames_path": f"{FRAMES_DIR}/unit_<unit_id>/NNNN.jpg",
        "unit_jsonl": AGENT_JSONL,
    }
    rec.update(meta)
    _atomic_write(job_file(job_id, data_dir, FRAME_MANIFEST), rec)
    return rec


def load_frame_manifest(job_id, data_dir):
    return _read_json(job_file(job_id, data_dir, FRAME_MANIFEST), {})


def job_size_bytes(job_id, data_dir):
    """Total on-disk size of a job folder (frames dominate)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(job_dir(job_id, data_dir)):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def write_agent_jsonl(job_id, data_dir, units):
    """One JSON object per LLM request - the bundle an external agent consumes."""
    lines = []
    for u in units:
        uid = u["unit_id"]
        images = [f"{FRAMES_DIR}/unit_{uid}/{n}"
                  for n in _unit_frame_names(job_id, uid, data_dir)]
        lines.append({
            "unit_id": uid,
            "start": u.get("start", 0),
            "end": u.get("end", 0),
            "shot_ids": u.get("shot_ids", []),
            "mode": u.get("mode", ""),
            "images": images,
        })
    path = job_file(job_id, data_dir, AGENT_JSONL)
    with open(path, "w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------

def thumb_path(job_id, shot_id, data_dir):
    return job_file(job_id, data_dir, "thumbs", f"{shot_id}.jpg")


def ensure_thumbs(job_id, data_dir, video_path=None, max_width=320):
    """Generate one thumbnail per shot if missing (cached across runs)."""
    shots = load_shots(job_id, data_dir)
    missing = [s for s in shots
               if not os.path.isfile(thumb_path(job_id, s["shot_id"], data_dir))]
    if not missing:
        return shots
    if video_path is None:
        video_path = resolve_source_video(job_id, data_dir)
    if not video_path or not os.path.isfile(video_path):
        return shots
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        for s in missing:
            mid = (s.get("start_time", 0) + s.get("end_time", 0)) / 2.0
            cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if max_width and w > max_width:
                scale = max_width / float(w)
                frame = cv2.resize(frame, (max_width, int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                path = thumb_path(job_id, s["shot_id"], data_dir)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
    finally:
        cap.release()
    return shots


def thumb_data_url(job_id, shot_id, data_dir):
    import base64
    path = thumb_path(job_id, shot_id, data_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")


# ---------------------------------------------------------------------------
# run persistence
# ---------------------------------------------------------------------------

def begin_run(job_id, run, data_dir, **meta):
    rec = {
        "run": run,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
    }
    rec.update(meta)
    _atomic_write(run_file(job_id, run, data_dir, "meta.json"), rec)
    return rec


def update_run(job_id, run, data_dir, **fields):
    path = run_file(job_id, run, data_dir, "meta.json")
    rec = _read_json(path, {})
    rec.update(fields)
    rec.setdefault("run", run)
    _atomic_write(path, rec)


def load_run_meta(job_id, run, data_dir):
    return _read_json(run_file(job_id, run, data_dir, "meta.json"), {})


def load_stage1(job_id, run, data_dir):
    """{unit_id(str): {start,end,shot_ids,mode,description}}"""
    d = _read_json(run_file(job_id, run, data_dir, "stage1.json"), {})
    return d if isinstance(d, dict) else {}


def record_unit(job_id, run, data_dir, unit, description):
    """Incrementally persist one unit's description into a run's stage1.json."""
    path = run_file(job_id, run, data_dir, "stage1.json")
    d = load_stage1(job_id, run, data_dir)
    d[str(unit["unit_id"])] = {
        "start": unit.get("start", 0),
        "end": unit.get("end", 0),
        "shot_ids": unit.get("shot_ids", []),
        "mode": unit.get("mode", ""),
        "description": description,
    }
    _atomic_write(path, d)


def save_stage2(job_id, run, data_dir, ad_sentence_map):
    _atomic_write(run_file(job_id, run, data_dir, "stage2.json"),
                  {str(k): v for k, v in ad_sentence_map.items()})


def load_stage2(job_id, run, data_dir):
    d = _read_json(run_file(job_id, run, data_dir, "stage2.json"), {})
    return d if isinstance(d, dict) else {}


def write_run_files(job_id, run, data_dir, stage1_results, stage2_results,
                    ad_sentence_map, units, mirror_root=True, timestamp=None):
    """Write the canonical CSV/VTT set into runs/<n>/ and (optionally) mirror
    timestamped copies at DATA_DIR root so the "Outputs" folder stays usable."""
    import pandas as pd
    from processing.vtt_writer import write_vtt
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _write_csv(run_name, root_name, df):
        df.to_csv(run_file(job_id, run, data_dir, run_name),
                  index=False, encoding="utf-8-sig")
        if mirror_root:
            df.to_csv(os.path.join(data_dir, root_name),
                      index=False, encoding="utf-8-sig")

    _write_csv("DetailsDescription.csv",
               f"{timestamp}_DetailsDescription.csv", pd.DataFrame(stage1_results))

    if stage2_results is not None:
        _write_csv("AD.csv", f"{timestamp}_AD.csv", pd.DataFrame(stage2_results))

    output_df = pd.DataFrame([{
        "shot_id": u["unit_id"],
        "start": u.get("start", 0),
        "end": u.get("end", 0),
        "ad_sentence": ad_sentence_map.get(u["unit_id"], ""),
    } for u in units])
    _write_csv("final.csv", f"{timestamp}-final.csv", output_df)

    vtt_rows = output_df.to_dict("records")
    write_vtt(run_file(job_id, run, data_dir, "final.vtt"), vtt_rows)
    if mirror_root:
        write_vtt(os.path.join(data_dir, f"{timestamp}-final.vtt"), vtt_rows)

    return timestamp


def _api_common():
    import processing._api_common as m
    return m


# ---------------------------------------------------------------------------
# reprocess ("重生執行")
# ---------------------------------------------------------------------------

def build_film_grammar(video_type, custom_opening, unit, shots, lang=None):
    from processing.time_ranges import current_shot_indices
    return {
        "video_type": video_type,
        "label_type": "none",
        "char_text": "",
        "current_shots": current_shot_indices(unit, shots),
        "threads": [[j for j in range(len(shots))]],
        "shot_scales": [2] * len(shots),
        "prompt_variant": 4,
        "custom_opening": (custom_opening or "").strip() or None,
        "lang": lang,
    }


def describe_units(units, video_path, job_id, run, data_dir, backend, api_key,
                   openai_url, openai_model, video_type, custom_opening, lang,
                   use_context_extender, shots, status, logger, extractors,
                   model=None, emit=None, progress_start=50):
    """Describe each unit, persisting results incrementally into
    runs/<run>/stage1.json. Returns (descriptions, error_categories)."""
    from processing.vlm_describer import describe_frames
    descriptions = {}
    categories = {}
    total = len(units)
    for i, unit in enumerate(units):
        try:
            # Prefer the persisted payload: a replay must never touch the video.
            frames_b64 = load_unit_frames(job_id, unit["unit_id"], data_dir)
            if not frames_b64:
                frame_shot = {"start_time": unit["start"], "end_time": unit["end"]}
                if use_context_extender and shots:
                    context_shots = extractors["context_shots"](shots, unit.get("shot_ids", []))
                    frames_b64 = extractors["context"](video_path, frame_shot, context_shots)
                else:
                    frames_b64 = extractors["base"](video_path, frame_shot)
                save_unit_frames(job_id, unit["unit_id"], frames_b64, data_dir)
            film_grammar = build_film_grammar(video_type, custom_opening, unit, shots, lang)
            desc = describe_frames(frames_b64, api_key, backend=backend,
                                   film_grammar=film_grammar,
                                   openai_url=openai_url, openai_model=openai_model,
                                   model=model)
            descriptions[unit["unit_id"]] = desc or ""
            logger.info(f"reprocess unit {i + 1}/{total} done ({len(desc or '')} chars)")
        except Exception as e:
            descriptions[unit["unit_id"]] = ""
            logger.error(f"reprocess unit {i + 1}/{total} failed: {e}", exc_info=True)
            cat = _api_common().categorize_error(e)
            categories[cat] = categories.get(cat, 0) + 1
        record_unit(job_id, run, data_dir, unit, descriptions[unit["unit_id"]])
        if i < total - 1:
            time.sleep(3)
        status["detail"] = f"Describing unit {i + 1}/{total}"
        status["progress"] = progress_start + int((i + 1) / total * 20)
        if emit:
            emit(status)
    return descriptions, categories


def reprocess_task(task_id, job_id, selected_unit_ids, options, data_dir,
                   status, logger, extractors, emit=None, acquire=None,
                   release=None):
    """Re-describe only the units containing the selected shots, reuse every
    other unit's description from the current run, then re-run stage 2 and
    snapshot everything into a fresh run directory."""
    if acquire and not acquire():
        status["status"] = "failed"
        status["error"] = "Another task is already processing. Wait for it to finish."
        if emit:
            emit(status)
        return None
    try:
        job = load_job(job_id, data_dir)
        if not job:
            status["status"] = "failed"
            status["error"] = "Job not found"
            if emit:
                emit(status)
            return None

        shots = load_shots(job_id, data_dir)
        units = load_units(job_id, data_dir)

        selected = set(selected_unit_ids)
        targets = [u for u in units if u["unit_id"] in selected]
        if not targets:
            raise RuntimeError("Selected shots map to no AD units.")

        # Persisted frames make the source video unnecessary; only fall back to
        # it for units whose payload was never saved (legacy jobs).
        video_path = resolve_source_video(job_id, data_dir)
        needs_video = any(not has_unit_frames(job_id, u["unit_id"], data_dir)
                          for u in targets)
        if needs_video and (not video_path or not os.path.isfile(video_path)):
            raise RuntimeError(
                "Saved frames are missing for the selected units and the source "
                "video is no longer available - cannot reprocess.")

        prev_run = job.get("current_run", 0)
        new_run = (max(job.get("runs", []) or [0])) + 1
        base = load_stage1(job_id, prev_run, data_dir)

        # Seed the new run's stage1 with the previous snapshot so partial
        # failures still keep the reuse of unselected units.
        prev_path = run_file(job_id, prev_run, data_dir, "stage1.json")
        new_path = run_file(job_id, new_run, data_dir, "stage1.json")
        if os.path.isfile(prev_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copyfile(prev_path, new_path)
        else:
            _atomic_write(new_path, {})

        begin_run(job_id, new_run, data_dir, kind="reprocess", selected_units=len(targets))
        set_current_run(job_id, new_run, data_dir)

        backend = options.get("backend", "gemini")
        api_key = options.get("api_key") or ""
        status["step"] = "stage1_vlm"
        status["detail"] = f"Re-describing {len(targets)} AD unit(s)..."
        status["progress"] = 40
        if emit:
            emit(status)

        _, error_categories = describe_units(
            targets, video_path, job_id, new_run, data_dir, backend, api_key,
            options.get("openai_url"), options.get("openai_model"),
            options.get("video_type", "movie"),
            options.get("custom_opening"),
            options.get("lang"),
            bool(options.get("use_context_extender")), shots, status, logger,
            extractors, model=options.get("model"), emit=emit, progress_start=50)

        # Merge: selected units get their fresh descriptions, everything else
        # keeps the previous run's text.
        merged = load_stage1(job_id, new_run, data_dir)
        stage1_results = []
        for u in units:
            rec = merged.get(str(u["unit_id"]), {})
            stage1_results.append({
                "shot_id": u["unit_id"],
                "start": u.get("start", 0),
                "end": u.get("end", 0),
                "description": rec.get("description", ""),
                "mode": u.get("mode", ""),
            })
        success_count = sum(1 for r in stage1_results if str(r["description"]).strip())
        stage1_ready = len(units) > 0 and success_count / len(units) >= 0.5
        status["detail"] = f"Described {success_count}/{len(units)} units"
        status["progress"] = 80
        if emit:
            emit(status)

        ad_sentence_map = {}
        stage2_results = None
        run_status = "completed"
        error_msg = None

        if not options.get("skip_stage2") and api_key and stage1_ready:
            status["step"] = "stage2_summarize"
            status["detail"] = "Summarizing AD sentences..."
            if emit:
                emit(status)
            from processing.llm_summarizer import batch_summarize
            stage1_for_stage2 = [{
                "shot_id": r["shot_id"], "start": r["start"], "end": r["end"],
                "description": r["description"], "mode": r.get("mode"),
            } for r in stage1_results]
            stage2_results = batch_summarize(
                stage1_for_stage2, api_key, backend=backend,
                video_type=options.get("video_type", "movie"),
                openai_url=options.get("openai_url"),
                openai_model=options.get("openai_model"),
                model=options.get("model"),
                lang=options.get("lang"),
                ad_chars_per_sec=options.get("ad_chars_per_sec"))
            ad_sentence_map = {r["shot_id"]: r["ad_sentence"] for r in stage2_results}
            save_stage2(job_id, new_run, data_dir, ad_sentence_map)
        elif api_key and not stage1_ready:
            run_status = "failed"
            _ac = _api_common()
            dominant = _ac.dominant_category(error_categories)
            error_msg = _ac.build_stage1_blocked_message(
                dominant, success_count, len(units))
            status["status"] = "failed"
            status["partial"] = True
            status["error"] = error_msg
        else:
            run_status = "completed" if success_count else "failed"
            if not success_count:
                status["status"] = "failed"
                status["error"] = "Stage 1 produced no descriptions (check API key / network)."

        if stage2_results is not None or status.get("status") != "failed":
            timestamp = write_run_files(
                job_id, new_run, data_dir, stage1_results, stage2_results,
                ad_sentence_map, units, mirror_root=True)
            update_run(job_id, new_run, data_dir,
                       status="completed", stage1_count=success_count,
                       stage2_count=len(ad_sentence_map))
            status["status"] = "completed"
            status["progress"] = 100
            status["job_id"] = job_id
            status["run"] = new_run
            status["stage1_path"] = f"{timestamp}_DetailsDescription.csv"
            if stage2_results is not None:
                status["stage2_path"] = f"{timestamp}_AD.csv"
                status["output_path"] = f"{timestamp}-final.csv"
                status["vtt_path"] = f"{timestamp}-final.vtt"
            logger.info(f"Job {job_id}: reprocess run {new_run} completed "
                        f"(stage1 {success_count}/{len(units)}, stage2 {len(ad_sentence_map)})")
        else:
            # still snapshot stage-1-only run (no stage 2)
            update_run(job_id, new_run, data_dir,
                       status="failed", stage1_count=success_count)
            if status.get("status") != "failed":
                status["status"] = "completed"
            logger.warning(f"Job {job_id}: reprocess run {new_run} finished "
                           f"with status={status.get('status')}")

        if emit:
            emit(status)
        return {"run": new_run, "status": status.get("status")}
    except Exception as e:
        status["status"] = "failed"
        status["error"] = str(e)
        logger.critical(f"Job {job_id}: reprocess failed: {e}", exc_info=True)
        if emit:
            emit(status)
        return None
    finally:
        if release:
            release(task_id)