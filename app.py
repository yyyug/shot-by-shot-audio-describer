"""
Flask web server for Shot-by-Shot Video Processor
"""
import os
import uuid
import json
import threading
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, Response, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from processing.shot_detector import detect_shots
from processing.whisper_transcriber import transcribe_video
from processing.vlm_describer import describe_frames
from processing.llm_summarizer import batch_summarize, estimate_word_limit
from processing.csv_merger import merge_shots_subtitles, merge_with_descriptions
from processing.film_grammar import get_effective_shot_scale, select_prompt_variant
from processing.character_recognizer import detect_faces, extract_face_embeddings, cluster_faces

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

processing_status = {}


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


def process_video_task(task_id, video_path, options):
    status = {"status": "processing", "step": "initializing", "progress": 0}
    processing_status[task_id] = status
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
        shots = detect_shots(video_path)
        status["shots_count"] = len(shots)
        status["progress"] = 20
        status["detail"] = f"Found {len(shots)} shots"
        status["step"] = "transcription"
        if options.get("use_whisper", True):
            try:
                subtitles = transcribe_video(video_path, language="auto")
                status["detail"] = f"Transcribed {len(subtitles)} segments"
            except:
                subtitles = []
                status["detail"] = "Transcription failed, using empty subtitles"
        else:
            subtitles = []
            status["detail"] = "Skipped"
        status["progress"] = 40
        status["step"] = "merging"
        status["detail"] = "Merging shots with subtitles..."
        merged_df = merge_shots_subtitles(shots, subtitles)
        status["progress"] = 50
        descriptions_dict = {}
        if api_key:
            status["step"] = "stage1_vlm"
            status["detail"] = "Starting VLM descriptions..."
            shot_scales = [2] * len(shots)
            threads = [[i] for i in range(len(shots))]
            for i, shot in enumerate(shots):
                try:
                    frames_b64 = extract_frames_base64(video_path, shot)
                    film_grammar = {"video_type": options.get("video_type", "movie"), "label_type": "none",
                                    "char_text": "", "current_shots": [shot["shot_id"]-1], "threads": threads,
                                    "shot_scales": shot_scales, "prompt_variant": 4}
                    desc = describe_frames(frames_b64, api_key, backend=backend, film_grammar=film_grammar)
                    descriptions_dict[shot["shot_id"]] = desc
                    status["detail"] = f"Described shot {i+1}/{len(shots)}"
                except Exception as e:
                    descriptions_dict[shot["shot_id"]] = ""
                    status["detail"] = f"Shot {i+1} failed: {str(e)[:50]}"
                status["progress"] = 50 + int((i/len(shots))*25)
            merged_df = merge_with_descriptions(merged_df, [{"shot_id": k, "description": v} for k, v in descriptions_dict.items()])
        status["detail"] = f"Described {len(descriptions_dict)} shots"
        status["progress"] = 80
        if not options.get("skip_stage2") and api_key and descriptions_dict:
            status["step"] = "stage2_summarize"
            status["detail"] = "Summarizing audio descriptions..."
            stage1_results = [{"shot_id": s["shot_id"], "start": s["start_time"], "end": s["end_time"],
                               "description": descriptions_dict.get(s["shot_id"], "")} for s in shots]
            stage2_results = batch_summarize(stage1_results, api_key, backend=backend,
                                            video_type=options.get("video_type", "movie"))
            pd.DataFrame(stage1_results).to_csv(os.path.join(app.config['OUTPUT_FOLDER'], f"{task_id}_stage1.csv"), index=False)
            pd.DataFrame(stage2_results).to_csv(os.path.join(app.config['OUTPUT_FOLDER'], f"{task_id}_stage2.csv"), index=False)
            status["stage1_path"] = f"{task_id}_stage1.csv"
            status["stage2_path"] = f"{task_id}_stage2.csv"
        status["progress"] = 100
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{task_id}.csv")
        merged_df.to_csv(output_path, index=False)
        status["status"] = "completed"
        status["output_path"] = f"{task_id}.csv"
    except Exception as e:
        status["status"] = "failed"
        status["error"] = str(e)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    file.save(video_path)
    options = {"backend": request.form.get("backend", "gemini"),
               "api_key": request.form.get("api_key") or request.form.get("gemini_key") or request.form.get("openrouter_key"),
               "video_type": request.form.get("video_type", "movie"),
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
    if not status or status.get("status") != "completed":
        return jsonify({"error": "Not ready"}), 404
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], status["output_path"]),
                     as_attachment=True, download_name=f"shot_by_shot_{task_id[:8]}.csv")

if __name__ == '__main__':
    app.run(debug=False, port=5000)
