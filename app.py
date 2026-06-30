import os
import uuid
import json
import threading
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

from processing.shot_detector import detect_shots
from processing.whisper_transcriber import transcribe_video
from processing.vlm_describer import describe_frames
from processing.csv_merger import merge_shots_subtitles, merge_with_descriptions, format_original_style

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store processing status
processing_status = {}

def process_video_task(task_id, video_path, options):
    """Background task for video processing."""
    try:
        processing_status[task_id] = {"status": "processing", "step": "shot_detection", "progress": 0}
        
        # Step 1: Shot detection
        shots = detect_shots(video_path)
        processing_status[task_id]["progress"] = 25
        processing_status[task_id]["shots_count"] = len(shots)
        
        # Step 2: Transcription
        processing_status[task_id]["step"] = "transcription"
        try:
            subtitles = transcribe_video(
                video_path,
                options.get("whisper_path", "whisper-cpp"),
                options.get("model_path"),
                options.get("language")
            )
        except RuntimeError:
            subtitles = []
        processing_status[task_id]["progress"] = 50
        
        # Step 3: Merge
        processing_status[task_id]["step"] = "merging"
        merged_df = merge_shots_subtitles(shots, subtitles)
        processing_status[task_id]["progress"] = 75
        
        # Step 4: VLM (optional)
        openrouter_key = options.get("openrouter_key")
        if openrouter_key:
            processing_status[task_id]["step"] = "vlm_description"
            descriptions = []
            for shot in shots:
                try:
                    frames_b64 = extract_frames_base64(video_path, shot)
                    desc = describe_frames(frames_b64, openrouter_key)
                    descriptions.append({"shot_id": shot["shot_id"], "description": desc})
                except Exception:
                    descriptions.append({"shot_id": shot["shot_id"], "description": ""})
            merged_df = merge_with_descriptions(merged_df, descriptions)
        
        processing_status[task_id]["progress"] = 100
        
        # Format output
        fmt = options.get("format", "basic")
        if fmt == "original":
            output_df = format_original_style(merged_df)
        elif fmt == "full":
            output_df = merged_df
        else:
            output_df = merged_df[["shot_id", "start_time", "end_time", "subtitle"]]
        
        # Save
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{task_id}.csv")
        output_df.to_csv(output_path, index=False)
        
        processing_status[task_id]["status"] = "completed"
        processing_status[task_id]["output_path"] = output_path
        
    except Exception as e:
        processing_status[task_id]["status"] = "failed"
        processing_status[task_id]["error"] = str(e)

def extract_frames_base64(video_path, shot, num_frames=8):
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

@app.route('/')
def index():
    return app.send_static_file('index.html') if os.path.exists('static/index.html') else "Shot-by-Shot Processing"

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    task_id = str(uuid.uuid4())
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
    file.save(video_path)
    
    options = {
        "whisper_path": request.form.get("whisper_path", "whisper-cpp"),
        "model_path": request.form.get("model_path"),
        "language": request.form.get("language"),
        "openrouter_key": request.form.get("openrouter_key"),
        "format": request.form.get("format", "basic")
    }
    
    # Start processing in background
    thread = threading.Thread(target=process_video_task, args=(task_id, video_path, options))
    thread.start()
    
    return jsonify({"task_id": task_id, "status": "started"})

@app.route('/progress/<task_id>')
def progress(task_id):
    def generate():
        while True:
            status = processing_status.get(task_id, {"status": "unknown"})
            yield f"data: {json.dumps(status)}\n\n"
            
            if status.get("status") in ["completed", "failed"]:
                break
            
            import time
            time.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/download/<task_id>')
def download(task_id):
    status = processing_status.get(task_id)
    if not status or status.get("status") != "completed":
        return jsonify({"error": "File not ready"}), 404
    
    return send_file(
        status["output_path"],
        as_attachment=True,
        download_name=f"shot_by_shot_{task_id[:8]}.csv"
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)