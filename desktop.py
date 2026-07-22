"""
Desktop application using pywebview
"""
import os
import sys
import json
import uuid
import threading
import logging
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shot_by_shot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import webview, provide fallback message if not available
try:
    import webview
except ImportError:
    print("Error: pywebview not installed. Run: uv pip install pywebview --system")
    print("Alternatively, use the web version: python app.py")
    sys.exit(1)

from processing.shot_detector import detect_shots
from processing.whisper_transcriber import transcribe_video
from processing.vlm_describer import describe_frames
from processing.llm_summarizer import batch_summarize, estimate_word_limit
from processing.csv_merger import merge_shots_subtitles, merge_with_descriptions
from processing.film_grammar import get_effective_shot_scale, select_prompt_variant
from processing.character_recognizer import detect_faces, extract_face_embeddings, cluster_faces


class AppBridge:
    """Bridge between JavaScript frontend and Python backend."""
    
    def __init__(self):
        self.window = None
        self.processing_status = {}
    
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
        """Open file dialog to select video file."""
        import webview
        result = self.window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=('Video Files (*.mp4;*.mkv;*.avi;*.mov)',)
        )
        if result and len(result) > 0:
            return result[0]
        return None
    
    # Processing functions
    def process_video(self, video_path, options):
        """Start video processing in background thread."""
        task_id = str(uuid.uuid4())
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
        
        try:
            # Generate timestamp for file names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get video duration for progress display
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
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
            status["detail"] = f"Found {len(shots)} shots"
            status["progress"] = 20
            self.emit_progress(task_id, status)
            
            # Step 2: Transcription
            status["step"] = "transcription"
            status["detail"] = "Starting transcription..."
            self.emit_progress(task_id, status)
            
            if options.get("use_whisper", True):
                def trans_progress(progress, message):
                    status["detail"] = f"Transcription: {message}"
                    status["progress"] = 20 + progress * 10  # 20-30%
                    self.emit_progress(task_id, status)
                
                try:
                    subtitles = transcribe_video(video_path, language="auto", callback=trans_progress)
                    status["detail"] = f"Transcribed {len(subtitles)} segments"
                except:
                    subtitles = []
                    status["detail"] = "Transcription failed, using empty subtitles"
            else:
                subtitles = []
                status["detail"] = "Skipped"
            status["progress"] = 30
            self.emit_progress(task_id, status)
            
            # Step 3: Merge
            status["step"] = "merging"
            status["detail"] = "Merging shots with subtitles..."
            merged_df = merge_shots_subtitles(shots, subtitles)
            status["progress"] = 40
            self.emit_progress(task_id, status)
            
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
                status["detail"] = f"Starting VLM descriptions... ({len(shots)} shots to process)"
                logger.info(f"Starting VLM descriptions: {len(shots)} shots, backend={backend}, context={use_context}")
                self.emit_progress(task_id, status)
                
                descriptions_dict = {}
                for i, shot in enumerate(shots):
                    try:
                        logger.info(f"Processing shot {i+1}/{len(shots)}...")
                        
                        if use_context:
                            # Get context shots (2 before + current + 2 after)
                            context_shots = self._get_context_shots(shots, i)
                            status["detail"] = f"Describing shot {i+1}/{len(shots)} with {len(context_shots)} context shots"
                            frames_b64 = self._extract_frames_with_context(video_path, shot, context_shots)
                        else:
                            frames_b64 = self._extract_frames(video_path, shot)
                        
                        duration = shot["end_time"] - shot["start_time"]
                        num_frames = len(frames_b64)
                        status["detail"] = f"Describing shot {i+1}/{len(shots)} ({duration:.0f}s, {num_frames} frames)"
                        
                        logger.info(f"Calling Gemini API for shot {i+1} ({num_frames} frames)...")
                        
                        # Build film grammar for prompt selection
                        film_grammar = {
                            "video_type": options.get("video_type", "movie"),
                            "label_type": "none",
                            "char_text": "",
                            "current_shots": [i],
                            "threads": [[j for j in range(len(shots))]],
                            "shot_scales": [2] * len(shots),
                            "prompt_variant": None
                        }
                        
                        desc = describe_frames(
                            frames_b64, api_key, 
                            backend=backend, 
                            film_grammar=film_grammar,
                            openai_url=options.get("openai_url"),
                            openai_model=options.get("openai_model")
                        )
                        logger.info(f"Shot {i+1} completed: {len(desc)} chars")
                        descriptions_dict[shot["shot_id"]] = desc
                    except Exception as e:
                        logger.error(f"Shot {i+1} failed: {e}")
                        descriptions_dict[shot["shot_id"]] = ""
                        status["detail"] = f"Shot {i+1} failed: {str(e)[:50]}"
                    
                    # Rate limit protection: wait between API calls
                    if i < len(shots) - 1:
                        import time
                        time.sleep(3)  # Wait 3 seconds to stay under 15 RPM
                    
                    status["progress"] = 50 + (i / len(shots)) * 20
                    self.emit_progress(task_id, status)
                
                descriptions = [{"shot_id": k, "description": v} for k, v in descriptions_dict.items()]
                merged_df = merge_with_descriptions(merged_df, descriptions)
            else:
                status["detail"] = "Skipped (no API key)"
            
            status["detail"] = f"Described {len(descriptions_dict)} shots"
            status["progress"] = 80
            self.emit_progress(task_id, status)
            
            # Step 6: Stage 2 summarization
            if not options.get("skip_stage2") and api_key:
                status["step"] = "summarize"
                status["detail"] = "Summarizing audio descriptions..."
                self.emit_progress(task_id, status)
                
                stage1_results = []
                for shot in shots:
                    stage1_results.append({
                        "shot_id": shot["shot_id"],
                        "start": shot["start_time"],
                        "end": shot["end_time"],
                        "description": descriptions_dict.get(shot["shot_id"], "")
                    })
                
                stage2_results = batch_summarize(
                    stage1_results, api_key, backend=backend,
                    video_type=options.get("video_type", "movie")
                )
                
                # Save Stage 1 and Stage 2 outputs
                import pandas as pd
                output_dir = os.path.join(os.path.dirname(__file__), "outputs")
                os.makedirs(output_dir, exist_ok=True)
                
                stage1_df = pd.DataFrame(stage1_results)
                stage1_path = os.path.join(output_dir, f"{timestamp}_DetailsDescription.csv")
                stage1_df.to_csv(stage1_path, index=False)
                status["stage1_path"] = stage1_path
                
                stage2_df = pd.DataFrame(stage2_results)
                stage2_path = os.path.join(output_dir, f"{timestamp}_AD.csv")
                stage2_df.to_csv(stage2_path, index=False)
                status["stage2_path"] = stage2_path
            
            status["progress"] = 100
            status["status"] = "completed"
            
            # Save final output
            output_dir = os.path.join(os.path.dirname(__file__), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{timestamp}-final.csv")
            merged_df.to_csv(output_path, index=False)
            status["output_path"] = output_path
            
            self.emit_progress(task_id, status)
            
        except Exception as e:
            status["status"] = "failed"
            status["error"] = str(e)
            self.emit_progress(task_id, status)
    
    def _get_context_shots(self, shots, current_idx):
        """Get context shots: 2 before + current + 2 after."""
        context = []
        # Add 2 shots before
        for i in range(max(0, current_idx - 2), current_idx):
            context.append(shots[i])
        # Add current shot
        context.append(shots[current_idx])
        # Add 2 shots after
        for i in range(current_idx + 1, min(len(shots), current_idx + 3)):
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


def main():
    """Start the desktop application."""
    bridge = AppBridge()
    
    # Get the frontend HTML path
    html_path = os.path.join(os.path.dirname(__file__), "templates", "desktop.html")
    
    if not os.path.exists(html_path):
        # Fallback to index.html
        html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    
    # Read HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Create window with HTML content
    window = webview.create_window(
        title="Shot-by-Shot Video Processor",
        html=html_content,
        js_api=bridge,
        width=1200,
        height=800,
        min_size=(800, 600),
    )
    
    bridge.attach_window(window)
    
    # Start pywebview
    webview.start(debug=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Try using the web version instead: python app.py")
