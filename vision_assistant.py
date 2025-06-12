#!/usr/bin/env python3
"""
Complete Python-Only Vision Assistant for Visually Impaired
Combines object detection, distance estimation, and TTS in one program
"""

import cv2
import json
import time
import numpy as np
import os
import sys
import threading
import queue
from ultralytics import YOLO

# TTS Options - choose one based on your preference and system

# Option 1: Piper TTS (same as your C++ version, but called from Python)
import subprocess
import tempfile

# Option 2: gTTS (Google Text-to-Speech) - requires internet
# from gtts import gTTS
# import pygame

# Option 3: pyttsx3 (offline, cross-platform)
# import pyttsx3

# Option 4: espeak (Linux only, very lightweight)
# import subprocess

class TTSEngine:
    """Text-to-Speech engine with multiple backend options"""
    
    def __init__(self, engine_type="piper"):
        self.engine_type = engine_type
        self.setup_engine()
        
    def setup_engine(self):
        if self.engine_type == "piper":
            # Using your existing Piper setup
            self.piper_cmd = [
                "./piper/piper",
                "--model", "./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx",
                "--config", "./piper/voices/en_US-amy-medium/en_US-amy-medium.onnx.json",
                "--json-input"
            ]
            
        elif self.engine_type == "pyttsx3":
            import pyttsx3
            self.engine = pyttsx3.init()
            # Configure voice properties
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)  # Use first available voice
            self.engine.setProperty('rate', 150)  # Speed of speech
            self.engine.setProperty('volume', 0.9)  # Volume level
            
        elif self.engine_type == "gtts":
            import pygame
            pygame.mixer.init()
            
        elif self.engine_type == "espeak":
            # espeak is usually pre-installed on Linux
            pass
    
    def speak(self, text):
        """Speak the given text using the selected TTS engine"""
        try:
            if self.engine_type == "piper":
                self._speak_piper(text)
            elif self.engine_type == "pyttsx3":
                self._speak_pyttsx3(text)
            elif self.engine_type == "gtts":
                self._speak_gtts(text)
            elif self.engine_type == "espeak":
                self._speak_espeak(text)
        except Exception as e:
            print(f"TTS Error: {e}", file=sys.stderr)
            # Fallback to espeak if available
            try:
                self._speak_espeak(text)
            except:
                print(f"Speaking: {text}", file=sys.stderr)  # At least print it
    
    def _speak_piper(self, text):
        """Use Piper TTS (your current setup)"""
        json_input = json.dumps({"text": text})
        
        # Create temporary files for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_file = tmp_file.name
        
        try:
            # Run Piper
            cmd = self.piper_cmd + ["--output_file", audio_file]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(input=json_input.encode())
            
            if process.returncode == 0:
                # Play audio file
                subprocess.run(["aplay", audio_file], check=True)
            else:
                print(f"Piper error: {stderr.decode()}", file=sys.stderr)
                
        finally:
            # Clean up temp file
            if os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def _speak_pyttsx3(self, text):
        """Use pyttsx3 (offline, cross-platform)"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def _speak_gtts(self, text):
        """Use Google TTS (requires internet)"""
        from gtts import gTTS
        import pygame
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            audio_file = tmp_file.name
        
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(audio_file)
            
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        finally:
            if os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def _speak_espeak(self, text):
        """Use espeak (Linux, very lightweight)"""
        subprocess.run(["espeak", text], check=True)

class ImprovedDistanceEstimator:
    """Improved distance estimation (from previous artifact)"""
    
    def __init__(self):
        # Real-world object heights (your existing data)
        self.REAL_HEIGHTS = {
            "Person": 1.70, "Car": 1.50, "Chair": 0.90, "Bicycle": 1.10,
            "Motorcycle": 1.30, "Bus": 3.00, "Truck": 3.50, 
            "Traffic Light": 3.00, "Stop Sign": 2.10, "Bench": 0.45,
            # ... add all your existing heights
        }
        
        # Real-world widths for dual estimation
        self.REAL_WIDTHS = {
            "Person": 0.50, "Car": 1.80, "Bicycle": 0.60, "Bus": 2.50,
            "Truck": 2.50, "Chair": 0.60, "Motorcycle": 0.80,
            "Stop Sign": 0.75, "Bench": 1.50
        }
        
        # Distance history for smoothing
        self.distance_history = {}
    
    def estimate_distance(self, box, img_h, img_w, label):
        """Improved distance estimation with multiple methods"""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        h_px = float(y2 - y1)
        w_px = float(x2 - x1)
        
        # Better focal length estimation
        focal_px = img_w / (2 * np.tan(np.radians(35)))  # ~70 deg FOV
        
        estimates = []
        
        # Method 1: Height-based estimation
        real_h = self.REAL_HEIGHTS.get(label)
        if real_h:
            distance_h = (real_h * focal_px) / h_px
            estimates.append(distance_h)
        
        # Method 2: Width-based estimation
        real_w = self.REAL_WIDTHS.get(label)
        if real_w:
            distance_w = (real_w * focal_px) / w_px
            estimates.append(distance_w)
        
        # Combine estimates or use fallback
        if estimates:
            if len(estimates) == 2:
                # Weight height more than width (usually more reliable)
                distance = estimates[0] * 0.7 + estimates[1] * 0.3
            else:
                distance = estimates[0]
        else:
            # Fallback for unknown objects
            apparent_size = (h_px * w_px) / (img_h * img_w)
            if apparent_size > 0.2:
                distance = 2.0
            elif apparent_size > 0.1:
                distance = 5.0
            elif apparent_size > 0.05:
                distance = 10.0
            else:
                distance = 20.0
        
        # Ground plane correction for ground objects
        if label in ["Person", "Car", "Bicycle", "Chair", "Dog", "Cat"]:
            camera_height = 1.5  # Adjust based on your camera mounting
            box_bottom_y = float(y2)
            img_center_y = img_h / 2
            
            if box_bottom_y > img_center_y:
                ground_factor = 1.0 + 0.3 * (box_bottom_y - img_center_y) / img_center_y
                distance *= ground_factor
        
        # Apply temporal smoothing
        distance = self._smooth_distance(label, distance)
        
        return max(0.5, min(distance, 100.0))
    
    def _smooth_distance(self, label, distance, max_history=3):
        """Simple temporal smoothing to reduce jitter"""
        if label not in self.distance_history:
            self.distance_history[label] = []
        
        self.distance_history[label].append(distance)
        if len(self.distance_history[label]) > max_history:
            self.distance_history[label].pop(0)
        
        return sum(self.distance_history[label]) / len(self.distance_history[label])

class VisionAssistant:
    """Main vision assistant class"""
    
    def __init__(self, tts_engine="piper", model_path="./Insight/insight_deploy/models/yolo11n_object365.pt"):
        print("[INFO] Initializing Vision Assistant...")
        
        # Initialize components
        self.tts = TTSEngine(tts_engine)
        self.distance_estimator = ImprovedDistanceEstimator()
        
        # Load YOLO model
        self.model = YOLO(model_path, task="detect")
        self.labels = self.model.names
        
        # Configuration
        self.conf_threshold = 0.45
        self.near_threshold = 50.0  # meters
        self.detection_interval = 1.0  # seconds between detections
        
        # Initialize camera
        self.setup_camera()
        
        # Threading for non-blocking TTS
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        print("[INFO] Vision Assistant initialized successfully!")
    
    def setup_camera(self):
        """Initialize camera with optimal settings"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Warm up camera
        for _ in range(5):
            self.cap.read()
        
        print("[INFO] Camera initialized")
    
    def _tts_worker(self):
        """Background worker for TTS to avoid blocking"""
        while True:
            try:
                text = self.tts_queue.get(timeout=1)
                if text is None:  # Shutdown signal
                    break
                self.tts.speak(text)
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}", file=sys.stderr)
    
    def speak_async(self, text):
        """Add text to TTS queue for non-blocking speech"""
        try:
            # Clear queue if it's getting too long (skip old messages)
            while self.tts_queue.qsize() > 2:
                try:
                    self.tts_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.tts_queue.put(text)
        except Exception as e:
            print(f"Error queuing TTS: {e}", file=sys.stderr)
    
    def create_response_text(self, nearby_objects):
        """Create natural language response for detected objects"""
        if not nearby_objects:
            return None
        
        if len(nearby_objects) == 1:
            dist, label = nearby_objects[0]
            return f"There is a {label} approximately {dist:.0f} metres ahead."
        
        # Sort by distance
        nearby_objects.sort(key=lambda x: x[0])
        
        if len(nearby_objects) == 2:
            obj1, obj2 = nearby_objects
            return f"There is a {obj1[1]} approximately {obj1[0]:.0f} metres ahead, and a {obj2[1]} at {obj2[0]:.0f} metres."
        
        # For 3+ objects
        parts = []
        for i, (dist, label) in enumerate(nearby_objects):
            if i == 0:
                parts.append(f"There is a {label} at {dist:.0f} metres")
            elif i == len(nearby_objects) - 1:
                parts.append(f"and a {label} at {dist:.0f} metres ahead")
            else:
                parts.append(f"a {label} at {dist:.0f} metres")
        
        return ", ".join(parts) + "."
    
    def process_frame(self):
        """Process single frame and return detection results"""
        # Flush camera buffer to get fresh frame
        for _ in range(3):
            self.cap.read()
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Run YOLO detection
        results = self.model(frame, imgsz=640, conf=self.conf_threshold)[0]
        
        nearby_objects = []
        all_objects = []
        
        for box in results.boxes:
            label = self.labels[int(box.cls[0])]
            distance = self.distance_estimator.estimate_distance(
                box, frame.shape[0], frame.shape[1], label
            )
            
            all_objects.append((distance, label))
            
            if distance <= self.near_threshold:
                nearby_objects.append((distance, label))
        
        # Debug output
        print(f"DEBUG: Detected {len(all_objects)} total objects", file=sys.stderr)
        for dist, label in all_objects:
            print(f"DEBUG: {label} at {dist:.1f}m", file=sys.stderr)
        print(f"DEBUG: {len(nearby_objects)} objects within {self.near_threshold}m", file=sys.stderr)
        
        return nearby_objects
    
    def run(self):
        """Main loop"""
        print("[INFO] Starting Vision Assistant...")
        
        # Power-on announcement
        self.speak_async("Power on, Insight is your assistant")
        
        try:
            while True:
                nearby_objects = self.process_frame()
                
                if nearby_objects:
                    response_text = self.create_response_text(nearby_objects)
                    if response_text:
                        print(f"[INFO] {response_text}")
                        self.speak_async(response_text)
                
                time.sleep(self.detection_interval)
                
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down Vision Assistant...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        self.tts_queue.put(None)  # Signal TTS worker to stop
        self.tts_thread.join(timeout=2)
        print("[INFO] Vision Assistant stopped")

def main():
    """Main function with configuration options"""
    
    # Configuration - adjust these based on your setup
    TTS_ENGINE = "piper"  # Options: "pyttsx3", "piper", "gtts", "espeak"
    MODEL_PATH = "./Insight/insight_deploy/models/yolo11n_object365.pt"
    
    # Alternative TTS engines you can try:
    # TTS_ENGINE = "piper"    # Your current setup, high quality
    # TTS_ENGINE = "gtts"     # Google TTS, requires internet, very natural
    # TTS_ENGINE = "espeak"   # Very lightweight, robotic voice
    
    try:
        assistant = VisionAssistant(tts_engine=TTS_ENGINE, model_path=MODEL_PATH)
        assistant.run()
    except Exception as e:
        print(f"[ERROR] Failed to start Vision Assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()