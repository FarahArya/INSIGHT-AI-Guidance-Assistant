#!/usr/bin/env python3
"""
Complete Python-Only Vision Assistant for Visually Impaired
Fixes overlapping TTS issue by implementing speech-aware detection
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
import subprocess
import tempfile

class TTSEngine:
    """Text-to-Speech engine with speech state tracking"""
    
    def __init__(self, engine_type="piper"):
        self.engine_type = engine_type
        self.setup_engine()
        self.is_speaking = threading.Event()  # Track if TTS is active
        
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
            voices = self.engine.getProperty('voices')
            if voices:
                self.engine.setProperty('voice', voices[0].id)
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            
        elif self.engine_type == "gtts":
            import pygame
            pygame.mixer.init()
            
        elif self.engine_type == "espeak":
            pass
    
    def speak(self, text):
        """Speak the given text using the selected TTS engine"""
        self.is_speaking.set()  # Mark as speaking
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
            try:
                self._speak_espeak(text)
            except:
                print(f"Speaking: {text}", file=sys.stderr)
        finally:
            self.is_speaking.clear()  # Mark as finished speaking
    
    def _speak_piper(self, text):
        """Use Piper TTS (your current setup)"""
        json_input = json.dumps({"text": text})
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio_file = tmp_file.name
        
        try:
            # Run Piper
            cmd = self.piper_cmd + ["--output_file", audio_file]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(input=json_input.encode())
            
            if process.returncode == 0:
                # Play audio file and wait for completion
                subprocess.run(["aplay", audio_file], check=True)
            else:
                print(f"Piper error: {stderr.decode()}", file=sys.stderr)
                
        finally:
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
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        finally:
            if os.path.exists(audio_file):
                os.unlink(audio_file)
    
    def _speak_espeak(self, text):
        """Use espeak (Linux, very lightweight)"""
        subprocess.run(["espeak", text], check=True)

class ImprovedDistanceEstimator:
    """Improved distance estimation"""
    
    def __init__(self):
        # Real-world object heights
        self.REAL_HEIGHTS = {
            "Person": 1.70, "Car": 1.50, "Chair": 0.90, "Bicycle": 1.10,
            "Motorcycle": 1.30, "Bus": 3.00, "Truck": 3.50, 
            "Traffic Light": 3.00, "Stop Sign": 2.10, "Bench": 0.45,
            "Dog": 0.60, "Cat": 0.25, "Bottle": 0.25, "Cup": 0.12,
            "Bowl": 0.08, "Laptop": 0.02, "Cell Phone": 0.15,
            "Book": 0.03, "Clock": 0.30, "Vase": 0.25, "Scissors": 0.20,
            "Teddy Bear": 0.30, "Hair Drier": 0.25, "Toothbrush": 0.18,
            "Backpack": 0.50
        }
        
        # Real-world widths for dual estimation
        self.REAL_WIDTHS = {
            "Person": 0.50, "Car": 1.80, "Bicycle": 0.60, "Bus": 2.50,
            "Truck": 2.50, "Chair": 0.60, "Motorcycle": 0.80,
            "Stop Sign": 0.75, "Bench": 1.50, "Dog": 0.40, "Cat": 0.25,
            "Backpack": 0.35
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
    """Main vision assistant class with speech-aware detection"""
    
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
        self.speech_pause_time = 0.5  # seconds to wait after speech ends
        
        # Initialize camera
        self.setup_camera()
        
        # Threading for TTS with proper synchronization
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
        # Track last detection to avoid repetition
        self.last_detection_time = 0
        self.last_objects = []
        
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
        """Background worker for TTS with proper blocking"""
        while True:
            try:
                text = self.tts_queue.get(timeout=1)
                if text is None:  # Shutdown signal
                    break
                
                print(f"[TTS] Speaking: {text}")
                self.tts.speak(text)  # This blocks until speech is complete
                
                # Small pause after speech to prevent immediate overlap
                time.sleep(self.speech_pause_time)
                
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS worker error: {e}", file=sys.stderr)
    
    def speak_async(self, text):
        """Add text to TTS queue for speech"""
        try:
            # Clear old messages if queue is getting long
            while self.tts_queue.qsize() > 1:
                try:
                    old_text = self.tts_queue.get_nowait()
                    print(f"[TTS] Skipping: {old_text}")
                except queue.Empty:
                    break
            
            self.tts_queue.put(text)
        except Exception as e:
            print(f"Error queuing TTS: {e}", file=sys.stderr)
    
    def is_speaking(self):
        """Check if TTS is currently active"""
        return self.tts.is_speaking.is_set() or not self.tts_queue.empty()
    
    def objects_changed(self, new_objects):
        """Check if detected objects have significantly changed"""
        if len(new_objects) != len(self.last_objects):
            return True
        
        # Sort both lists by distance for comparison
        new_sorted = sorted(new_objects, key=lambda x: (x[1], x[0]))  # Sort by label, then distance
        old_sorted = sorted(self.last_objects, key=lambda x: (x[1], x[0]))
        
        for (new_dist, new_label), (old_dist, old_label) in zip(new_sorted, old_sorted):
            if new_label != old_label:
                return True
            # Consider changed if distance differs by more than 2 meters
            if abs(new_dist - old_dist) > 2.0:
                return True
        
        return False
    
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
            return f"There is a {label1} approximately {obj1[0]:.0f} metres ahead, and a {obj2[1]} at {obj2[0]:.0f} metres."
        
        # For 3+ objects, limit to 3 closest
        nearby_objects = nearby_objects[:3]
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
        # Skip processing if TTS is active
        if self.is_speaking():
            print("[DEBUG] Skipping detection - TTS active")
            return None
        
        # Flush camera buffer to get fresh frame
        for _ in range(2):
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
        if all_objects:
            print(f"[DEBUG] Detected {len(all_objects)} total objects")
            for dist, label in all_objects[:5]:  # Show first 5
                print(f"[DEBUG] {label} at {dist:.1f}m")
            print(f"[DEBUG] {len(nearby_objects)} objects within {self.near_threshold}m")
        
        return nearby_objects
    
    def run(self):
        """Main loop with speech-aware detection"""
        print("[INFO] Starting Vision Assistant...")
        
        # Power-on announcement
        self.speak_async("Power on, Insight is your assistant")
        
        try:
            while True:
                current_time = time.time()
                
                # Only process if enough time has passed and not speaking
                if (current_time - self.last_detection_time >= self.detection_interval and 
                    not self.is_speaking()):
                    
                    nearby_objects = self.process_frame()
                    
                    if nearby_objects is not None:
                        # Only announce if objects have changed significantly
                        if self.objects_changed(nearby_objects):
                            response_text = self.create_response_text(nearby_objects)
                            if response_text:
                                print(f"[INFO] {response_text}")
                                self.speak_async(response_text)
                                self.last_objects = nearby_objects.copy()
                        
                        self.last_detection_time = current_time
                
                # Short sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down Vision Assistant...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        self.tts_queue.put(None)  # Signal TTS worker to stop
        self.tts_thread.join(timeout=3)
        print("[INFO] Vision Assistant stopped")

def main():
    """Main function with configuration options"""
    
    # Configuration - adjust these based on your setup
    TTS_ENGINE = "piper"  # Options: "pyttsx3", "piper", "gtts", "espeak"
    MODEL_PATH = "./Insight/insight_deploy/models/yolo11n_object365.pt"
    
    try:
        assistant = VisionAssistant(tts_engine=TTS_ENGINE, model_path=MODEL_PATH)
        assistant.run()
    except Exception as e:
        print(f"[ERROR] Failed to start Vision Assistant: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()