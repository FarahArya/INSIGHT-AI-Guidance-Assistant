#!/usr/bin/env python3
import cv2, json, time, numpy as np, os
from ultralytics import YOLO
import sys
"""
Insight – YOLOv11n (NCNN) → Piper TTS
Run on Raspberry Pi 4:
 python3 insight_infer.py | \
 piper --model /home/pi/voices/en_GB-alba-low.onnx --json-input
"""
REAL_HEIGHTS = {
    "person": 1.7,
    "bicycle": 1.1,
    "car": 1.5,
    "motorcycle": 1.3,
    "airplane": 3.5,
    "bus": 3.0,
    "train": 3.5,
    "truck": 3.0,
    "boat": 2.0,
    "traffic light": 2.5,
    "fire hydrant": 0.9,
    "stop sign": 0.75,
    "parking meter": 1.4,
    "bench": 0.5,
    "bird": 0.25,
    "cat": 0.3,
    "dog": 0.5,
    "horse": 1.6,
    "sheep": 0.9,
    "cow": 1.4,
    "elephant": 3.0,
    "bear": 2.0,
    "zebra": 1.4,
    "giraffe": 4.5,
    "backpack": 0.6,
    "umbrella": 0.9,
    "handbag": 0.4,
    "tie": 0.6,
    "suitcase": 0.7,
    "frisbee": 0.25,
    "skis": 1.6,
    "snowboard": 1.5,
    "sports ball": 0.25,
    "kite": 1.0,
    "baseball bat": 0.9,
    "baseball glove": 0.25,
    "skateboard": 0.9,
    "surfboard": 1.7,
    "tennis racket": 0.7,
    "bottle": 0.3,
    "wine glass": 0.25,
    "cup": 0.15,
    "fork": 0.2,
    "knife": 0.2,
    "spoon": 0.2,
    "bowl": 0.2,
    "banana": 0.2,
    "apple": 0.15,
    "sandwich": 0.1,
    "orange": 0.15,
    "broccoli": 0.25,
    "carrot": 0.25,
    "hot dog": 0.15,
    "pizza": 0.3,
    "donut": 0.15,
    "cake": 0.3,
    "chair": 1.0,
    "couch": 1.2,
    "potted plant": 0.7,
    "bed": 0.9,
    "dining table": 0.75,
    "toilet": 0.8,
    "tv": 0.6,
    "laptop": 0.4,
    "mouse": 0.05,
    "remote": 0.2,
    "keyboard": 0.1,
    "cell phone": 0.15,
    "microwave": 0.5,
    "oven": 0.9,
    "toaster": 0.3,
    "sink": 0.8,
    "refrigerator": 1.8,
    "book": 0.25,
    "clock": 0.35,
    "vase": 0.4,
    "scissors": 0.2,
    "teddy bear": 0.4,
    "hair drier": 0.25,
    "toothbrush": 0.2
}
# ───────────────────────── CONFIG ──────────────────────────
MODEL_DIR = "/home/rpi-farah/INSIGHT-AI-Guidance-Assistant/Insight/insight_deploy/yolo11n.pt"
LABELS = YOLO("yolo11n.pt").names
FOCAL_PX = 600
CONF_THRES = 0.45
NEAR_THRESH_METRES = 6  # Increased from 5 to 6 meters
TRIGGER_FILE = "/home/rpi-farah/INSIGHT-AI-Guidance-Assistant/trigger.txt"  # Trigger file to wait for
FEEDBACK_FILE = "/home/rpi-farah/INSIGHT-AI-Guidance-Assistant/feedback.json"
# ─────────────────────────────────────────────────────────────
# Load YOLO model
model = YOLO(MODEL_DIR, task="detect")
# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def estimate_distance(box, img_h):
    x1, y1, x2, y2 = box.xyxy[0]
    h_px = float(y2 - y1)  # Convert to float
    label = LABELS[int(box.cls[0])]
    real_h = REAL_HEIGHTS.get(label, None)
    if real_h:
        return (real_h * FOCAL_PX) / h_px
    return (img_h / h_px) * 0.5

def create_response_text(nearby_objects):
    """Create natural language response for all nearby objects"""
    if len(nearby_objects) == 1:
        dist, label = nearby_objects[0]
        return f"There is a {label} approximately {dist:.0f} metres ahead."
    
    # Sort by distance for better readability
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

while True:
    # Wait for trigger
    while not os.path.exists(TRIGGER_FILE):
        time.sleep(0.05)
    
    # Flush camera buffer to get fresh frame
    for _ in range(3):  # Clear 3 stale frames
        cap.read()
    
    ok, frame = cap.read()  # Get fresh frame
    if not ok:
        continue
    
    res = model(frame, imgsz=640, conf=CONF_THRES)[0]
    h = frame.shape[0]
    
    # Get all objects with their distances
    nearby_objects = []
    all_objects = []  # For debugging
    
    for b in res.boxes:
        d = estimate_distance(b, h)
        label = LABELS[int(b.cls[0])]
        all_objects.append((d, label))
        
        if d <= NEAR_THRESH_METRES:  # Keep this as-is (already inclusive)
            nearby_objects.append((d, label))
    
    # Debug output to stderr
    print(f"DEBUG: Detected {len(all_objects)} total objects", file=sys.stderr, flush=True)
    for dist, label in all_objects:
        print(f"DEBUG: {label} at {dist:.1f}m", file=sys.stderr, flush=True)
    print(f"DEBUG: {len(nearby_objects)} objects within {NEAR_THRESH_METRES}m threshold", file=sys.stderr, flush=True)
    
    if not nearby_objects:
        print("DEBUG: No nearby objects, removing trigger file", file=sys.stderr, flush=True)
        os.remove(TRIGGER_FILE)
        continue
    
    # Create response for all nearby objects
    sentence = create_response_text(nearby_objects)
    
    # Send to stdout for piping to TTS
    print(json.dumps({"text": sentence}, ensure_ascii=False), flush=True)
    # Human-readable log to stderr
    print(sentence, file=sys.stderr, flush=True)
    
    # Write detailed feedback to file (atomic write using temp file)
    try:
        response = {
            "text": sentence,
            "objects_detected": len(nearby_objects),
            "details": [{"object": label, "distance_metres": round(float(dist), 1)} for dist, label in nearby_objects]
        }
        # Write to temp file first, then atomic rename
        temp_file = FEEDBACK_FILE + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(response, f)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk
        
        # Atomic rename - this prevents C++ from reading partial files
        os.rename(temp_file, FEEDBACK_FILE)
        
    except Exception as e:
        print(f"Error writing to feedback file: {e}", file=sys.stderr, flush=True)
    
    # Remove trigger file to indicate completion
    if os.path.exists(TRIGGER_FILE):
        os.remove(TRIGGER_FILE)