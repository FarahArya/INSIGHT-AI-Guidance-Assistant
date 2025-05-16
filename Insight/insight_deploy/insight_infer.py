#!/usr/bin/env python3

import cv2, json, time, numpy as np, os
from ultralytics import YOLO
import sys

"""
Insight – YOLOv11n (NCNN) → Piper TTS
Run on Raspberry Pi 4:

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




# ─────────────────────────  CONFIG  ──────────────────────────
MODEL_DIR = "/home/rpi-farah/INSIGHT-AI-Guidance-Assistant/Insight/insight_deploy/yolo11n.pt"
LABELS = YOLO("yolo11n.pt").names
FOCAL_PX = 600
CONF_THRES = 0.45
NEAR_THRESH_METRES = 5
TRIGGER_FILE = "trigger.txt"  # Trigger file to wait for
# ─────────────────────────────────────────────────────────────

# Load YOLO model
model = YOLO(MODEL_DIR, task="detect")

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def estimate_distance(box, img_h):
    x1, y1, x2, y2 = box.xyxy[0]
    h_px = y2 - y1
    label = LABELS[int(box.cls[0])]
    real_h = REAL_HEIGHTS.get(label, None)
    if real_h:
        return (real_h * FOCAL_PX) / h_px
    return (img_h / h_px) * 0.5

while True:
    # Wait for trigger
    while not os.path.exists(TRIGGER_FILE):
        time.sleep(0.05)

    ok, frame = cap.read()
    if not ok:
        continue

    res = model(frame, imgsz=640, conf=CONF_THRES)[0]
    h = frame.shape[0]

    items = []
    for b in res.boxes:
        d = estimate_distance(b, h)
        items.append((d, b))

    if not items:
        os.remove(TRIGGER_FILE)
        continue

    dist, box = min(items, key=lambda t: t[0])
    if dist > NEAR_THRESH_METRES:
        os.remove(TRIGGER_FILE)
        continue

    label = LABELS[int(box.cls[0])]
    sentence = f"There is a {label} approximately {dist:.0f} metres ahead."
    # print(json.dumps({"text": sentence}, ensure_ascii=False), flush=True)

    if dist <= NEAR_THRESH_METRES:
        label = LABELS[int(box.cls[0])]
        sentence = f"There is a {label} approximately {dist:.0f} metres ahead."
        print(json.dumps({"text": sentence}, ensure_ascii=False), flush=True)
        print(sentence, file=sys.stderr, flush=True)  # Human-readable log to stderr


    os.remove(TRIGGER_FILE)

