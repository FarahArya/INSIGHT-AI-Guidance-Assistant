# Insight Inference Component (YOLOv11n + Piper TTS)

This README describes the **inference component** of the Insight project. This module runs object detection and distance estimation using YOLOv11n on a Raspberry Pi and emits structured output suitable for text-to-speech via Piper.

---

## Component Purpose

This component is designed to:

* Detect objects in the environment using a camera feed
* Estimate the distance to each object using bounding box dimensions
* Emit a JSON line per frame with a natural language sentence describing the nearest object within range
* Interface seamlessly with Piper TTS for audio feedback

---

## Directory Structure (Component Only)

```
insight_deploy/
├── models/
│   ├── yolo11n_ncnn_model/          # NCNN model files for Raspberry Pi
│   │   ├── yolo11n_ncnn_model.param
│   │   └── yolo11n_ncnn_model.bin
│   └── yolo11n.onnx                 # Optional ONNX fallback model
├── insight_infer.py                 # Main Python script (entry point)
├── labels.txt                       # COCO class labels
└── README.md                        # This file
```

---

## Requirements

* Python 3.8+
* Raspberry Pi 4 or later (or any Linux machine for testing)

### Dependencies

The following Python packages are required:

* `ultralytics` (provides YOLO and NCNN support)
* `piper-tts` (text-to-speech engine)
* `opencv-python` (camera interface)

Install them using:

```bash
sudo apt update
sudo apt install python3-opencv -y
pip install ultralytics piper-tts
```

Note: These are minimal runtime dependencies. Avoid exporting the entire virtual environment to prevent bundling unnecessary packages.

---

## How to Run This on a Raspberry Pi (Beginner Friendly)

### 1. Prepare the Raspberry Pi

* Use a Raspberry Pi 4 or later.
* Install the latest Raspberry Pi OS (64-bit recommended).
* Make sure your Pi is connected to the internet and updated:

  ```bash
  sudo apt update && sudo apt full-upgrade -y
  sudo reboot
  ```

### 2. Transfer the Project Folder

* From your development machine, transfer the `insight_deploy/` directory to the Pi using SCP or a USB drive:

  ```bash
  scp -r insight_deploy pi@raspberrypi.local:~/
  ```

### 3. Install the Required Packages

On the Pi, open a terminal and run:

```bash
sudo apt update
sudo apt install python3-opencv -y
pip install ultralytics piper-tts
```

### 4. Download a Piper Voice (if not already present)

For example:

```bash
mkdir -p ~/voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/onnx/en_GB-alba-low.onnx -O ~/voices/en_GB-alba-low.onnx
```

### 5. Run the System

From within the `insight_deploy/` directory:

```bash
cd ~/insight_deploy
python3 insight_infer.py | \
  piper --model ~/voices/en_GB-alba-low.onnx --json-input
```

You should see printed messages like:

```json
{"text": "There is a person approximately 3 metres ahead."}
```

And you should hear Piper speaking these lines.

---

## Running Automatically on Boot Using systemd

To run the system automatically when the Raspberry Pi boots, you can set up a `systemd` service:

### 1. Create a systemd service file

Create a new service file:

```bash
sudo nano /etc/systemd/system/insight.service
```

Paste the following:

```ini
[Unit]
Description=Insight Object Detection and TTS
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/insight_deploy/insight_infer.py | /usr/local/bin/piper --model /home/pi/voices/en_GB-alba-low.onnx --json-input
WorkingDirectory=/home/pi/insight_deploy
StandardOutput=journal
StandardError=journal
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

### 2. Enable and start the service

```bash
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable insight.service
sudo systemctl start insight.service
```

### 3. Check the service status

```bash
sudo systemctl status insight.service
```

To see logs:

```bash
journalctl -u insight.service -e
```

---

## Configuration Parameters

Edit `insight_infer.py` to customize:

```python
CONF_THRES = 0.45              # Confidence threshold
NEAR_THRESH_METRES = 5         # Maximum distance for spoken feedback
FOCAL_PX = 600                 # Approximate focal length (pixel units)
```

---

## Integration

This component is intended to run independently and stream structured JSON lines. These lines can be consumed by:

* Piper for real-time speech
* A log file for debugging
* A TCP or HTTP interface if extended

---

## Status

* NCNN export tested and functional
* Distance estimation heuristic implemented
* Piper integration verified
* systemd unit recommended for automatic startup

---

## License

MIT License

For integration or contribution to the full Insight project, refer to the global project README.
