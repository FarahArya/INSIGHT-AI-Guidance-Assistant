# Vision Assistant for the Visually Impaired - Technical Documentation

## Overview
This system is a vision assistance application designed for Raspberry Pi that helps visually impaired users understand their surroundings. It uses dual YOLO models for object detection and specialized TTS for audio feedback. The modular architecture ensures maintainability and extensibility.

![System Architecture](https://example.com/system-arch.png) *(Example architecture diagram)*

## Quick Start Guide
1. **Install dependencies**:
```bash
pip install opencv-python-headless numpy ultralytics
```

2. **Run the application**:
```bash
python main.py
```

3. **Basic commands**:
- Press `Ctrl+C` to exit
- Audio announcements describe nearby objects

## Key Components

### 1. Configuration System (`config.py`)
Centralizes all settings using Python dataclasses:

```python
@dataclass
class VisionConfig:
    main_model_path: str = "./models/yolo11n_object365.pt"
    architectural_model_path: str = "./models/architectural_model.pt"
    conf_threshold: float = 0.45
    # ...20+ other parameters
```

**Key Features**:
- Separate configurations for TTS, vision, and application
- Predefined object dimensions for distance estimation
- Debug mode flag

### 2. Text-to-Speech System (`tts/`)
Modular TTS engine implementations:

```python
class BaseTTS(ABC):
    @abstractmethod
    def speak(self, text: str): ...

class PiperTTS(BaseTTS):
    def speak(self, text: str):
        # Piper-specific implementation
```

**Supported Engines**:
- `PiperTTS`: Local high-quality synthesis
- `Pyttsx3TTS`: Cross-platform solution
- (Add custom engines by implementing `BaseTTS`)

### 3. Vision Processing (`vision/`)
#### a. Dual Model Manager (`models.py`)
```python
class DualModelManager:
    def __init__(self, main_model_path, arch_model_path):
        self.main_model = YOLO(main_model_path)
        self.arch_model = YOLO(arch_model_path)
        # Alternates between models
```

**Key Features**:
- Loads both object detection and architectural models
- Alternates models every N detections
- Optimized for Raspberry Pi performance

#### b. Distance Estimator (`distance.py`)
```python
class DistanceEstimator:
    def estimate_distance(self, box, img_h, img_w, label):
        # Uses real-world dimensions + perspective correction
        # Applies temporal smoothing
```

**Algorithms**:
- Focal length calculation based on FOV
- Height and width-based estimation
- Ground plane correction
- Temporal smoothing (3-sample history)

#### c. Frame Processor (`processor.py`)
```python
class FrameProcessor:
    def process(self, frame):
        # Runs detection
        # Estimates distances
        # Calculates positions
        return objects, model_type
```

### 4. Camera System (`utils/camera.py`)
```python
class Camera:
    def __init__(self, width=480, height=360, fps=15):
        # Raspberry Pi optimized settings
        self.cap = cv2.VideoCapture(0)
```

**Optimizations**:
- Reduced resolution (480x360)
- Lower FPS (15) for Pi4
- Buffer minimization
- Warm-up routine

### 5. Concurrency System (`utils/threading.py`)
```python
class TTSWorker(threading.Thread):
    def run(self):
        while not self._stop_event.is_set():
            # Processes TTS queue
            # Manages speech overlap
```

**Features**:
- Dedicated thread for TTS
- Queue management
- Speech prioritization
- Thread-safe operations

## Core Workflow
1. **Initialization**:
   - Load configuration
   - Initialize camera
   - Load YOLO models
   - Start TTS worker thread

2. **Processing Loop**:
   ```mermaid
   graph TD
   A[Capture Frame] --> B[Select Model]
   B --> C[Run Detection]
   C --> D[Estimate Distances]
   D --> E[Determine Positions]
   E --> F[Filter Nearby Objects]
   F --> G[Generate Announcement]
   G --> H[Queue TTS]
   H --> A
   
   ```

3. **Audio Output**:
   - Prioritizes important objects (doors/walls)
   - Uses combined position descriptions
   - Limits to 3 most relevant objects

## Customization Guide

### 1. Adjusting Detection Parameters
Edit `config.py`:
```python
VisionConfig(
    conf_threshold=0.5,  # Confidence threshold
    near_threshold=5.0,   # Object proximity threshold
    model_switch_interval=3  # Model alternation frequency
)
```

### 2. Adding Object Dimensions
Extend the real_heights dictionary:
```python
AppConfig(
    vision=VisionConfig(
        real_heights={
            "New_Object": 1.2,
            # Add new entries
        },
        real_widths={
            "New_Object": 0.8,
            # Add new entries
        }
    )
)
```

### 3. Implementing New TTS Engine
1. Create `tts/new_engine.py`
2. Implement BaseTTS interface:
```python
class NewEngineTTS(BaseTTS):
    def speak(self, text: str):
        # Implementation here
```
3. Update configuration:
```python
TTSConfig(engine="new_engine")
```

## Performance Considerations
1. **Raspberry Pi Optimization**:
   - Reduced frame resolution
   - Model half-precision (FP16)
   - Detection interval throttling
   - Camera buffer minimization

2. **Memory Management**:
   - Frame-by-frame processing
   - Object detection result pruning
   - TTS audio file cleanup

## Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| **Camera not initializing** | Check Pi camera permissions: `sudo raspi-config` → Interface Options → Camera |
| **Model loading fails** | Verify model paths in `config.py` and file permissions |
| **Low FPS** | Reduce camera resolution in `config.py` or increase detection interval |
| **Audio distortion** | Adjust headphone volume: `amixer -q sset Headphone 90%` |
| **Missing objects** | Lower confidence threshold in `config.py` |

## Directory Structure
```
vision_assistant/
├── config.py             # Central configuration
├── main.py               # Entry point
├── tts/                  # Text-to-speech engines
│   ├── base.py           # Abstract base class
│   ├── piper.py          # Piper TTS implementation
│   └── pyttsx3.py        # pyttsx3 implementation
├── utils/                # Utility modules
│   ├── camera.py         # Camera handling
│   └── threading.py      # Concurrency utilities
└── vision/               # Computer vision components
    ├── detector.py       # Detection utilities
    ├── distance.py       # Distance estimation
    ├── models.py         # Model management
    └── processor.py      # Frame processing
```

## Extension Points
1. **Add New Sensors**:
   - Implement in `utils/sensors.py`
   - Integrate with main processing loop

2. **Implement Object Tracking**:
   - Create `vision/tracker.py`
   - Use ByteTrack or similar algorithm

3. **Add Spatial Audio**:
   - Modify `tts/base.py` to support 3D audio
   - Integrate HRTF processing

4. **Cloud Integration**:
   - Create `cloud/aws.py` or `cloud/azure.py`
   - Add fallback to cloud-based vision

## License
Apache 2.0 License - See LICENSE file for details

> **Note**: This documentation covers the core architecture and key components. For implementation details, refer to source code comments in each module. The system is designed so developers can work on individual components without needing deep knowledge of the entire codebase.
