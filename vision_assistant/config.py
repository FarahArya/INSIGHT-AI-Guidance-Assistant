import os
from dataclasses import dataclass, field


@dataclass
class TTSConfig:
    engine: str = "piper"
    piper_path: str = "../piper/piper"
    piper_model: str = "../piper/voices/en_US-amy-medium/en_US-amy-medium.onnx"
    piper_config: str = "../piper/voices/en_US-amy-medium/en_US-amy-medium.onnx.json"
    speech_rate: int = 150
    volume: float = 0.9


@dataclass
class VisionConfig:
    main_model_path: str = "../Insight/insight_deploy/models/best_custom_trained_model.yolo11n.pt"
    conf_threshold: float = 0.45
    near_threshold: float = 6.0  # meters
    arch_near_threshold: float = 4.0
    detection_interval: float = 1.5
    model_switch_interval: int = 5
    camera_width: int = 480
    camera_height: int = 360
    camera_fps: int = 15
    real_heights: dict = field(default_factory=lambda: {
        "Person": 1.70,
        "Car": 1.50,
        "Door": 2.00,
        "Sofa": 0.80,  # Same as "Couch" in example
        "Chair": 0.90,
        "Table": 0.75,  # Typical dining table height
        "Lamp": 1.20,
        "TV": 0.50,  # From "Monitor/TV"
        "Laptop": 0.03,
        "Wardrobe": 2.00,  # Typical wardrobe height
        "Window": 1.50,
        "Potted Plant": 0.60,
        "Photo Frame": 0.40,  # From "Picture/Frame"
        "Bed": 0.60,
        "Wall": 3.00,
        "Stairs": 3.00  # Floo
    })
    real_widths: dict = field(default_factory=lambda: {
        "Person": 0.50,
        "Car": 1.80,
        "Door": 0.90,
        "Sofa": 2.00,  # Same as "Couch" in example
        "Chair": 0.45,
        "Table": 0.80,  # Typical table width
        "Lamp": 0.30,  # Default width for lamps
        "TV": 1.20,
        "Laptop": 0.35,
        "Wardrobe": 0.60,  # Typical wardrobe depth
        "Window": 1.20,
        "Potted Plant": 0.35,  # Typical planter width
        "Photo Frame": 0.30,  # Standard frame width
        "Bed": 1.60,
        "Wall": 0.30,  # Wall thickness
        "Stairs": 1.20,
        # Default fallback
        "default": 0.30
    })


def create_vision_config():
    """Factory function to create VisionConfig with custom real_heights and real_widths"""
    return VisionConfig(
        real_heights={
            "Person": 1.70,
            "Car": 1.50,
            "Door": 2.00,
            "Sofa": 0.80,
            "Chair": 0.90,
            "Table": 0.75,
            "Lamp": 1.20,
            "TV": 0.50,
            "Laptop": 0.03,
            "Wardrobe": 2.00,
            "Window": 1.50,
            "Potted Plant": 0.60,
            "Photo Frame": 0.40,
            "Bed": 0.60,
            "Wall": 3.00,
            "Stairs": 3.00
        },
        real_widths={
            "Person": 0.50,
            "Car": 1.80,
            "Door": 0.90,
            "Sofa": 2.00,
            "Chair": 0.45,
            "Table": 0.80,
            "Lamp": 0.30,
            "TV": 1.20,
            "Laptop": 0.35,
            "Wardrobe": 0.60,
            "Window": 1.20,
            "Potted Plant": 0.35,
            "Photo Frame": 0.30,
            "Bed": 1.60,
            "Wall": 0.30,
            "Stairs": 1.20,
            "default": 0.30
        }
    )


@dataclass
class AppConfig:
    tts: TTSConfig = field(default_factory=TTSConfig)
    vision: VisionConfig = field(default_factory=create_vision_config)
    debug: bool = True
