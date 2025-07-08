from dataclasses import dataclass
from typing import Dict, Tuple
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    model: YOLO
    labels: Dict[int, str]
    model_type: str

class DualModelManager:
    def __init__(self, main_model_path: str, arch_model_path: str, switch_interval: int = 5):
        logger.info("Loading dual models...")
        self.main_model = self._load_model(main_model_path, "objects")
        self.arch_model = self._load_model(arch_model_path, "architecture")
        self.current_model = self.main_model
        self.switch_interval = switch_interval
        self.detection_count = 0
        logger.info(f"Models loaded: {self.main_model.model_type} and {self.arch_model.model_type}")

    def _load_model(self, path: str, model_type: str) -> ModelInfo:
        model = YOLO(path, task="detect")
        return ModelInfo(
            model=model,
            labels=model.names,
            model_type=model_type
        )

    def get_current_model(self) -> ModelInfo:
        return self.current_model

    def should_switch(self) -> bool:
        self.detection_count += 1
        if self.detection_count >= self.switch_interval:
            self.detection_count = 0
            self.current_model = self.arch_model if self.current_model == self.main_model else self.main_model
            logger.info(f"Switched to {self.current_model.model_type} model")
            return True
        return False

    def detect(self, frame, conf_threshold: float = 0.45) -> Tuple[list, ModelInfo]:
        """Run detection with current model"""
        results = self.current_model.model(
            frame,
            imgsz=480,
            conf=conf_threshold,
            half=True,
            device='cpu',
            verbose=False
        )[0]
        return results, self.current_model