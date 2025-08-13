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

class SingleModelManager:
    def __init__(self, model_path: str):
        logger.info("Loading unified model...")
        self.model_info = self._load_model(model_path, "unified")
        logger.info(f"Model loaded: {self.model_info.model_type} with {len(self.model_info.labels)} classes")

        # Log the available classes
        logger.info(f"Available classes: {list(self.model_info.labels.values())}")

    def _load_model(self, path: str, model_type: str) -> ModelInfo:
        model = YOLO(path, task="detect")
        return ModelInfo(
            model=model,
            labels=model.names,
            model_type=model_type
        )

    def get_current_model(self) -> ModelInfo:
        return self.model_info

    def detect(self, frame, conf_threshold: float = 0.45) -> Tuple[list, ModelInfo]:
        """Run detection with the unified model"""
        results = self.model_info.model(
            frame,
            imgsz=480,
            conf=conf_threshold,
            half=True,
            device='cpu',
            verbose=False
        )[0]
        return results, self.model_info