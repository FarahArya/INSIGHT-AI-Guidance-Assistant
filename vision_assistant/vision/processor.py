from typing import List, Tuple, Optional
from .distance import DistanceEstimator, PositionInfo
from .models import DualModelManager
import logging

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self, model_manager: DualModelManager, distance_estimator: DistanceEstimator):
        self.model_manager = model_manager
        self.distance_estimator = distance_estimator

    def process(self, frame) -> Tuple[List[Tuple], Optional[str]]:
        """Process frame and return detected objects with detailed positions"""
        results, model_info = self.model_manager.detect(frame)
        objects = []

        for box in results.boxes:
            label = model_info.labels[int(box.cls[0])]
            distance = self.distance_estimator.estimate_distance(
                box, frame.shape[0], frame.shape[1], label
            )
            pos_info = self.distance_estimator.get_detailed_position(
                box, frame.shape[1], frame.shape[0]
            )
            # Store detailed position information
            objects.append((
                distance,
                label,
                box,
                pos_info.combined  # Use combined position string
            ))

        # Check if we should switch models for next detection
        model_switched = self.model_manager.should_switch()
        model_type = model_info.model_type if model_switched else None

        logger.debug(f"Detected {len(objects)} objects with {model_info.model_type} model")
        return objects, model_type