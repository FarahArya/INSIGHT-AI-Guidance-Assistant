# processor.py

from typing import List, Tuple, Any
import logging

from .distance import DistanceEstimator
from .models import SingleModelManager, ModelInfo
from .unknown_detector import UnknownObjectDetector

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self,
                 model_manager: SingleModelManager,
                 distance_estimator: DistanceEstimator):
        self.model_manager = model_manager
        self.distance_estimator = distance_estimator

        # Initialize unknown object detector with conservative settings
        self.unknown_detector = UnknownObjectDetector(
            min_area=800,           # Larger minimum area for Pi performance
            max_area=40000,         # Conservative max area
            min_solidity=0.4,       # Higher solidity requirement
            max_aspect_ratio=4.0,   # Stricter aspect ratio
            detection_frequency=4   # Only check every 4th frame
        )

    def process(self, frame) -> Tuple[List[Tuple[float, str, Any, str]], ModelInfo]:
        """
        Process a single frame and return:
          - objects: List of (distance, label, box_or_None, position_str)
          - model_info: the ModelInfo from the detection call
        """
        # 1. Run the YOLO detection
        results, model_info = self.model_manager.detect(frame, conf_threshold=0.35)
        objects: List[Tuple[float, str, Any, str]] = []
        known_boxes: List[Tuple[int, int, int, int]] = []

        # 2. Process known objects
        for box in results.boxes:
            label = model_info.labels[int(box.cls[0])]
            distance = self.distance_estimator.estimate_distance(
                box, frame.shape[0], frame.shape[1], label
            )
            pos_info = self.distance_estimator.get_detailed_position(
                box, frame.shape[1], frame.shape[0]
            )

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            known_boxes.append((int(x1), int(y1), int(x2), int(y2)))

            objects.append((
                distance,
                label,
                box,
                pos_info.combined
            ))

        # 3. Detect unknown objects if there aren't too many known ones
        if len(objects) < 5:
            try:
                unknown_objects = self.unknown_detector.detect_unknown_objects(frame, known_boxes)
                for unk in unknown_objects:
                    if unk.estimated_distance <= 8.0:
                        objects.append((
                            unk.estimated_distance,
                            "unknown object",
                            None,
                            unk.position
                        ))
            except Exception as e:
                logger.warning(f"Unknown object detection failed: {e}")

        logger.debug(
            f"Detected {len(objects)} objects "
            f"({len([o for o in objects if o[1] == 'unknown object'])} unknown)"
        )

        # 4. Return both objects list and model_info so callers can unpack correctly
        return objects, model_info
