# vision/processor.py - Modified version
from typing import List, Tuple, Optional
from distance import DistanceEstimator, PositionInfo
from models import DualModelManager
from unknown_detector import UnknownObjectDetector
import logging

logger = logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self, model_manager: DualModelManager, distance_estimator: DistanceEstimator):
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

    def process(self, frame) -> Tuple[List[Tuple], Optional[str]]:
        """Process frame and return detected objects with unknown object detection"""
        results, model_info = self.model_manager.detect(frame)
        objects = []
        known_boxes = []  # Track known object locations

        # Process known objects first
        for box in results.boxes:
            label = model_info.labels[int(box.cls[0])]
            distance = self.distance_estimator.estimate_distance(
                box, frame.shape[0], frame.shape[1], label
            )
            pos_info = self.distance_estimator.get_detailed_position(
                box, frame.shape[1], frame.shape[0]
            )

            # Store known object info
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            known_boxes.append((int(x1), int(y1), int(x2), int(y2)))

            objects.append((
                distance,
                label,
                box,
                pos_info.combined
            ))

        # Detect unknown objects (only if not too many known objects to keep performance good)
        if len(objects) < 5:  # Conservative limit
            try:
                unknown_objects = self.unknown_detector.detect_unknown_objects(frame, known_boxes)

                # Add unknown objects to results
                for unknown_obj in unknown_objects:
                    # Only add if reasonably close (within 8 meters)
                    if unknown_obj.estimated_distance <= 8.0:
                        objects.append((
                            unknown_obj.estimated_distance,
                            "unknown object",  # Generic label
                            None,  # No YOLO box
                            unknown_obj.position
                        ))

            except Exception as e:
                logger.warning(f"Unknown object detection failed: {e}")

        # Check if we should switch models for next detection
        model_switched = self.model_manager.should_switch()
        model_type = model_info.model_type if model_switched else None

        logger.debug(f"Detected {len(objects)} objects ({len([o for o in objects if o[1] == 'unknown object'])} unknown) with {model_info.model_type} model")
        return objects, model_type