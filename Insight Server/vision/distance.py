import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class PositionInfo:
    horizontal: str
    vertical: str
    combined: str
    normalized_x: float
    normalized_y: float

class DistanceEstimator:
    def __init__(self, real_heights: Dict[str, float], real_widths: Dict[str, float]):
        self.real_heights = real_heights
        self.real_widths = real_widths
        self.distance_history: Dict[str, List[float]] = {}
        self.max_history = 3

    def estimate_distance(self, box, img_h: int, img_w: int, label: str) -> float:
        """Enhanced distance estimation with better focal length and ground plane correction"""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        h_px = float(y2 - y1)
        w_px = float(x2 - x1)

        focal_px = img_w / (2 * np.tan(np.radians(35)))  # ~70 deg / 2
        real_h = self.real_heights.get(label)

        if not real_h:
            return self._estimate_by_size(h_px, w_px, img_h, img_w)

        distance_h = (real_h * focal_px) / h_px
        real_w = self.real_widths.get(label)

        if real_w:
            distance_w = (real_w * focal_px) / w_px
            distance = (distance_h * 0.7 + distance_w * 0.3)
        else:
            distance = distance_h

        distance = self._apply_corrections(distance, label, y2, img_h, h_px, w_px)

        # Apply smoothing to known objects
        if real_h:
            distance = max(0.5, min(distance, 100.0))
            return self._smooth_distance(label, distance)
        else:
            # For unknown objects, return raw distance estimate
            return max(0.5, min(distance, 100.0))

    def _smooth_distance(self, label: str, distance: float) -> float:
        """Simple temporal smoothing to reduce jitter"""
        if label not in self.distance_history:
            self.distance_history[label] = []

        self.distance_history[label].append(distance)
        if len(self.distance_history[label]) > self.max_history:
            self.distance_history[label].pop(0)

        return sum(self.distance_history[label]) / len(self.distance_history[label])

    def _estimate_by_size(self, h_px: float, w_px: float, img_h: int, img_w: int) -> float:
        """Fallback estimation for unknown objects"""
        apparent_size = (h_px * w_px) / (img_h * img_w)
        if apparent_size > 0.2: return 2.0
        if apparent_size > 0.1: return 5.0
        if apparent_size > 0.05: return 10.0
        return 20.0

    def _apply_corrections(self, distance: float, label: str, box_bottom_y: float, img_h: int, h_px: float, w_px: float) -> float:
        """Apply ground plane and special object corrections"""
        # Ground plane correction
        if label in ["Person", "Car", "Bicycle", "Chair", "Dog", "Cat"]:
            img_center_y = img_h / 2
            if box_bottom_y > img_center_y:
                ground_factor = 1.0 + 0.3 * (box_bottom_y - img_center_y) / img_center_y
                distance *= ground_factor

        # Special handling for walls
        if label == "wall":
            wall_area = (h_px * w_px) / (img_h * img_w)
            if wall_area > 0.5:
                distance = min(distance, 3.0)

        return distance

    def get_detailed_position(self, box, img_width: int, img_height: int) -> PositionInfo:
        """Get detailed position information including vertical position"""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        norm_x, norm_y = center_x / img_width, center_y / img_height

        # Horizontal position
        if norm_x < 0.33: horizontal = "left"
        elif norm_x > 0.67: horizontal = "right"
        else: horizontal = "center"

        # Vertical position
        if norm_y < 0.33: vertical = "top"
        elif norm_y > 0.67: vertical = "bottom"
        else: vertical = "middle"

        # Combined description
        if horizontal == "center":
            combined = "forward" if vertical == "middle" else f"{vertical} forward"
        else:
            combined = horizontal if vertical == "middle" else f"{vertical} {horizontal}"

        return PositionInfo(
            horizontal=horizontal,
            vertical=vertical,
            combined=combined,
            normalized_x=float(norm_x),
            normalized_y=float(norm_y)
        )