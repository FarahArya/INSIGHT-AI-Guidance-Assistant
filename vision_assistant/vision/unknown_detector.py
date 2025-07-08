import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class UnknownObject:
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    estimated_distance: float
    position: str

class UnknownObjectDetector:
    def __init__(self,
                 min_area: int = 500,          # Minimum contour area (conservative)
                 max_area: int = 50000,        # Maximum contour area
                 min_solidity: float = 0.3,    # Shape solidity threshold
                 max_aspect_ratio: float = 5.0, # Width/height ratio limit
                 detection_frequency: int = 3   # Detect every N frames
                 ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_solidity = min_solidity
        self.max_aspect_ratio = max_aspect_ratio
        self.detection_frequency = detection_frequency
        self.frame_count = 0

        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )

        # Morphological operations kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect_unknown_objects(self,
                               frame: np.ndarray,
                               known_boxes: List[Tuple[int, int, int, int]]) -> List[UnknownObject]:
        """
        Detect unknown objects not covered by known object detections

        Args:
            frame: Input frame
            known_boxes: List of (x1, y1, x2, y2) bounding boxes of known objects

        Returns:
            List of UnknownObject instances
        """
        self.frame_count += 1

        # Only process every N frames for performance
        if self.frame_count % self.detection_frequency != 0:
            return []

        h, w = frame.shape[:2]

        # Create mask to exclude known object areas
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Mask out known objects with some padding
        padding = 20
        for x1, y1, x2, y2 in known_boxes:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            mask[y1:y2, x1:x2] = 0

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Combine with exclusion mask
        combined_mask = cv2.bitwise_and(fg_mask, mask)

        # Morphological operations to clean up
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        unknown_objects = []

        for contour in contours:
            # Apply conservative filters
            if self._is_valid_unknown_object(contour):
                unknown_obj = self._create_unknown_object(contour, frame.shape)
                if unknown_obj:
                    unknown_objects.append(unknown_obj)

        return unknown_objects

    def _is_valid_unknown_object(self, contour: np.ndarray) -> bool:
        """Apply conservative filters to determine if contour represents an unknown object"""

        # Area filter
        area = cv2.contourArea(contour)
        if area < self.min_area or area > self.max_area:
            return False

        # Solidity filter (filled area vs convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < self.min_solidity:
                return False

        # Aspect ratio filter
        x, y, w, h = cv2.boundingRect(contour)
        if h > 0:
            aspect_ratio = w / h
            if aspect_ratio > self.max_aspect_ratio or aspect_ratio < (1.0 / self.max_aspect_ratio):
                return False

        # Minimum bounding box size
        if w < 30 or h < 30:  # Too small to be significant
            return False

        return True

    def _create_unknown_object(self, contour: np.ndarray, frame_shape: Tuple[int, int, int]) -> Optional[UnknownObject]:
        """Create UnknownObject from validated contour"""
        h, w = frame_shape[:2]

        # Get bounding rectangle
        x, y, rect_w, rect_h = cv2.boundingRect(contour)

        # Calculate center
        center_x = x + rect_w // 2
        center_y = y + rect_h // 2

        # Estimate distance based on size (very rough)
        area = cv2.contourArea(contour)
        # Assume objects further away appear smaller
        estimated_distance = max(1.0, min(15.0, 50000 / area))

        # Get position description
        position = self._get_position_description(center_x, center_y, w, h)

        return UnknownObject(
            contour=contour,
            center=(center_x, center_y),
            area=area,
            bounding_box=(x, y, rect_w, rect_h),
            estimated_distance=estimated_distance,
            position=position
        )

    def _get_position_description(self, center_x: int, center_y: int, img_w: int, img_h: int) -> str:
        """Get position description similar to main detection system"""
        norm_x = center_x / img_w
        norm_y = center_y / img_h

        # Horizontal position
        if norm_x < 0.33:
            horizontal = "left"
        elif norm_x > 0.67:
            horizontal = "right"
        else:
            horizontal = "center"

        # Vertical position
        if norm_y < 0.33:
            vertical = "top"
        elif norm_y > 0.67:
            vertical = "bottom"
        else:
            vertical = "middle"

        # Combined description
        if horizontal == "center":
            return "forward" if vertical == "middle" else f"{vertical} forward"
        else:
            return horizontal if vertical == "middle" else f"{vertical} {horizontal}"

    def reset_background(self):
        """Reset background model - call when environment changes significantly"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )