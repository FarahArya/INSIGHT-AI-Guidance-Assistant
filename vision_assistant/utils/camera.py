import cv2
import logging

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, width: int = 480, height: int = 360, fps: int = 15):
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm up camera
        for _ in range(3):
            self.cap.read()

        logger.info(f"Camera initialized: {width}x{height} @ {fps}fps")

    def capture(self):
        """Capture a frame, flushing the buffer first"""
        # Flush buffer
        for _ in range(2):
            self.cap.read()

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture frame from camera")
            return None
        return frame

    def release(self):
        self.cap.release()
        logger.info("Camera released")