import time
import queue
import threading
import logging
import os
from typing import List, Tuple
from config import AppConfig
from tts.piper import PiperTTS
from vision.models import SingleModelManager  # Updated import
from vision.distance import DistanceEstimator
from vision.processor import FrameProcessor
from utils.camera import Camera
from utils.threading import TTSWorker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionAssistant:
    def __init__(self, config: AppConfig):
        self.config = config
        self.setup_components()
        self.last_detection_time = 0
        self.last_objects = []

    def setup_components(self):
        """Initialize all system components"""
        # TTS System
        self.tts = PiperTTS(self.config.tts)
        self.tts_queue = queue.Queue()
        self.tts_worker = TTSWorker(self.tts, self.tts_queue)
        self.tts_worker.start()

        # Vision System
        distance_estimator = DistanceEstimator(
            self.config.vision.real_heights,
            self.config.vision.real_widths
        )

        # Use single model instead of dual model
        self.model_manager = SingleModelManager(
            self.config.vision.main_model_path
        )

        self.processor = FrameProcessor(self.model_manager, distance_estimator)
        self.camera = Camera(
            width=self.config.vision.camera_width,
            height=self.config.vision.camera_height,
            fps=self.config.vision.camera_fps
        )

    def run(self):
        """Main execution loop"""
        logger.info("Starting Vision Assistant")
        self.speak_async("System initialized and ready")

        try:
            while True:
                current_time = time.time()

                if (current_time - self.last_detection_time >= self.config.vision.detection_interval and
                        not self.is_speaking()):

                    frame = self.camera.capture()
                    if frame is None:
                        continue

                    objects, model_info = self.processor.process(frame)

                    # Use single threshold for all objects
                    threshold = self.config.vision.near_threshold
                    nearby_objects = [obj for obj in objects if obj[0] <= threshold]

                    if self._should_announce(nearby_objects):
                        announcement = self._create_announcement(nearby_objects)
                        self.speak_async(announcement)

                    self.last_detection_time = current_time
                    self.last_objects = nearby_objects

                time.sleep(0.05)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.cleanup()

    def _should_announce(self, new_objects: list) -> bool:
        """Determine if we should announce the new objects"""
        if not new_objects:
            # Only announce if we had objects before and now they are gone
            if self.last_objects:
                return True
            return False

        if not self.last_objects:
            return True

        # Check if objects have changed significantly
        new_labels = {obj[1] for obj in new_objects}
        old_labels = {obj[1] for obj in self.last_objects}

        # If different objects are detected
        if new_labels != old_labels:
            return True

        # Check if any object has moved significantly
        for new_obj in new_objects:
            for old_obj in self.last_objects:
                if new_obj[1] == old_obj[1]:
                    if abs(new_obj[0] - old_obj[0]) > 1.0:  # 1 meter change
                        return True

        return False

    def _create_announcement(self, objects: List[Tuple[float, str, any, str]]) -> str:
        """Create natural language announcement with detailed positions including unknown objects"""
        if not objects:
            return "No objects detected nearby."

        # Sort by distance and importance (closest first)
        objects.sort(key=lambda x: x[0])

        # Separate known and unknown objects
        known_objects = [obj for obj in objects if obj[1] != "unknown object"]
        unknown_objects = [obj for obj in objects if obj[1] == "unknown object"]

        # Prioritize doors and walls that are close
        priority_objects = [
            obj for obj in known_objects
            if obj[1] in ["Door", "Wall"] and obj[0] < 3.0  # Updated to match new class names
        ]

        # Build announcement prioritizing known objects
        announcement_objects = []

        if priority_objects:
            # Use priority objects first
            announcement_objects.extend(priority_objects[:2])
            remaining_slots = 3 - len(announcement_objects)
        else:
            # Use closest known objects
            announcement_objects.extend(known_objects[:2])
            remaining_slots = 3 - len(announcement_objects)

        # Add unknown objects if we have remaining slots and they're close
        if remaining_slots > 0 and unknown_objects:
            close_unknown = [obj for obj in unknown_objects if obj[0] < 5.0]
            announcement_objects.extend(close_unknown[:remaining_slots])

        if len(announcement_objects) == 1:
            dist, label, _, position = announcement_objects[0]
            if label == "unknown object":
                return f"There is an unknown object approximately {dist:.1f} meters {position}."
            else:
                # Convert to lowercase for natural speech
                article = "an" if label.lower()[0] in 'aeiou' else "a"
                return f"There is {article} {label.lower()} approximately {dist:.1f} meters {position}."

        # Create announcement with combined position information
        parts = []
        for i, (dist, label, _, position) in enumerate(announcement_objects):
            if label == "unknown object":
                article = "an unknown object"
            else:
                article = f"{'an' if label.lower()[0] in 'aeiou' else 'a'} {label.lower()}"

            if i == 0:
                parts.append(f"There is {article} at {dist:.1f} meters {position}")
            elif i == len(announcement_objects) - 1:
                parts.append(f"and {article} at {dist:.1f} meters {position}")
            else:
                parts.append(f"{article} at {dist:.1f} meters {position}")

        return ", ".join(parts) + "."

    def speak_async(self, text: str):
        """Add text to TTS queue"""
        if text and not self.is_speaking():
            self.tts_queue.put(text)

    def is_speaking(self) -> bool:
        """Check if TTS is active"""
        return self.tts.is_busy() or not self.tts_queue.empty()

    def cleanup(self):
        """Clean up resources"""
        self.camera.release()
        self.tts_worker.stop()
        self.tts_worker.join()


def main():
    config = AppConfig()

    # Check if model exists (only need to check one model now)
    if not os.path.exists(config.vision.main_model_path):
        logger.error(f"Model file not found: {config.vision.main_model_path}")
        return

    try:
        assistant = VisionAssistant(config)
        assistant.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
