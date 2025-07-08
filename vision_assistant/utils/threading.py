import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)

class TTSWorker(threading.Thread):
    def __init__(self, tts_engine, tts_queue):
        super().__init__(daemon=True)
        self.tts_engine = tts_engine
        self.tts_queue = tts_queue
        self._stop_event = threading.Event()
        self.pause_duration = 0.5  # Seconds to pause between TTS messages

    def run(self):
        while not self._stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=1)
                if text is None:
                    continue

                logger.info(f"Speaking: {text}")
                self.tts_engine.speak(text)
                time.sleep(self.pause_duration)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS Worker Error: {e}")

    def stop(self):
        self._stop_event.set()
        self.tts_engine.stop()