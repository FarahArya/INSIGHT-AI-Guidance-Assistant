from abc import ABC, abstractmethod
import threading

class BaseTTS(ABC):
    def __init__(self):
        self.is_speaking = threading.Event()

    @abstractmethod
    def speak(self, text: str):
        """Speak the given text"""
        pass

    @abstractmethod
    def stop(self):
        """Stop any ongoing speech"""
        pass

    def is_busy(self) -> bool:
        """Check if TTS is currently speaking"""
        return self.is_speaking.is_set()