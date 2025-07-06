import json
import os
import subprocess
import tempfile
from .base import BaseTTS

class PiperTTS(BaseTTS):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._validate_paths()

    def _validate_paths(self):
        for path in [self.config.piper_path, self.config.piper_model, self.config.piper_config]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Piper TTS file not found: {path}")

    def speak(self, text: str):
        self.is_speaking.set()
        try:
            self._speak_piper(text)
        finally:
            self.is_speaking.clear()

    def _speak_piper(self, text: str):
        # Remove the JSON wrapper - send plain text directly
        subprocess.run(["amixer", "-q", "sset", "Headphone", "90%"], check=False)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_file:
            cmd = [
                self.config.piper_path,
                "--model", self.config.piper_model,
                "--config", self.config.piper_config,
                "--output_file", tmp_file.name
            ]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Send plain text instead of JSON
            stdout, stderr = process.communicate(input=text.encode())

            if process.returncode == 0:
                subprocess.run(["aplay", tmp_file.name], check=True)
            else:
                raise RuntimeError(f"Piper error: {stderr.decode()}")

    def stop(self):
        subprocess.run(["pkill", "-f", "aplay"], check=False)