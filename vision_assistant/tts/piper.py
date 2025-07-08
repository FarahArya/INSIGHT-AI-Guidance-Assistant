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

        # Piper command to output to stdout
        piper_cmd = [
            self.config.piper_path,
            "--model", self.config.piper_model,
            "--config", self.config.piper_config,
            "--output_file", "-"  # Output to stdout
        ]

        # aplay command to read from stdin
        aplay_cmd = ["aplay", "-"]

        # Create the pipeline: piper | aplay
        piper_process = subprocess.Popen(
            piper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        aplay_process = subprocess.Popen(
            aplay_cmd,
            stdin=piper_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Close piper's stdout in the parent process so aplay can receive EOF
        piper_process.stdout.close()

        # Send text to piper
        piper_stdout, piper_stderr = piper_process.communicate(input=text.encode())

        # Wait for aplay to finish
        aplay_stdout, aplay_stderr = aplay_process.communicate()

        # Check for errors
        if piper_process.returncode != 0:
            raise RuntimeError(f"Piper error: {piper_stderr.decode()}")
        if aplay_process.returncode != 0:
            raise RuntimeError(f"aplay error: {aplay_stderr.decode()}")

    def stop(self):
        subprocess.run(["pkill", "-f", "aplay"], check=False)