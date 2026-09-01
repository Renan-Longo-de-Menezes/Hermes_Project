"""Captura de áudio do microfone."""
from __future__ import annotations

import numpy as np
import sounddevice as sd

from hermes.config import Settings


class AudioRecorder:
    """Gerencia a captura contínua de áudio via sounddevice."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sample_rate = settings.sample_rate
        self.channels = settings.channels
        self.chunk = settings.chunk_samples

    def stream(self):
        """Abre um fluxo de entrada de áudio (context manager)."""
        return sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.chunk,
        )

    @staticmethod
    def save(audio: np.ndarray, path, sample_rate: int) -> None:
        """Salva o áudio como WAV de 16 bits."""
        import wave

        audio = np.asarray(audio, dtype=np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
