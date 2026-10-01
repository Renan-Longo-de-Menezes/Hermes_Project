"""Captura e salvamento de áudio via sounddevice."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from hermes.config import Settings


class AudioRecorder:
    def __init__(self, settings: Settings):
        self.settings = settings

    def stream(self):
        """Retorna um stream de áudio aberto para leitura contínua."""
        return sd.InputStream(
            samplerate=self.settings.sample_rate,
            channels=self.settings.channels,
            dtype="int16",
            blocksize=self.settings.chunk_samples,
        )

    @staticmethod
    def save(audio: np.ndarray, path: Path, sample_rate: int) -> None:
        """Salva um array numpy como arquivo WAV."""
        # Garante que a pasta existe (fallback caso config.py não tenha criado)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Normaliza para int16 se vier em float
        if audio.dtype != np.int16:
            audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
