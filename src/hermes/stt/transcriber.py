"""Transcrição de fala para texto usando faster-whisper."""
from __future__ import annotations

from pathlib import Path

from hermes.config import Settings


class Transcriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    def _load(self):
        if self._model is None:
            print(f"🧠 Carregando modelo Whisper '{self.settings.whisper_model}'...")
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.settings.whisper_model,
                device="cpu",
                compute_type="int8",
            )
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        """Retorna o texto transcrito de um arquivo WAV."""
        model = self._load()
        segments, _info = model.transcribe(
            str(audio_path),
            language="pt",
            vad_filter=True,
            beam_size=5,
        )
        texts = [segment.text.strip() for segment in segments]
        return "\n".join(texts)
