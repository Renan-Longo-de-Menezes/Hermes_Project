"""Transcrição de fala para texto usando faster-whisper."""
from __future__ import annotations

from pathlib import Path

from hermes.config import Settings


class Transcriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    def _resolve_device(self) -> str:
        device = self.settings.whisper_device.lower()
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device

    def _resolve_compute_type(self, device: str) -> str:
        ct = self.settings.whisper_compute_type.lower()
        if ct != "auto":
            return ct
        return "float16" if device == "cuda" else "int8"

    def _load(self):
        if self._model is None:
            device = self._resolve_device()
            compute_type = self._resolve_compute_type(device)
            print(f" Carregando modelo Whisper '{self.settings.whisper_model}' "
                  f"(device={device}, compute_type={compute_type})...")
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.settings.whisper_model,
                device=device,
                compute_type=compute_type,
            )
        return self._model

    def transcribe_segments(self, audio_path: Path) -> list[dict]:
        """Retorna segmentos com timestamps e palavras (para diarização)."""
        model = self._load()
        segments, _info = model.transcribe(
            str(audio_path),
            language="pt",
            vad_filter=True,
            beam_size=5,
            word_timestamps=True,
        )
        result = []
        for seg in segments:
            words = [
                {"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word}
                for w in (seg.words or [])
            ]
            result.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "words": words,
            })
        return result

    def transcribe(self, audio_path: Path) -> str:
        """Retorna apenas o texto concatenado (retrocompatível)."""
        segments = self.transcribe_segments(audio_path)
        return "\n".join(seg["text"] for seg in segments)
