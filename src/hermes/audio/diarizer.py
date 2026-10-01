"""Diariação de falantes usando pyannote.audio.

Funciona apenas se HF_TOKEN estiver configurado e pyannote.audio instalado.
Sem token, o módulo retorna lista vazia e o pipeline segue sem rotular os falantes.
"""
from __future__ import annotations

from pathlib import Path

from hermes.config import Settings


def _fmt_ts(seconds: float) -> str:
    """Formata segundos como MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _speaker_label(raw: str) -> str:
    """Converte SPEAKER_00 → Pessoa 1, SPEAKER_01 → Pessoa 2."""
    if raw.startswith("SPEAKER_"):
        try:
            num = int(raw.split("_")[1]) + 1
            return f"Pessoa {num}"
        except (IndexError, ValueError):
            pass
    return raw


def format_dialogue(segments: list[dict]) -> str:
    """Formata segmentos rotulados como diálogo legível para o LLM/PDF."""
    lines = []
    for seg in segments:
        speaker = _speaker_label(seg.get("speaker", ""))
        ts = f"{_fmt_ts(seg['start'])} - {_fmt_ts(seg['end'])}"
        if speaker:
            lines.append(f"[{speaker}] ({ts}): {seg['text']}")
        else:
            lines.append(f"({ts}): {seg['text']}")
    return "\n".join(lines)


class Diarizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pipeline = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.hf_token)

    def _load(self):
        if self._pipeline is None:
            if not self.settings.hf_token:
                raise ValueError("HF_TOKEN não configurado — diariação desabilitada")
            try:
                from pyannote.audio import Pipeline
            except ImportError:
                raise ImportError(
                    "pyannote.audio não instalado. Rode: pip install pyannote.audio"
                )

            print("🗣️  Carregando modelo de diariação (pyannote/speaker-diarization-3.1)...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.settings.hf_token,
            )

            # GPU opcional
            device = self.settings.whisper_device.lower()
            if device in ("auto", "cuda"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._pipeline.to(torch.device("cuda"))
                        print("    Usando CUDA para diariação")
                except Exception:
                    pass
        return self._pipeline

    def diarize(self, audio_path: Path) -> list[dict]:
        """Retorna [{start, end, speaker}, ...] em ordem cronológica."""
        if not self.enabled:
            print("⚠️  HF_TOKEN não configurado — diariação desativada.")
            return []
        pipeline = self._load()
        diarization = pipeline(str(audio_path))
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker,
            })
        return segments

    # ------------------------------------------------------------------
    # Fusão transcrição + diariação
    # ------------------------------------------------------------------
    @staticmethod
    def _dominant_speaker(start: float, end: float, diarization: list[dict]) -> str:
        """Retorna o falante com maior sobreposição temporal com [start, end]."""
        best_speaker = ""
        best_overlap = 0.0
        for turn in diarization:
            overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        return best_speaker

    @staticmethod
    def merge(transcript_segments: list[dict], diarization: list[dict]) -> list[dict]:
        """Atribui um falante a cada trecho transcrito."""
        result = []
        for seg in transcript_segments:
            if diarization:
                speaker = Diarizer._dominant_speaker(seg["start"], seg["end"], diarization)
            else:
                speaker = ""
            result.append({**seg, "speaker": speaker})
        return result
