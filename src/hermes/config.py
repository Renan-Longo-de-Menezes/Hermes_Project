"""Configurações globais do HERMES (carregadas de variáveis de ambiente)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv 

load_dotenv()


def _bool(value: str, default: bool = False) -> bool: 
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "sim"}


@dataclass
class Settings:
    # ----- Acionamento -----
    keyboard_mode: bool = _bool(os.getenv("HERMES_KEYBOARD_MODE"), default=True)
    wake_word_model: str = os.getenv("HERMES_WAKE_WORD_MODEL", "hey_jarvis")
    wake_threshold: float = float(os.getenv("HERMES_WAKE_THRESHOLD", "0.5"))

    # ----- Áudio -----
    sample_rate: int = 16_000
    channels: int = 1
    chunk_samples: int = 1_280  # 80 ms a 16 kHz (exigido pelo openWakeWord)
    max_recording_seconds: int = int(os.getenv("HERMES_MAX_RECORDING_SECONDS", "3600"))

    # ----- Transcrição -----
    whisper_model: str = os.getenv("HERMES_WHISPER_MODEL", "small")
    whisper_device: str = os.getenv("HERMES_WHISPER_DEVICE", "auto")
    whisper_compute_type: str = os.getenv("HERMES_WHISPER_COMPUTE_TYPE", "auto")

    # ----- LLM -----
    llm_provider: str = os.getenv("HERMES_LLM_PROVIDER", "ollama").lower()
    ollama_model: str = os.getenv("HERMES_OLLAMA_MODEL", "llama3.1")
    ollama_host: str = os.getenv("HERMES_OLLAMA_HOST", "http://localhost:11434")
    openai_model: str = os.getenv("HERMES_OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    summary_language: str = os.getenv("HERMES_SUMMARY_LANGUAGE", "português")

    # ----- Pastas -----
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    output_dir: Path = Path(os.getenv("HERMES_OUTPUT_DIR", "output"))
    temp_dir: Path = Path(os.getenv("HERMES_TEMP_DIR", "tmp"))

    def __post_init__(self) -> None:
        # Resolve caminhos relativos à raiz do projeto
        if not self.output_dir.is_absolute():
            self.output_dir = self.base_dir / self.output_dir
        if not self.temp_dir.is_absolute():
            self.temp_dir = self.base_dir / self.temp_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
