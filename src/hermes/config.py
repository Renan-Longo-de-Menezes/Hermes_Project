"""Configurações centralizadas do HERMES (carrega do .env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


def _find_env() -> Path:
    here = Path(__file__).resolve().parent.parent.parent
    env = here / ".env"
    if env.exists():
        return env
    return here.parent / ".env"


load_dotenv(_find_env())


class Settings(BaseModel):
    # ---- Paths ----
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    temp_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "temp")
    output_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "output")
    memory_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "memory_db")

    # ---- Whisper / STT ----
    whisper_model: str = os.getenv("whisper_model", "small")
    whisper_device: str = os.getenv("whisper_device", "cpu")
    whisper_compute_type: str = os.getenv("whisper_compute_type", "int8")

    # ---- LLM ----
    llm_provider: str = os.getenv("llm_provider", "ollama")
    ollama_model: str = os.getenv("ollama_model", "llama3.1")
    ollama_host: str = os.getenv("ollama_host", "http://localhost:11434")
    openai_model: str = os.getenv("openai_model", "gpt-4o-mini")
    openai_api_key: str = os.getenv("openai_api_key", "")

    # ---- Resumo ----
    summary_language: str = os.getenv("summary_language", "português")

    # ---- Áudio ----
    sample_rate: int = int(os.getenv("sample_rate", "16000"))
    channels: int = int(os.getenv("channels", "1"))
    chunk_samples: int = int(os.getenv("chunk_samples", "4096"))
    max_recording_seconds: int = int(os.getenv("max_recording_seconds", "300"))

    # ---- Wake word ----
    keyboard_mode: bool = os.getenv("keyboard_mode", "true").lower() in ("1", "true", "yes")
    wake_word_model: str = os.getenv("wake_word_model", "hey_jarvis")
    wake_threshold: float = float(os.getenv("wake_threshold", "0.5"))

    # ---- Diariação ----
    hf_token: str = os.getenv("HF_TOKEN", "")

    # ---- Fase 2: Memória ----
    memory_enabled: bool = os.getenv("memory_enabled", "true").lower() in ("1", "true", "yes")
    memory_collection: str = os.getenv("memory_collection", "hermes_memories")
    memory_top_k: int = int(os.getenv("memory_top_k", "3"))

    # ---- Fase 2: Referências ----
    semantic_scholar_enabled: bool = os.getenv("semantic_scholar_enabled", "true").lower() in ("1", "true", "yes")
    semantic_scholar_max_papers: int = int(os.getenv("semantic_scholar_max_papers", "3"))

    # ---- Fase 2: TTS ----
    tts_enabled: bool = os.getenv("tts_enabled", "false").lower() in ("1", "true", "yes")
    tts_voice: str = os.getenv("tts_voice", "pt-BR-FranciscaNeural")

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context) -> None:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
