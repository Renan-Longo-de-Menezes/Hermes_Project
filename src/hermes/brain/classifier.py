"""Classificador de tema da conversa via LLM."""
from __future__ import annotations

from hermes.config import Settings
from hermes.brain.llm_client import LLMClient


# Categorias possíveis
CATEGORIES = [
    "aula",
    "debate",
    "reunião",
    "entrevista",
    "apresentação",
    "conversa informal",
    "notícia",
    "outro",
]


class Classifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)

    def classify(self, transcript: str) -> str:
        """Classifica o tema da conversa. Retorna uma das CATEGORIES."""
        prompt = (
            "Classifique a transcrição abaixo em UMA das seguintes categorias:\n"
            + ", ".join(CATEGORIES)
            + "\n\n"
            "Responda APENAS com o nome da categoria, sem explicações.\n\n"
            f"Transcrição:\n{transcript[:2000]}"  # limita para economizar tokens
        )
        response = self.llm.chat(prompt).strip().lower()

        # Valida se a resposta está nas categorias
        for cat in CATEGORIES:
            if cat in response:
                return cat
        return "outro"
