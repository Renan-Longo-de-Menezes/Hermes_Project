"""Cliente LLM isolado (sem dependências circulares)."""
from __future__ import annotations

from hermes.config import Settings


class LLMClient:
    """Cliente unificado para Ollama e OpenAI."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self.settings.llm_provider == "ollama":
                from ollama import Client
                self._client = Client(host=self.settings.ollama_host)
            else:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def chat(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.settings.llm_provider == "ollama":
            response = self.client.chat(
                model=self.settings.ollama_model,
                messages=messages,
            )
            return response["message"]["content"]
        else:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
            )
            return response.choices[0].message.content
