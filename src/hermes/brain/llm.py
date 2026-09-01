"""Cérebro do HERMES: interpreta a transcrição e gera o resumo."""
from __future__ import annotations

from hermes.config import Settings

SUMMARY_PROMPT = """Você é o HERMES, um assistente que observa conversas das quais não participa \
e produz resumos úteis e bem organizados.

Abaixo está a transcrição de uma conversa (pode conter múltiplos falantes):

--- INÍCIO DA TRANSCRIÇÃO ---
{transcript}
--- FIM DA TRANSCRIÇÃO ---

Gere um resumo estruturado em {language}, com as seguintes seções:

1. **Tema principal** — identifique em poucas frases sobre o que a conversa tratou.
2. **Pontos-chave** — liste os principais assuntos/argumentos discutidos.
3. **Conclusões / encaminhamentos** — o que ficou decidido ou concluído.
4. **Próximos passos** — ações ou pendências identificadas (se houver).

Seja claro, objetivo e fiel ao conteúdo. Não invente informações que não estejam na transcrição.
Responda já formatado para ser inserido em um documento."""


class Summarizer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def summarize(self, transcript: str) -> str:
        prompt = SUMMARY_PROMPT.format(
            transcript=transcript,
            language=self.settings.summary_language,
        )
        if self.settings.llm_provider == "openai":
            return self._summarize_openai(prompt)
        return self._summarize_ollama(prompt)

    # ------------------------------------------------------------------
    def _summarize_ollama(self, prompt: str) -> str:
        import ollama

        print(f"🤖 Resumindo com Ollama ({self.settings.ollama_model})...")
        client = ollama.Client(host=self.settings.ollama_host)
        response = client.chat(
            model=self.settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()

    # ------------------------------------------------------------------
    def _summarize_openai(self, prompt: str) -> str:
        from openai import OpenAI

        print(f"🤖 Resumindo com OpenAI ({self.settings.openai_model})...")
        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
