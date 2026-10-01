"""Interface com LLMs para resumo e análise."""
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
        """Envia um prompt e retorna a resposta."""
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


class Summarizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)

    def summarize(self, transcript: str, mode: str = "resumo") -> str:
        """Gera análise da transcrição conforme o modo escolhido.

        Modos:
          - resumo: resumo clássico + referências
          - pros_contras: análise de prós e contras
          - moral: análise moral/ética
        """
        prompts = {
            "resumo": self._prompt_resumo,
            "pros_contras": self._prompt_pros_contras,
            "moral": self._prompt_moral,
        }

        prompt_fn = prompts.get(mode, self._prompt_resumo)
        prompt = prompt_fn(transcript)

        print(f"🤖 Resumindo com {self.settings.llm_provider} ({self.settings.ollama_model})...")
        return self.llm.chat(prompt)

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    @staticmethod
    def _prompt_resumo(transcript: str) -> str:
        return (
            "Você é um assistente especializado em analisar transcrições de áudio.\n\n"
            "Analise a transcrição abaixo e gere um relatório estruturado em Markdown com:\n"
            "1. **Tema principal** (1-2 frases)\n"
            "2. **Pontos-chave** (lista com bullets)\n"
            "3. **Conclusões / Encaminhamentos**\n"
            "4. **Próximos passos** (se houver)\n"
            "5. **Referências acadêmicas sugeridas** (2-3 temas/palavras-chave para pesquisa)\n\n"
            f"Transcrição:\n{transcript}"
        )

    @staticmethod
    def _prompt_pros_contras(transcript: str) -> str:
        return (
            "Você é um analista crítico especializado em debates e discussões.\n\n"
            "Analise a transcrição abaixo e gere um relatório estruturado em Markdown com:\n"
            "1. **Tema central** (1-2 frases)\n"
            "2. **Argumentos a favor** (lista com bullets)\n"
            "3. **Argumentos contra** (lista com bullets)\n"
            "4. **Pontos neutros / fatos objetivos**\n"
            "5. **Conclusão equilibrada** (parágrafo)\n\n"
            f"Transcrição:\n{transcript}"
        )

    @staticmethod
    def _prompt_moral(transcript: str) -> str:
        return (
            "Você é um especialista em ética e filosofia moral.\n\n"
            "Analise a transcrição abaixo sob uma perspectiva ética/moral e gere um relatório estruturado em Markdown com:\n"
            "1. **Tema central** (1-2 frases)\n"
            "2. **Questões éticas identificadas** (lista)\n"
            "3. **Análise sob diferentes perspectivas morais**:\n"
            "   - Utilitarismo (consequências)\n"
            "   - Deontologia (deveres/regras)\n"
            "   - Ética das virtudes (caráter)\n"
            "4. **Dilemas morais** (se houver)\n"
            "5. **Conclusão ética** (parágrafo)\n\n"
            f"Transcrição:\n{transcript}"
        )
