"""Extratores especializados de conteúdo (TODO, Flashcards, Ata)."""
from __future__ import annotations

from hermes.config import Settings
from hermes.brain.llm_client import LLMClient  # ← agora importa daqui


class Extractor:
    """Extrai informações estruturadas da transcrição via LLM."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)

    # ------------------------------------------------------------------
    # Modo 4: TODO list
    # ------------------------------------------------------------------
    def extract_todos(self, transcript: str) -> str:
        prompt = (
            "Você é um assistente de produtividade. Analise a transcrição abaixo e extraia "
            "TODAS as tarefas, compromissos e ações mencionadas.\n\n"
            "Gere uma lista em Markdown com:\n"
            "1. **Tarefas identificadas** (lista com bullets, cada uma começando com [ ])\n"
            "2. **Responsáveis** (se mencionados)\n"
            "3. **Prazos** (se mencionados)\n"
            "4. **Prioridade sugerida** (alta/média/baixa) para cada tarefa\n\n"
            "Se não houver tarefas explícitas, inferir ações implícitas.\n\n"
            f"Transcrição:\n{transcript}"
        )
        return self.llm.chat(prompt)

    # ------------------------------------------------------------------
    # Modo 5: Flashcards (formato Anki CSV)
    # ------------------------------------------------------------------
    def extract_flashcards(self, transcript: str) -> str:
        prompt = (
            "Você é um especialista em aprendizagem e memorização. Analise a transcrição abaixo "
            "e crie flashcards no formato pergunta/resposta.\n\n"
            "Regras:\n"
            "- Cada flashcard deve ter uma PERGUNTA clara e uma RESPOSTA concisa\n"
            "- Foque em fatos, conceitos, definições e informações-chave\n"
            "- Crie entre 5 e 10 flashcards\n"
            "- Use linguagem simples e direta\n\n"
            "Formato de saída (Markdown):\n"
            "### Flashcard 1\n"
            "**Pergunta:** [pergunta]\n"
            "**Resposta:** [resposta]\n\n"
            "### Flashcard 2\n"
            "... (repetir)\n\n"
            f"Transcrição:\n{transcript}"
        )
        return self.llm.chat(prompt)

    def export_flashcards_csv(self, flashcards_md: str) -> str:
        """Converte flashcards Markdown para CSV compatível com Anki."""
        import re

        cards = re.findall(
            r"\*\*Pergunta:\*\*\s*(.*?)\n\*\*Resposta:\*\*\s*(.*?)(?=\n###|\Z)",
            flashcards_md,
            re.DOTALL,
        )

        lines = ["frente;verso"]
        for pergunta, resposta in cards:
            pergunta = pergunta.strip().replace(";", ",").replace("\n", " ")
            resposta = resposta.strip().replace(";", ",").replace("\n", " ")
            lines.append(f"{pergunta};{resposta}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Modo 6: Ata de reunião
    # ------------------------------------------------------------------
    def extract_ata(self, transcript: str, tema: str = "") -> str:
        prompt = (
            "Você é um secretário executivo especializado em redação de atas de reunião.\n\n"
            "Analise a transcrição abaixo e gere uma ATA FORMAL em Markdown com:\n\n"
            "1. **Cabeçalho**:\n"
            "   - Data e hora (inferir ou usar 'não informada')\n"
            "   - Local (se mencionado)\n"
            "   - Participantes (se identificáveis)\n"
            "   - Pauta / Assunto\n\n"
            "2. **Deliberações** (lista numerada das decisões tomadas)\n\n"
            "3. **Encaminhamentos** (lista com responsável e prazo, se houver)\n\n"
            "4. **Observações** (pontos relevantes não decididos)\n\n"
            "5. **Encerramento** (parágrafo final)\n\n"
            "Use linguagem formal e impessoal. Se alguma informação não estiver na transcrição, "
            "indique como 'não informado'.\n\n"
            f"Tema da reunião: {tema or 'não especificado'}\n\n"
            f"Transcrição:\n{transcript}"
        )
        return self.llm.chat(prompt)
