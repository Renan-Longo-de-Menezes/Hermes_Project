"""Interface com LLMs para resumo e análise (com RAG, referências e extractors)."""
from __future__ import annotations

from hermes.config import Settings
from hermes.memory.memory import Memory
from hermes.references.semantic_scholar import SemanticScholarClient
from hermes.brain.llm_client import LLMClient  # ← agora importa daqui
from hermes.brain.extractors import Extractor


class Summarizer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings)
        self.memory = Memory(settings)
        self.references = SemanticScholarClient(
            max_papers=settings.semantic_scholar_max_papers
        ) if settings.semantic_scholar_enabled else None
        self.extractor = Extractor(settings)

    def summarize(self, transcript: str, mode: str = "resumo") -> str:
        extractors = {
            "todo": lambda: self.extractor.extract_todos(transcript),
            "flashcards": lambda: self.extractor.extract_flashcards(transcript),
            "ata": lambda: self.extractor.extract_ata(transcript),
        }
        if mode in extractors:
            print(f"🤖 Extraindo com {self.settings.llm_provider} ({self.settings.ollama_model})...")
            return extractors[mode]()

        prompts = {
            "resumo": self._prompt_resumo,
            "pros_contras": self._prompt_pros_contras,
            "moral": self._prompt_moral,
        }
        prompt_fn = prompts.get(mode, self._prompt_resumo)

        context = ""
        if self.memory.enabled:
            context = self.memory.format_context(transcript, top_k=self.settings.memory_top_k)

        refs = ""
        if self.references:
            print("📚 Buscando referências acadêmicas (Semantic Scholar)...")
            refs = self.references.format_references(transcript[:300])

        prompt = prompt_fn(transcript, context, refs)

        print(f"🤖 Analisando com {self.settings.llm_provider} ({self.settings.ollama_model})...")
        return self.llm.chat(prompt)

    # ------------------------------------------------------------------
    # Prompts clássicos
    # ------------------------------------------------------------------
    @staticmethod
    def _prompt_resumo(transcript: str, context: str, refs: str) -> str:
        base = (
            "Você é um assistente especializado em analisar transcrições de áudio.\n\n"
            "Analise a transcrição abaixo e gere um relatório estruturado em Markdown com:\n"
            "1. **Tema principal** (1-2 frases)\n"
            "2. **Pontos-chave** (lista com bullets)\n"
            "3. **Conclusões / Encaminhamentos**\n"
            "4. **Próximos passos** (se houver)\n"
        )
        if context:
            base += f"\n\n# Contexto de conversas anteriores:\n{context}\n"
        if refs:
            base += f"\n\n# Referências acadêmicas encontradas:\n{refs}\n"
            base += "\nInclua as referências acima no final do relatório, em uma seção 'Referências'."
        base += f"\n\nTranscrição:\n{transcript}"
        return base

    @staticmethod
    def _prompt_pros_contras(transcript: str, context: str, refs: str) -> str:
        base = (
            "Você é um analista crítico especializado em debates e discussões.\n\n"
            "Analise a transcrição abaixo e gere um relatório estruturado em Markdown com:\n"
            "1. **Tema central** (1-2 frases)\n"
            "2. **Argumentos a favor** (lista com bullets)\n"
            "3. **Argumentos contra** (lista com bullets)\n"
            "4. **Pontos neutros / fatos objetivos**\n"
            "5. **Conclusão equilibrada** (parágrafo)\n"
        )
        if context:
            base += f"\n\n# Contexto de conversas anteriores:\n{context}\n"
        if refs:
            base += f"\n\n# Referências acadêmicas encontradas:\n{refs}\n"
            base += "\nInclua as referências acima no final do relatório."
        base += f"\n\nTranscrição:\n{transcript}"
        return base

    @staticmethod
    def _prompt_moral(transcript: str, context: str, refs: str) -> str:
        base = (
            "Você é um especialista em ética e filosofia moral.\n\n"
            "Analise a transcrição abaixo sob uma perspectiva ética/moral e gere um relatório estruturado em Markdown com:\n"
            "1. **Tema central** (1-2 frases)\n"
            "2. **Questões éticas identificadas** (lista)\n"
            "3. **Análise sob diferentes perspectivas morais**:\n"
            "   - Utilitarismo (consequências)\n"
            "   - Deontologia (deveres/regras)\n"
            "   - Ética das virtudes (caráter)\n"
            "4. **Dilemas morais** (se houver)\n"
            "5. **Conclusão ética** (parágrafo)\n"
        )
        if context:
            base += f"\n\n# Contexto de conversas anteriores:\n{context}\n"
        if refs:
            base += f"\n\n# Referências acadêmicas encontradas:\n{refs}\n"
            base += "\nInclua as referências acima no final do relatório."
        base += f"\n\nTranscrição:\n{transcript}"
        return base
