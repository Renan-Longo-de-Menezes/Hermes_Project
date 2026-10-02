"""Interface de alto nível para memória do HERMES."""
from __future__ import annotations

import datetime
import uuid
from pathlib import Path

from hermes.config import Settings
from hermes.memory.vector_store import VectorStore


class Memory:
    """Gerencia memória de longo prazo do HERMES."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = VectorStore(
            persist_dir=settings.memory_dir,
            collection_name=settings.memory_collection,
        ) if settings.memory_enabled else None

    @property
    def enabled(self) -> bool:
        return self.store is not None

    def save(
        self,
        transcript: str,
        summary: str,
        tema: str,
        mode: str,
        duration: float,
    ) -> str:
        """Salva uma conversa na memória."""
        if not self.enabled:
            return ""

        content = f"TEMA: {tema}\nMODO: {mode}\nRESUMO:\n{summary}\n\nTRANSCRIÇÃO:\n{transcript}"
        metadata = {
            "tema": tema,
            "mode": mode,
            "duration": duration,
            "date": datetime.datetime.now().isoformat(),
        }
        doc_id = f"hermes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.store.add(content=content, metadata=metadata, doc_id=doc_id)
        print(f"💾 Salvo na memória ({self.store.count} registros totais)")
        return doc_id

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        """Recupera memórias relevantes para a query."""
        if not self.enabled:
            return []
        return self.store.search(query, top_k=top_k)

    def format_context(self, query: str, top_k: int = 3) -> str:
        """Formata memórias relevantes como contexto para o LLM."""
        memories = self.recall(query, top_k)
        if not memories:
            return ""

        lines = ["# Conversas anteriores relevantes:\n"]
        for i, mem in enumerate(memories, 1):
            date = mem["metadata"].get("date", "desconhecida")[:10]
            tema = mem["metadata"].get("tema", "—")
            lines.append(f"## Memória {i} ({tema}, {date}) — relevância: {mem['relevance']}")
            # Pega só o resumo (primeiros 500 chars) para não estourar contexto
            content = mem["content"][:500]
            lines.append(content)
            lines.append("")
        return "\n".join(lines)
