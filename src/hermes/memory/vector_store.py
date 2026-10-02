"""Banco de dados vetorial para memória do HERMES (ChromaDB)."""
from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    """Wrapper simples sobre ChromaDB para armazenar conversas passadas."""

    def __init__(self, persist_dir: Path, collection_name: str = "hermes_memories"):
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        content: str,
        metadata: dict,
        doc_id: Optional[str] = None,
    ) -> str:
        """Adiciona um documento à memória. Retorna o ID."""
        doc_id = doc_id or str(uuid.uuid4())
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id],
        )
        return doc_id

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Busca documentos similares à query."""
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
        )
        memories = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            memories.append({
                "content": doc,
                "metadata": meta,
                "relevance": round(1 - distance, 3),
            })
        return memories

    @property
    def count(self) -> int:
        return self.collection.count()
