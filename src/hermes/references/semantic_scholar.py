"""Cliente da API Semantic Scholar para buscar referências acadêmicas."""
from __future__ import annotations

import httpx


class SemanticScholarClient:
    """Busca papers acadêmicos via Semantic Scholar API (gratuita)."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, max_papers: int = 3):
        self.max_papers = max_papers
        self._client = httpx.Client(timeout=10.0)

    def search(self, query: str) -> list[dict]:
        """Busca papers relacionados à query. Retorna lista de dicts."""
        try:
            response = self._client.get(
                self.BASE_URL,
                params={
                    "query": query,
                    "limit": self.max_papers,
                    "fields": "title,year,authors,abstract,citationCount,externalIds",
                },
                headers={"User-Agent": "HermesProject/1.0"},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            print(f"⚠️  Erro ao buscar referências: {exc}")
            return []

        papers = []
        for paper in data.get("data", []):
            authors = ", ".join(
                a.get("name", "") for a in (paper.get("authors") or [])[:3]
            )
            if len(paper.get("authors") or []) > 3:
                authors += " et al."

            # Tenta pegar DOI ou link
            ext_ids = paper.get("externalIds") or {}
            doi = ext_ids.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else ""

            papers.append({
                "title": paper.get("title", "Sem título"),
                "year": paper.get("year", "—"),
                "authors": authors or "Autores desconhecidos",
                "abstract": (paper.get("abstract") or "")[:200],
                "citations": paper.get("citationCount", 0),
                "link": link,
            })
        return papers

    def format_references(self, query: str) -> str:
        """Busca e formata referências em Markdown."""
        papers = self.search(query)
        if not papers:
            return ""

        lines = ["# Referências acadêmicas sugeridas:\n"]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. **{p['title']}**")
            lines.append(f"   - Autores: {p['authors']}")
            lines.append(f"   - Ano: {p['year']} · Citações: {p['citations']}")
            if p["link"]:
                lines.append(f"   - Link: {p['link']}")
            lines.append("")
        return "\n".join(lines)
