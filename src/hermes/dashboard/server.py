"""Dashboard web do HERMES via FastAPI."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hermes.config import settings


app = FastAPI(title="HERMES Dashboard")

# Templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Estado global (compartilhado com main.py)
_state = {
    "status": "idle",  # idle | listening | recording | processing
    "current_transcript": "",
    "current_summary": "",
    "last_update": None,
    "session_count": 0,
}

# Histórico de sessões
_history: list[dict] = []


def get_state() -> dict:
    return _state


def update_state(status: str, transcript: str = "", summary: str = "") -> None:
    _state["status"] = status
    _state["current_transcript"] = transcript
    _state["current_summary"] = summary
    _state["last_update"] = datetime.now().isoformat()


def add_to_history(
    transcript: str,
    summary: str,
    tema: str,
    mode: str,
    pdf_path: Path,
    duration: float,
) -> None:
    entry = {
        "id": len(_history) + 1,
        "timestamp": datetime.now().isoformat(),
        "tema": tema,
        "mode": mode,
        "duration": duration,
        "transcript": transcript,
        "summary": summary,
        "pdf_path": str(pdf_path),
    }
    _history.append(entry)
    _state["session_count"] = len(_history)


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html", {"state": _state, "history": _history})


@app.get("/api/status")
async def api_status():
    return _state


@app.get("/api/history")
async def api_history():
    return _history


@app.get("/api/pdf/{pdf_id}")
async def api_pdf(pdf_id: int):
    if pdf_id < 1 or pdf_id > len(_history):
        return {"error": "PDF não encontrado"}
    entry = _history[pdf_id - 1]
    pdf_path = Path(entry["pdf_path"])
    if not pdf_path.exists():
        return {"error": "Arquivo não existe"}
    return FileResponse(pdf_path, media_type="application/pdf")


@app.get("/api/export/{session_id}/{format}")
async def api_export(session_id: int, format: str):
    """Exporta conteúdo em diferentes formatos."""
    if session_id < 1 or session_id > len(_history):
        return {"error": "Sessão não encontrada"}
    entry = _history[session_id - 1]

    if format == "json":
        return entry
    elif format == "txt":
        return HTMLResponse(content=entry["summary"], media_type="text/plain")
    else:
        return {"error": f"Formato '{format}' não suportado"}
