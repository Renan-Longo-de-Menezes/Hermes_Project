"""Dashboard web do HERMES via FastAPI."""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

from hermes.config import settings


app = FastAPI(title="HERMES Dashboard")

# Templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

HISTORY_FILE = settings.base_dir / "history.json"
_state_lock = threading.Lock()
_history_lock = threading.Lock()

# Estado global (compartilhado com main.py)
_state = {
    "status": "idle",  # idle | listening | recording | processing
    "current_transcript": "",
    "current_summary": "",
    "last_update": None,
    "session_count": 0,
}

# Histórico de sessões (persistido em history.json)
_history: list[dict] = []


def _load_history() -> None:
    """Carrega o histórico salvo em disco (se existir)."""
    global _history
    if not HISTORY_FILE.exists():
        _history = []
        return
    try:
        raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            _history = raw
    except (json.JSONDecodeError, OSError) as exc:
        print(f"⚠️  Não foi possível carregar o histórico: {exc}")
        _history = []


def _save_history() -> None:
    """Persiste o histórico em disco."""
    try:
        HISTORY_FILE.write_text(
            json.dumps(_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"⚠️  Não foi possível salvar o histórico: {exc}")


# Carrega o histórico inicial
_load_history()
_state["session_count"] = len(_history)


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


def update_state(status: str, transcript: str = "", summary: str = "") -> None:
    with _state_lock:
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
    with _history_lock:
        _history.append(entry)
        with _state_lock:
            _state["session_count"] = len(_history)
        _save_history()


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    with _state_lock:
        state = dict(_state)
    with _history_lock:
        history = list(_history)
    return templates.TemplateResponse(request, "index.html", {"state": state, "history": history})


@app.get("/api/status")
async def api_status():
    with _state_lock:
        return dict(_state)


@app.get("/api/history")
async def api_history():
    with _history_lock:
        return list(_history)


@app.get("/api/pdf/{pdf_id}")
async def api_pdf(pdf_id: int):
    with _history_lock:
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
    with _history_lock:
        if session_id < 1 or session_id > len(_history):
            return {"error": "Sessão não encontrada"}
        entry = _history[session_id - 1]

    if format == "json":
        return entry
    elif format == "txt":
        return HTMLResponse(content=entry["summary"], media_type="text/plain")
    else:
        return {"error": f"Formato '{format}' não suportado"}
