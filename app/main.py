"""FastAPI web application for shared-embedding SGD verification."""

from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Importing the engine pins torch to one CPU thread for these tiny matrices.
from .engine import ValidationError, evaluate
from .samples import SAMPLES

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Shared-Embedding SGD Verifier", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/samples")
def samples() -> dict:
    return {"samples": SAMPLES}


@app.post("/api/run")
def run(payload: dict = Body(...)) -> dict:
    try:
        return evaluate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
