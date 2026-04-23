from __future__ import annotations

import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.agent import router as agent_router
from app.api.ask import router as ask_router
from app.api.documents import router as documents_router
from app.api.evaluation import router as evaluation_router
from app.api.retrieve import router as retrieve_router
from app.core.logging import setup_logging

setup_logging()
log = logging.getLogger("app.main")


def _build_allowed_origins() -> list[str]:
    origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    env_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
    if env_origins:
        extra = [o.strip() for o in env_origins.split(",") if o.strip()]
        origins.extend(extra)

    # de-duplicate while preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for origin in origins:
        if origin not in seen:
            deduped.append(origin)
            seen.add(origin)

    return deduped


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.getenv("DATABASE_URL")
    api_base_url = os.getenv("API_BASE_URL", "not-set")
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "not-set")

    log.info("[startup] Agentic SQL RAG API starting")
    log.info("[startup] PID=%s", os.getpid())
    log.info("[startup] Python=%s", sys.executable)
    log.info("[startup] CWD=%s", os.getcwd())
    log.info("[startup] DATABASE_URL set? %s", "YES" if db_url else "NO")
    log.info("[startup] API_BASE_URL=%s", api_base_url)
    log.info("[startup] ALLOWED_ORIGINS=%s", allowed_origins)
    log.info(
        "[startup] Expected routes: /, /health, /ask, /agent/ask, "
        "/documents, /documents/upload, /retrieval/retrieve, "
        "/evaluation/summary, /evaluation/runs, /docs"
    )

    yield

    log.info("[shutdown] Agentic SQL RAG API stopping")


app = FastAPI(
    title="Agentic SQL RAG API",
    version="0.3.0",
    description=(
        "Backend API for retrieval, grounded answer generation, agentic retrieval, "
        "citations, evidence inspection, document upload/indexing, evaluation metrics, "
        "and rerank-aware responses."
    ),
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log.exception(
        "[error] Unhandled exception on %s %s\n%s",
        request.method,
        request.url.path,
        tb,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": f"{type(exc).__name__}: {exc}",
            "path": request.url.path,
            "method": request.method,
        },
    )


allowed_origins = _build_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(retrieve_router, prefix="/retrieval", tags=["retrieval"])
app.include_router(ask_router, tags=["generation"])
app.include_router(agent_router, tags=["agent"])
app.include_router(documents_router)
app.include_router(evaluation_router, tags=["evaluation"])


@app.get("/")
def root():
    return {
        "name": "Agentic SQL RAG API",
        "status": "running",
        "version": "0.3.0",
        "docs": "/docs",
        "expected_routes": {
            "health": "/health",
            "ask": "/ask",
            "agent_ask": "/agent/ask",
            "documents_list": "/documents",
            "documents_upload": "/documents/upload",
            "retrieval": "/retrieval/retrieve",
            "evaluation_summary": "/evaluation/summary",
            "evaluation_runs": "/evaluation/runs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "agentic-sql-rag-api",
        "version": "0.3.0",
    }
