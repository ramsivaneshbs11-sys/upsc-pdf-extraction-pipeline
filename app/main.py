"""
app/main.py
────────────
FastAPI application entry point.

Start server:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""
import logging
import sys
from pathlib import Path

# ── Ensure workspace root is on sys.path ───────────────────────────────────
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent  # daily/
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

# Disable HuggingFace symlinks on Windows (required for Docling models)
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.database.session import engine, Base
from app.api.routes import documents
from app.services.qdrant_service import ensure_collections


# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: create tables + Qdrant collections on startup ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — creating database tables if needed …")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready ✓")

    logger.info("Verifying Qdrant collections …")
    ensure_collections()

    yield
    logger.info("Shutting down …")



# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="UPSC RAG — Document Ingestion API",
    description=(
        "Single endpoint to register and extract UPSC PDF documents.\n\n"
        "**Pipeline:** Validate → Save → Register (PostgreSQL) → Extract (Docling) → Update status"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(documents.router)


# ── Health check ──────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}
