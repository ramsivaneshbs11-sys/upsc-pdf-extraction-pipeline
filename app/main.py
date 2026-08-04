import logging
import sys
from pathlib import Path
import os

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database.session import engine, Base
from app.api.routes import documents
from app.services.qdrant_service import ensure_collections

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up - creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")
    ensure_collections()
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="UPSC RAG - Document Ingestion API",
    description="Single endpoint to register and extract UPSC PDF documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(documents.router)

@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}
