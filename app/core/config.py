import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL: str = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/upsc_rag")
UPLOAD_DIR: Path = BASE_DIR / "uploads"
EXTRACTED_DIR: Path = BASE_DIR / "data" / "extracted"
PREPROCESSED_DIR: Path = BASE_DIR / "data" / "preprocessed"
ALLOWED_CLASSIFICATIONS = ["History", "Anthropology"]
QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_MAP: dict[str, str] = {
    "History": "history_collection",
    "Anthropology": "anthropology_collection",
}
EMBEDDING_MODEL_NAME: str = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSION: int = 384

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR / "history").mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR / "anthropology").mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
