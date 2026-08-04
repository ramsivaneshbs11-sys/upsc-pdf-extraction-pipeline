import logging
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_MAP, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)
_qdrant_client = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client

def ensure_collections():
    try:
        client = get_qdrant_client()
        existing = {col.name for col in client.get_collections().collections}
        for classification, collection_name in QDRANT_COLLECTION_MAP.items():
            if collection_name not in existing:
                client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE))
    except Exception as exc:
        logger.warning(f"Qdrant collections init warning: {exc}")

def run_qdrant_upsert(file_id: str, classification: str, embedded_chunks: list[dict]):
    collection_name = QDRANT_COLLECTION_MAP.get(classification)
    if not collection_name or not embedded_chunks:
        return False, "Invalid collection or empty chunks"
    try:
        client = get_qdrant_client()
        existing = {col.name for col in client.get_collections().collections}
        if collection_name not in existing:
            client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE))
        points = [PointStruct(id=chunk["chunk_id"], vector=chunk["vector"], payload={"file_id": file_id, "chunk_id": chunk["chunk_id"], "classification": classification, "text": chunk["text"], "metadata": chunk.get("metadata", {})}) for chunk in embedded_chunks]
        client.upsert(collection_name=collection_name, points=points, wait=True)
        return True, None
    except Exception as exc:
        return False, str(exc)
