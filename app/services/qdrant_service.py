"""
app/services/qdrant_service.py
───────────────────────────────
Connects to Qdrant vector database (running in Docker at localhost:6333),
manages per-classification collections, and upserts embedded vectors.

Collections:
    - history_collection
    - anthropology_collection

Payload stored per point:
    {
        "file_id":      str,
        "chunk_id":     str,
        "classification": str,
        "text":         str,
        "metadata":     dict,
    }
"""
import logging
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_MAP,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton QdrantClient instance."""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} …")
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info("Qdrant client connected ✓")
    return _qdrant_client


def ensure_collections():
    """
    Idempotently ensure that all required Qdrant collections exist.
    Called on FastAPI application startup.
    """
    try:
        client = get_qdrant_client()
        existing = {col.name for col in client.get_collections().collections}

        for classification, collection_name in QDRANT_COLLECTION_MAP.items():
            if collection_name not in existing:
                logger.info(
                    f"Creating Qdrant collection '{collection_name}' "
                    f"(dim={EMBEDDING_DIMENSION}, Cosine) for '{classification}' …"
                )
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection '{collection_name}' ✓")
            else:
                logger.info(f"Qdrant collection '{collection_name}' exists ✓")
    except Exception as exc:
        logger.warning(
            f"Could not initialize Qdrant collections (is Qdrant running?): {exc}"
        )


def run_qdrant_upsert(
    file_id: str,
    classification: str,
    embedded_chunks: list[dict],
) -> tuple[bool, str | None]:
    """
    Upsert embedded chunk vectors into the corresponding Qdrant collection.

    Args:
        file_id:         UUID of the document record.
        classification:  'History' or 'Anthropology'.
        embedded_chunks: Output list from run_embedding().

    Returns:
        (success: bool, error_message: str | None)
    """
    collection_name = QDRANT_COLLECTION_MAP.get(classification)
    if not collection_name:
        error_msg = f"No Qdrant collection mapped for classification '{classification}'"
        logger.error(f"[{file_id}] {error_msg}")
        return False, error_msg

    if not embedded_chunks:
        error_msg = "No embedded chunks provided for Qdrant upsert."
        logger.error(f"[{file_id}] {error_msg}")
        return False, error_msg

    logger.info(
        f"[{file_id}] Upserting {len(embedded_chunks)} vectors to Qdrant "
        f"collection '{collection_name}' …"
    )

    try:
        client = get_qdrant_client()

        # Re-verify collection exists
        existing = {col.name for col in client.get_collections().collections}
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )

        # Build PointStruct list
        points: list[PointStruct] = []
        for chunk in embedded_chunks:
            payload: dict[str, Any] = {
                "file_id": file_id,
                "chunk_id": chunk["chunk_id"],
                "classification": classification,
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
            }

            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector=chunk["vector"],
                    payload=payload,
                )
            )

        # Upsert in batch
        operation_info = client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        logger.info(
            f"[{file_id}] Qdrant upsert complete ✓ — "
            f"collection='{collection_name}', status={operation_info.status}"
        )
        return True, None

    except Exception as exc:
        error_msg = f"Qdrant upsert failed: {exc}"
        logger.exception(f"[{file_id}] {error_msg}")
        return False, error_msg
