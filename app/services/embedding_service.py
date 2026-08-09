"""
app/services/embedding_service.py
──────────────────────────────────
Loads BAAI/bge-base-en-v1.5 once as a module-level singleton and
embeds all chunks from a preprocessed JSON file.

Returns:
    (success: bool, embedded_chunks: list[dict] | None, error_message: str | None)

Each embedded_chunk dict contains:
    {
        "chunk_id":    str,
        "text":        str,
        "metadata":    dict,   # original metadata from the preprocessed JSON
        "vector":      list[float],  # 768-dim embedding
    }
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Load model once at import time ─────────────────────────────────────────
# Importing here avoids reloading the model on every API request.
_model = None


def _get_model():
    """Lazy-load the SentenceTransformer model (once per process)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from app.core.config import EMBEDDING_MODEL_NAME

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} …")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded ✓")
    return _model


def run_embedding(
    preprocessed_json_path: Path,
) -> tuple[bool, list[dict] | None, str | None]:
    """
    Embed all chunks from a preprocessed JSON file.

    Args:
        preprocessed_json_path: Path to the *_preprocessed.json file.

    Returns:
        (success, embedded_chunks, error_message)
    """
    # ── Load the preprocessed JSON ─────────────────────────────────────────
    if not preprocessed_json_path.exists():
        error_msg = f"Preprocessed JSON not found: {preprocessed_json_path}"
        logger.error(error_msg)
        return False, None, error_msg

    try:
        with open(preprocessed_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        error_msg = f"Failed to read preprocessed JSON: {exc}"
        logger.error(error_msg)
        return False, None, error_msg

    chunks: list[dict] = data.get("chunks", [])
    if not chunks:
        error_msg = "Preprocessed JSON contains no chunks — nothing to embed."
        logger.error(error_msg)
        return False, None, error_msg

    logger.info(f"Embedding {len(chunks)} chunks from {preprocessed_json_path.name} …")

    # ── Embed all chunk texts in one batched call ───────────────────────────
    try:
        model = _get_model()
        texts = [chunk["text"] for chunk in chunks]
        # normalize_embeddings=True is recommended for BGE models (cosine similarity)
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:
        error_msg = f"Embedding failed: {exc}"
        logger.exception(error_msg)
        return False, None, error_msg

    # ── Build embedded chunk list ───────────────────────────────────────────
    embedded_chunks: list[dict] = []
    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {}),
                "vector": vector.tolist(),
            }
        )

    logger.info(f"Embedding complete ✓ — {len(embedded_chunks)} vectors generated")
    return True, embedded_chunks, None
