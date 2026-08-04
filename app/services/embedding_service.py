import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from app.core.config import EMBEDDING_MODEL_NAME
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def run_embedding(preprocessed_json_path: Path):
    if not preprocessed_json_path.exists():
        return False, None, f"Preprocessed JSON not found: {preprocessed_json_path}"
    try:
        with open(preprocessed_json_path, "r", encoding="utf-8") as f: data = json.load(f)
    except Exception as exc:
        return False, None, str(exc)
    chunks = data.get("chunks", [])
    if not chunks:
        return False, None, "No chunks to embed."
    try:
        model = _get_model()
        texts = [chunk["text"] for chunk in chunks]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:
        return False, None, str(exc)
    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append({"chunk_id": chunk["chunk_id"], "text": chunk["text"], "metadata": chunk.get("metadata", {}), "vector": vector.tolist()})
    return True, embedded_chunks, None
