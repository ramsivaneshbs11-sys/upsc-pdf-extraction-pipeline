"""
chunker.py
────────────
Layout-aware semantic text chunker.
Splits preprocessed text blocks into retrieval-optimized chunks for vector embeddings.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger("chunker")


def create_chunks(
    clean_data: Dict[str, Any],
    max_chunk_size: int = 1000,
    overlap: int = 200
) -> Dict[str, Any]:
    """
    Creates layout-aware text chunks from a cleaned document structure.

    Args:
        clean_data: Dict containing cleaned content blocks and metadata.
        max_chunk_size: Max characters per chunk (default 1000).
        overlap: Character overlap between consecutive chunks (default 200).

    Returns:
        Dict containing chunk list and summary metadata.
    """
    blocks = clean_data.get("blocks", [])
    doc_meta = clean_data.get("metadata", {})
    file_name = doc_meta.get("file_name", "unknown")

    chunks = []
    current_text = ""
    current_pages = set()
    current_block_ids = []
    chunk_counter = 1

    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue

        page_num = block.get("page_num", 1)
        block_id = block.get("block_id", "")

        # If adding block text exceeds max_chunk_size, commit current chunk
        if current_text and (len(current_text) + len(text) > max_chunk_size):
            chunks.append(_build_chunk_dict(
                chunk_id=f"chk_{chunk_counter:04d}",
                text=current_text.strip(),
                pages=sorted(list(current_pages)),
                block_ids=current_block_ids,
                file_name=file_name
            ))
            chunk_counter += 1

            # Prepare overlap text
            overlap_text = current_text[-overlap:] if len(current_text) > overlap else ""
            current_text = overlap_text + " " + text
            current_pages = {page_num}
            current_block_ids = [block_id]
        else:
            current_text = (current_text + " " + text).strip()
            current_pages.add(page_num)
            current_block_ids.append(block_id)

    # Commit final remaining chunk
    if current_text:
        chunks.append(_build_chunk_dict(
            chunk_id=f"chk_{chunk_counter:04d}",
            text=current_text.strip(),
            pages=sorted(list(current_pages)),
            block_ids=current_block_ids,
            file_name=file_name
        ))

    logger.info(f"Chunker: Created {len(chunks)} chunks for {file_name}")

    return {
        "metadata": doc_meta,
        "chunk_count": len(chunks),
        "chunks": chunks
    }


def _build_chunk_dict(
    chunk_id: str,
    text: str,
    pages: List[int],
    block_ids: List[str],
    file_name: str
) -> Dict[str, Any]:
    """Helper to build a chunk dict."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "character_count": len(text),
        "metadata": {
            "file_name": file_name,
            "page_numbers": pages,
            "source_block_ids": block_ids
        }
    }
