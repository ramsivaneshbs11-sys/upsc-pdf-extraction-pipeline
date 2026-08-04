"""
text_cleaner.py
────────────────
Cleans extracted JSON data prior to chunking.
Filters out boilerplate blocks (headers, footers, page numbers) and normalizes text.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger("text_cleaner")


def clean_extracted_json(json_path: Path) -> Dict[str, Any]:
    """
    Reads an extracted JSON file and extracts content blocks, filtering out boilerplate.

    Returns:
        Dict containing document metadata and list of non-boilerplate content blocks.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("document_metadata", {})
    blocks = data.get("text_blocks", [])

    clean_blocks = []
    for b in blocks:
        # Filter out boilerplate blocks
        if b.get("is_boilerplate", False):
            continue

        text = b.get("text", "").strip()
        if not text:
            continue

        clean_blocks.append({
            "block_id": b.get("block_id"),
            "page_num": b.get("page_num", 1),
            "type": b.get("type", "paragraph"),
            "text": text
        })

    logger.info(f"TextCleaner: Retained {len(clean_blocks)}/{len(blocks)} blocks for {json_path.name}")

    return {
        "metadata": meta,
        "blocks": clean_blocks
    }
