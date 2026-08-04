import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger("text_cleaner")

def clean_extracted_json(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f: data = json.load(f)
    meta = data.get("document_metadata", {})
    blocks = data.get("text_blocks", [])
    clean_blocks = []
    for b in blocks:
        if b.get("is_boilerplate", False): continue
        text = b.get("text", "").strip()
        if not text: continue
        clean_blocks.append({"block_id": b.get("block_id"), "page_num": b.get("page_num", 1), "type": b.get("type", "paragraph"), "text": text})
    logger.info(f"TextCleaner: Retained {len(clean_blocks)}/{len(blocks)} blocks for {json_path.name}")
    return {"metadata": meta, "blocks": clean_blocks}
