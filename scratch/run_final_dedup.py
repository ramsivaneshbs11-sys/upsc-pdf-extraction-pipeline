import sys
import json
import re
import fitz
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rapidocr_onnxruntime import RapidOCR
from extraction.block_cleaner import clean_extracted_blocks, _deduplicate_pages
from extraction.boilerplate_detector import tag_boilerplate_blocks
from extraction.content_corrector import correct_extracted_blocks
from extraction.ner_extractor import enrich_blocks_with_ner

json_path1 = Path(r"outputs/story_fixed/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")
json_path2 = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")

for p in [json_path1, json_path2]:
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = data.get("text_blocks", [])
    tables = data.get("tables", [])

    # Force page deduplication pass on final block list
    blocks = _deduplicate_pages(blocks)

    # Re-index block IDs contiguously
    for idx, b in enumerate(blocks, start=1):
        b["block_id"] = f"blk_{idx:04d}"

    out_data = {
        "text_blocks": blocks,
        "tables": tables,
        "page_images": data.get("page_images", []),
        "block_count": len(blocks),
        "table_count": len(tables)
    }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f"Applied final page deduplication to {p}. Final block count: {len(blocks)}")
