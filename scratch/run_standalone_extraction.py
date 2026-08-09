import sys
import os
import json
import re
import fitz
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from rapidocr_onnxruntime import RapidOCR
from extraction.block_cleaner import clean_extracted_blocks
from extraction.boilerplate_detector import tag_boilerplate_blocks
from extraction.content_corrector import correct_extracted_blocks
from extraction.ner_extractor import enrich_blocks_with_ner

pdf_path = Path(r"inputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org.pdf")
output_dir = Path(r"outputs/story_fixed")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Load existing extracted JSON if present
existing_json_path = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")
with open(existing_json_path, "r", encoding="utf-8") as f:
    existing_data = json.load(f)

text_blocks = existing_data.get("text_blocks", [])
tables = existing_data.get("tables", [])

print(f"Loaded {len(text_blocks)} initial blocks, {len(tables)} tables.")

# 2. Target recovery for missing pages 81, 85, 92 using RapidOCR at 150 DPI
doc = fitz.open(str(pdf_path))
ocr = RapidOCR()

for p_num in [81, 85, 92]:
    page = doc[p_num - 1]
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    res, _ = ocr(img)
    if res:
        lines = [r[1] for r in res if len(r) > 1 and r[1].strip()]
        page_text = "\n".join(lines)
        text_blocks.append({
            "block_id": f"blk_rec_p{p_num:04d}",
            "page_num": p_num,
            "type": "paragraph",
            "text": page_text,
            "bbox": [0.0, 0.0, page.rect.width, page.rect.height]
        })
        print(f"Recovered page {p_num}: {len(lines)} OCR lines extracted.")

doc.close()

# 3. Post-processing Pipeline
print("Applying postprocessing pipeline (cleaner, deduplicator, splitter, boilerplate, corrector)...")

# 3a. Block Cleaner & Deduplication & Collapsed Page Splitting
text_blocks = clean_extracted_blocks(text_blocks)

# 3b. Tag Boilerplate
text_blocks = tag_boilerplate_blocks(text_blocks)

# 3c. Text Correction
text_blocks = correct_extracted_blocks(text_blocks)

# 3d. NER Enrichment
text_blocks = enrich_blocks_with_ner(text_blocks)

# 3e. Re-index block_ids contiguously
for idx, b in enumerate(text_blocks, start=1):
    b["block_id"] = f"blk_{idx:04d}"

# 3f. Table header cleanup
for tbl in tables:
    headers = tbl.get("headers", [])
    if headers and all(str(h).isdigit() for h in headers):
        if len(headers) == 2:
            tbl["headers"] = ["Section / Unit", "Page Number"]
        else:
            tbl["headers"] = [f"Column_{int(h)+1}" for h in headers]

# 4. Save Final Fixed JSON Output in both output dirs
out_json_path1 = output_dir / "[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json"
out_json_path2 = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")

final_data = {
    "text_blocks": text_blocks,
    "tables": tables,
    "page_images": existing_data.get("page_images", []),
    "block_count": len(text_blocks),
    "table_count": len(tables)
}

with open(out_json_path1, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

with open(out_json_path2, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"Successfully saved fixed JSON to:\n  - {out_json_path1}\n  - {out_json_path2}")
