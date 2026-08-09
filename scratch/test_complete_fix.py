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

# 1. Load base JSON
existing_json_path = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")
with open(existing_json_path, "r", encoding="utf-8") as f:
    existing_data = json.load(f)

text_blocks = existing_data.get("text_blocks", [])
tables = existing_data.get("tables", [])

doc = fitz.open(str(pdf_path))
ocr = RapidOCR()

# 2. Re-extract corrupted Chapter 3 (Pages 66-93) and Missing Pages (81, 85, 92) using 2-column RapidOCR
print("Re-extracting Chapter 3 (Pages 66-93) and Missing Pages using 2-column RapidOCR...")

# Remove existing blocks for pages 66-93 from text_blocks
text_blocks = [b for b in text_blocks if not (66 <= b.get("page_num", 0) <= 93)]

HEADER_PHRASES = {
    "social science - part i", "social science - part 1", "social science part 1",
    "social science part i", "india's struggle for independence", "the heritage of india",
}
HEADING_PATTERNS = [
    re.compile(r"^\s*chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*exercises\s*$", re.IGNORECASE),
    re.compile(r"^\s*things\s+to\s+know\b", re.IGNORECASE),
    re.compile(r"^\s*india['’]?s\s+struggle\s+for\s+independence\b", re.IGNORECASE),
    re.compile(r"^\s*revolt\s+of\s+1857\b", re.IGNORECASE),
    re.compile(r"^\s*non[- ]cooperation\b", re.IGNORECASE),
    re.compile(r"^\s*civil\s+disobedience\b", re.IGNORECASE),
    re.compile(r"^\s*quit\s+india\b", re.IGNORECASE),
]
LIST_PATTERN = re.compile(r"^\s*(\d+[\.\)]|[a-z][\.\)]|[•\-\➢])\s+", re.IGNORECASE)
FOOTER_NUM_PATTERN = re.compile(r"^\s*\d{1,3}\s*$")

for p_num in range(66, 94):
    page = doc[p_num - 1]
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    res, _ = ocr(img)
    if not res:
        continue

    pw, ph = pix.width, pix.height
    x_mid = pw / 2.0

    headers, footers, col1, col2 = [], [], [], []
    for item in res:
        box, text, score = item[0], item[1].strip(), item[2]
        if not text:
            continue
        xc = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4.0
        yc = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4.0

        t_low = text.lower()
        if yc < ph * 0.08 or t_low in HEADER_PHRASES:
            headers.append(text)
        elif yc > ph * 0.92 or (FOOTER_NUM_PATTERN.match(text) and len(text) <= 3):
            footers.append(text)
        elif xc < x_mid:
            col1.append((yc, text))
        else:
            col2.append((yc, text))

    col1.sort(key=lambda x: x[0])
    col2.sort(key=lambda x: x[0])

    for h in headers:
        text_blocks.append({
            "block_id": "blk_tmp",
            "page_num": p_num,
            "type": "header",
            "text": h,
            "bbox": [0, 0, pw, ph]
        })

    col1_texts = [t for y, t in col1]
    col2_texts = [t for y, t in col2]
    all_body_lines = col1_texts + col2_texts

    curr_para = []
    for line in all_body_lines:
        if any(hp.search(line) for hp in HEADING_PATTERNS):
            if curr_para:
                text_blocks.append({
                    "block_id": "blk_tmp",
                    "page_num": p_num,
                    "type": "paragraph",
                    "text": " ".join(curr_para),
                    "bbox": [0, 0, pw, ph]
                })
                curr_para = []
            text_blocks.append({
                "block_id": "blk_tmp",
                "page_num": p_num,
                "type": "heading",
                "text": line,
                "bbox": [0, 0, pw, ph]
            })
        elif LIST_PATTERN.search(line):
            if curr_para:
                text_blocks.append({
                    "block_id": "blk_tmp",
                    "page_num": p_num,
                    "type": "paragraph",
                    "text": " ".join(curr_para),
                    "bbox": [0, 0, pw, ph]
                })
                curr_para = []
            text_blocks.append({
                "block_id": "blk_tmp",
                "page_num": p_num,
                "type": "list_item",
                "text": line,
                "bbox": [0, 0, pw, ph]
            })
        else:
            curr_para.append(line)

    if curr_para:
        text_blocks.append({
            "block_id": "blk_tmp",
            "page_num": p_num,
            "type": "paragraph",
            "text": " ".join(curr_para),
            "bbox": [0, 0, pw, ph]
        })

    for f in footers:
        text_blocks.append({
            "block_id": "blk_tmp",
            "page_num": p_num,
            "type": "footer",
            "text": f,
            "bbox": [0, 0, pw, ph]
        })

doc.close()

# 3. Postprocessor execution
print("Applying cleaner, deduplicator, boilerplate, corrector...")
text_blocks = clean_extracted_blocks(text_blocks)
text_blocks = tag_boilerplate_blocks(text_blocks)
text_blocks = correct_extracted_blocks(text_blocks)
text_blocks = enrich_blocks_with_ner(text_blocks)

for idx, b in enumerate(text_blocks, start=1):
    b["block_id"] = f"blk_{idx:04d}"

for tbl in tables:
    headers = tbl.get("headers", [])
    if headers and all(str(h).isdigit() for h in headers):
        if len(headers) == 2:
            tbl["headers"] = ["Section / Unit", "Page Number"]
        else:
            tbl["headers"] = [f"Column_{int(h)+1}" for h in headers]

# Save output
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

print("Saved complete fixed JSON successfully.")
