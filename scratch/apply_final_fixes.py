import sys
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
json_path = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

blocks = data.get("text_blocks", [])
tables = data.get("tables", [])

# 1. Recover missing pages (81, 85, 92) using 150 DPI RapidOCR
covered_pages = set(b.get("page_num") for b in blocks if isinstance(b.get("page_num"), int))
missing_pages = [p for p in [81, 85, 92] if p not in covered_pages or sum(1 for b in blocks if b.get("page_num") == p) == 0]

doc = fitz.open(str(pdf_path))
ocr = RapidOCR()

if missing_pages:
    for p_num in missing_pages:
        page = doc[p_num - 1]
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        res, _ = ocr(img)
        if res:
            lines = [r[1] for r in res if len(r) > 1 and r[1].strip()]
            blocks.append({
                "block_id": f"blk_rec_p{p_num:04d}",
                "page_num": p_num,
                "type": "paragraph",
                "text": "\n".join(lines),
                "bbox": [0, 0, page.rect.width, page.rect.height]
            })
            print(f"Recovered missing page {p_num}: {len(lines)} lines")

# 2. Re-extract Chapter 3 (Pages 66-93) with 2-Column Column-by-Column Flow
blocks = [b for b in blocks if not (66 <= b.get("page_num", 0) <= 93)]

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
        blocks.append({"block_id": "blk_tmp", "page_num": p_num, "type": "header", "text": h, "bbox": [0, 0, pw, ph]})

    col1_texts = [t for y, t in col1]
    col2_texts = [t for y, t in col2]
    all_body_lines = col1_texts + col2_texts

    curr_para = []
    for line in all_body_lines:
        if any(hp.search(line) for hp in HEADING_PATTERNS):
            if curr_para:
                blocks.append({"block_id": "blk_tmp", "page_num": p_num, "type": "paragraph", "text": " ".join(curr_para), "bbox": [0, 0, pw, ph]})
                curr_para = []
            blocks.append({"block_id": "blk_tmp", "page_num": p_num, "type": "heading", "text": line, "bbox": [0, 0, pw, ph]})
        elif LIST_PATTERN.search(line):
            if curr_para:
                blocks.append({"block_id": "blk_tmp", "page_num": p_num, "type": "paragraph", "text": " ".join(curr_para), "bbox": [0, 0, pw, ph]})
                curr_para = []
            blocks.append({"block_id": "blk_tmp", "page_num": p_num, "type": "list_item", "text": line, "bbox": [0, 0, pw, ph]})
        else:
            curr_para.append(line)

    if curr_para:
        blocks.append({"block_id": "blk_tmp", "page_num": p_num, "type": "paragraph", "text": " ".join(curr_para), "bbox": [0, 0, pw, ph]})

    for f in footers:
        blocks.append({"block_id": "blk_tmp", "page_num": p_num, "type": "footer", "text": f, "bbox": [0, 0, pw, ph]})

doc.close()

# 3. Apply Postprocessor Pipeline (Cleaner, Deduplicator, Boilerplate, Corrector)
print("Running complete postprocessor pipeline...")
blocks = clean_extracted_blocks(blocks)
blocks = tag_boilerplate_blocks(blocks)
blocks = correct_extracted_blocks(blocks)

for idx, b in enumerate(blocks, start=1):
    b["block_id"] = f"blk_{idx:04d}"

for tbl in tables:
    headers = tbl.get("headers", [])
    if headers and all(str(h).isdigit() for h in headers):
        if len(headers) == 2:
            tbl["headers"] = ["Section / Unit", "Page Number"]
        else:
            tbl["headers"] = [f"Column_{int(h)+1}" for h in headers]

# Save output
out_data = {
    "text_blocks": blocks,
    "tables": tables,
    "page_images": data.get("page_images", []),
    "block_count": len(blocks),
    "table_count": len(tables)
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

output_dir = Path(r"outputs/story_fixed")
output_dir.mkdir(parents=True, exist_ok=True)
out_json_path1 = output_dir / "[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json"
with open(out_json_path1, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

print("SUCCESSFULLY SAVED FIXED JSON TO:", json_path)
