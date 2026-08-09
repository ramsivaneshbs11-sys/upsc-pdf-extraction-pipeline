import sys
import os
import json
import re
import fitz
import numpy as np
from datetime import datetime
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

# Load existing tables if present
existing_json_path = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")
tables = []
page_images = []
if existing_json_path.exists():
    with open(existing_json_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
        tables = existing_data.get("tables", [])
        page_images = existing_data.get("page_images", [])

doc = fitz.open(str(pdf_path))
ocr = RapidOCR()
total_pdf_pages = len(doc)

HEADER_REGEX = re.compile(r"social\s*science|the\s*heritage\s*of\s*india", re.IGNORECASE)
HEADING_PATTERNS = [
    re.compile(r"^\s*chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*exercises\s*$", re.IGNORECASE),
    re.compile(r"^\s*things\s+to\s+know\b", re.IGNORECASE),
    re.compile(r"^\s*india['’]?s\s+struggle\s+for\s+independence\b", re.IGNORECASE),
    re.compile(r"^\s*revolt\s+of\s+1857\b", re.IGNORECASE),
    re.compile(r"^\s*rise\s+of\s+indian\s+nationalism\b", re.IGNORECASE),
    re.compile(r"^\s*indian\s+national\s+congress\b", re.IGNORECASE),
    re.compile(r"^\s*boycott\s+and\s+swadeshi\b", re.IGNORECASE),
    re.compile(r"^\s*morley[- ]minto\b", re.IGNORECASE),
    re.compile(r"^\s*khilafat\b", re.IGNORECASE),
    re.compile(r"^\s*non[- ]cooperation\b", re.IGNORECASE),
    re.compile(r"^\s*civil\s+disobedience\b", re.IGNORECASE),
    re.compile(r"^\s*quit\s+india\b", re.IGNORECASE),
    re.compile(r"^\s*the\s+medieval\s+period\b", re.IGNORECASE),
    re.compile(r"^\s*languages?\s+and\s+literature\b", re.IGNORECASE),
    re.compile(r"^\s*architecture\b", re.IGNORECASE),
    re.compile(r"^\s*cave\s*architecture\b", re.IGNORECASE),
    re.compile(r"^\s*the\s+heritage\s+of\s+india\b", re.IGNORECASE),
]
LIST_PATTERN = re.compile(r"^\s*(\d+[\.\)]|[a-z][\.\)]|[•\-\➢])\s+", re.IGNORECASE)
FOOTER_NUM_PATTERN = re.compile(r"^\s*\d{1,3}\s*$")

def build_blocks_for_col(lines, p_num, pw, ph):
    """
    Groups OCR lines in a column into headings, list items, and natural paragraph blocks based on line spacing and indentation.
    """
    if not lines:
        return []
    
    col_blocks = []
    curr_lines = []
    
    for i, line_tuple in enumerate(lines):
        min_y, max_y, min_x, max_x, text = line_tuple
        
        # Check if line matches heading pattern
        is_head = any(hp.search(text) for hp in HEADING_PATTERNS)
        is_list = LIST_PATTERN.search(text)
        
        if is_head:
            if curr_lines:
                col_blocks.append({
                    "block_id": f"blk_p{p_num:04d}",
                    "page_num": p_num,
                    "type": "paragraph",
                    "text": " ".join([l[4] for l in curr_lines]),
                    "bbox": [0, 0, pw, ph]
                })
                curr_lines = []
            col_blocks.append({
                "block_id": f"blk_p{p_num:04d}",
                "page_num": p_num,
                "type": "heading",
                "text": text,
                "bbox": [0, 0, pw, ph]
            })
        elif is_list:
            if curr_lines:
                col_blocks.append({
                    "block_id": f"blk_p{p_num:04d}",
                    "page_num": p_num,
                    "type": "paragraph",
                    "text": " ".join([l[4] for l in curr_lines]),
                    "bbox": [0, 0, pw, ph]
                })
                curr_lines = []
            col_blocks.append({
                "block_id": f"blk_p{p_num:04d}",
                "page_num": p_num,
                "type": "list_item",
                "text": text,
                "bbox": [0, 0, pw, ph]
            })
        else:
            if curr_lines:
                prev_min_y, prev_max_y, prev_min_x, prev_max_x, prev_t = curr_lines[-1]
                line_h = prev_max_y - prev_min_y
                gap = min_y - prev_max_y
                
                # Split paragraph if:
                # 1. Large vertical gap (> 1.35 * line height)
                # 2. Significant left indentation (> 14px)
                # 3. Previous line ended with sentence terminal AND current line starts with capital letter AND gap > 0.5 * line_h
                is_terminal = bool(re.search(r'[.!?:"’]\s*$', prev_t))
                starts_capital = bool(re.match(r'^[A-Z"’]', text))
                
                if gap > 1.35 * max(line_h, 10.0) or min_x > prev_min_x + 14.0 or (is_terminal and starts_capital and gap > 0.5 * max(line_h, 10.0)):
                    col_blocks.append({
                        "block_id": f"blk_p{p_num:04d}",
                        "page_num": p_num,
                        "type": "paragraph",
                        "text": " ".join([l[4] for l in curr_lines]),
                        "bbox": [0, 0, pw, ph]
                    })
                    curr_lines = [line_tuple]
                else:
                    curr_lines.append(line_tuple)
            else:
                curr_lines.append(line_tuple)
                
    if curr_lines:
        col_blocks.append({
            "block_id": f"blk_p{p_num:04d}",
            "page_num": p_num,
            "type": "paragraph",
            "text": " ".join([l[4] for l in curr_lines]),
            "bbox": [0, 0, pw, ph]
        })
        
    return col_blocks

print(f"Executing 2-column layout OCR pass with paragraph splitting across all {total_pdf_pages} pages...")
extracted_blocks = []

for p_num in range(1, total_pdf_pages + 1):
    page = doc[p_num - 1]
    pix = page.get_pixmap(dpi=150)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    res, _ = ocr(img)
    if not res:
        print(f"Page {p_num}: No text detected by OCR.")
        continue

    pw, ph = pix.width, pix.height
    x_mid = pw / 2.0

    headers, footers, col1, col2 = [], [], [], []
    for item in res:
        box, text, score = item[0], item[1].strip(), item[2]
        if not text:
            continue
        min_x = min(b[0] for b in box)
        max_x = max(b[0] for b in box)
        min_y = min(b[1] for b in box)
        max_y = max(b[1] for b in box)
        xc = (min_x + max_x) / 2.0
        yc = (min_y + max_y) / 2.0

        if yc < ph * 0.12 and (HEADER_REGEX.search(text) or (FOOTER_NUM_PATTERN.match(text) and len(text) <= 3)):
            headers.append(text)
        elif yc > ph * 0.90 and (FOOTER_NUM_PATTERN.match(text) or yc > ph * 0.94):
            footers.append(text)
        elif xc < x_mid:
            col1.append((min_y, max_y, min_x, max_x, text))
        else:
            col2.append((min_y, max_y, min_x, max_x, text))

    col1.sort(key=lambda x: x[0])
    col2.sort(key=lambda x: x[0])

    for h in headers:
        extracted_blocks.append({
            "block_id": f"blk_p{p_num:04d}",
            "page_num": p_num,
            "type": "header",
            "text": h,
            "bbox": [0, 0, pw, ph]
        })

    # Page 7 is Table of Contents: suppress paragraph capture if table exists
    if p_num == 7 and tables:
        pass
    else:
        c1_blocks = build_blocks_for_col(col1, p_num, pw, ph)
        c2_blocks = build_blocks_for_col(col2, p_num, pw, ph)
        extracted_blocks.extend(c1_blocks)
        extracted_blocks.extend(c2_blocks)

    for f in footers:
        extracted_blocks.append({
            "block_id": f"blk_p{p_num:04d}",
            "page_num": p_num,
            "type": "footer",
            "text": f,
            "bbox": [0, 0, pw, ph]
        })

doc.close()
print(f"Extracted {len(extracted_blocks)} raw blocks across {total_pdf_pages} pages.")

# Post-processing Pipeline
print("Applying cleaner, deduplicator, boilerplate, corrector, NER...")
extracted_blocks = clean_extracted_blocks(extracted_blocks)
extracted_blocks = tag_boilerplate_blocks(extracted_blocks)
extracted_blocks = correct_extracted_blocks(extracted_blocks)
extracted_blocks = enrich_blocks_with_ner(extracted_blocks)

for idx, b in enumerate(extracted_blocks, start=1):
    b["block_id"] = f"blk_{idx:04d}"

for tbl in tables:
    headers_list = tbl.get("headers", [])
    if headers_list and all(str(h).isdigit() for h in headers_list):
        if len(headers_list) == 2:
            tbl["headers"] = ["Section / Unit", "Page Number"]
        else:
            tbl["headers"] = [f"Column_{int(h)+1}" for h in headers_list]

covered_pages_set = set(b.get("page_num") for b in extracted_blocks if isinstance(b.get("page_num"), int))

final_data = {
    "document_metadata": {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "extracted_at": datetime.now().isoformat() + "Z",
        "total_pages": total_pdf_pages,
        "covered_pages": len(covered_pages_set)
    },
    "text_blocks": extracted_blocks,
    "tables": tables,
    "page_images": page_images,
    "block_count": len(extracted_blocks),
    "table_count": len(tables)
}

out_json_path1 = output_dir / "[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json"
out_json_path2 = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")

with open(out_json_path1, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)
with open(out_json_path2, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"Successfully generated clean fixed JSON output with {len(extracted_blocks)} total blocks covering {len(covered_pages_set)}/{total_pdf_pages} pages.")
