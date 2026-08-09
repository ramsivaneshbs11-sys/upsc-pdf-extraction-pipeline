import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from extraction.docling_extractor import _fill_missing_pages_via_fitz
from extraction.extraction_validator import audit_extraction_coverage_and_quality

pdf_path = ROOT_DIR / "inputs" / "[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org.pdf"
json_path = ROOT_DIR / "outputs" / "[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

blocks = data.get("text_blocks", [])

# Run updated _fill_missing_pages_via_fitz
updated_blocks = _fill_missing_pages_via_fitz(pdf_path, blocks)
data["text_blocks"] = updated_blocks

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

report = audit_extraction_coverage_and_quality(json_path, 107)
print("UPDATED EXTRACTION REPORT:")
print(json.dumps(report, indent=2))
