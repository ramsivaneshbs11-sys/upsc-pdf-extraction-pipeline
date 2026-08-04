import logging
import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from extraction.document_validator import validate_pdf, audit_extraction
from extraction.docling_extractor import extract_document
from extraction.json_builder import build_structured_json
from app.core.config import EXTRACTED_DIR

logger = logging.getLogger(__name__)

def run_extraction(file_id: str, pdf_path: Path):
    val_report = validate_pdf(pdf_path)
    if not val_report.get("is_valid", False):
        return False, None, f"PDF Validation failed: {'; '.join(val_report.get('errors', []))}"
    doc_out_dir = EXTRACTED_DIR / file_id
    doc_out_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_doc, extracted_data = extract_document(pdf_path, doc_out_dir)
    except Exception as exc:
        return False, None, f"Docling extraction failed: {exc}"
    if not extracted_data:
        return False, None, "Docling extraction returned empty data"
    json_path = EXTRACTED_DIR / f"{file_id}.json"
    try:
        json_path = build_structured_json(extracted_data=extracted_data, pdf_path=pdf_path, validation_report=val_report, output_json_path=json_path)
    except Exception as exc:
        return False, None, f"Failed to build structured JSON: {exc}"
    audit_extraction(json_path)
    return True, json_path, None
