import fitz
import json
import logging
from pathlib import Path
from typing import Dict, Any
from extraction.config import MAX_FILE_SIZE_MB

logger = logging.getLogger("document_validator")

def validate_pdf(pdf_path: Path, max_size_mb: float = MAX_FILE_SIZE_MB) -> Dict[str, Any]:
    report = {"is_valid": True, "file_name": pdf_path.name, "file_size_mb": 0.0, "page_count": 0, "errors": []}
    if not pdf_path.exists() or not pdf_path.is_file():
        report["is_valid"] = False; report["errors"].append(f"File not found: {pdf_path}"); return report
    file_size_mb = round(pdf_path.stat().st_size / (1024 * 1024), 2)
    report["file_size_mb"] = file_size_mb
    if file_size_mb == 0: report["is_valid"] = False; report["errors"].append("File is empty (0 bytes)"); return report
    if pdf_path.suffix.lower() != ".pdf": report["is_valid"] = False; report["errors"].append("Expected '.pdf'")
    if file_size_mb > max_size_mb: report["is_valid"] = False; report["errors"].append(f"Exceeds max {max_size_mb} MB")
    try:
        with open(pdf_path, "rb") as f:
            if f.read(5) != b"%PDF-": report["is_valid"] = False; report["errors"].append("Invalid PDF magic bytes")
    except Exception as e: report["is_valid"] = False; report["errors"].append(str(e)); return report
    try:
        doc = fitz.open(str(pdf_path))
        report["page_count"] = len(doc)
        doc.close()
        if report["page_count"] < 1: report["is_valid"] = False; report["errors"].append("0 pages")
    except Exception as e: report["is_valid"] = False; report["errors"].append(str(e))
    return report

def audit_extraction(json_path: Path) -> Dict[str, Any]:
    audit = {f"rule_{i}": False for i in range(1, 10)}
    if not json_path.exists(): return audit
    try:
        with open(json_path, "r", encoding="utf-8") as f: data = json.load(f)
        audit["rule_1"] = True
    except Exception: return audit
    meta = data.get("document_metadata", {})
    if meta.get("file_name") and meta.get("extracted_at"): audit["rule_2"] = True
    blocks = data.get("text_blocks", [])
    if isinstance(blocks, list) and len(blocks) > 0: audit["rule_3"] = True
    if not blocks: return audit
    audit["rule_4"] = all(isinstance(b.get("page_num"), int) and b.get("page_num") >= 1 for b in blocks)
    valid_types = {"heading", "paragraph", "list_item", "caption", "footnote", "header", "footer", "table"}
    audit["rule_5"] = all(b.get("type") in valid_types for b in blocks)
    audit["rule_6"] = all(isinstance(b.get("text"), str) and len(b.get("text").strip()) > 0 for b in blocks)
    audit["rule_7"] = all("is_boilerplate" in b for b in blocks)
    audit["rule_8"] = all(isinstance(tbl.get("table_id"), str) and isinstance(tbl.get("rows"), list) for tbl in data.get("tables", []))
    audit["rule_9"] = all("entities" in b for b in blocks)
    return audit
