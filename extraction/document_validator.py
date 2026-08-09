"""
document_validator.py
──────────────────────
Validates input PDF documents before processing and performs QA audits on
extracted JSON outputs.

Rules Checked:
  PDF Validation:
    1. File exists and is a readable file
    2. Non-zero byte size
    3. File extension is .pdf
    4. Valid PDF magic header bytes (%PDF-)
    5. PyMuPDF page count check (>= 1 page)
    6. File size under maximum threshold (default: 100 MB)

  JSON Extraction QA Audit:
    1. JSON structure contains required keys
    2. Document metadata present
    3. Non-empty text blocks list
    4. All blocks have valid page numbers
    5. Valid block types
    6. Non-empty block text strings
    7. Boilerplate tags populated
    8. Table structure validity
    9. Entity enrichment populated
"""

import fitz  # PyMuPDF
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from extraction.config import MAX_FILE_SIZE_MB

logger = logging.getLogger("document_validator")


# ── 1. PDF INPUT VALIDATION ───────────────────────────────────────────────────

def validate_pdf(pdf_path: Path, max_size_mb: float = MAX_FILE_SIZE_MB) -> Dict[str, Any]:
    """
    Validates PDF file integrity before sending to Docling extraction.

    Returns dict:
      {
        "is_valid": bool,
        "file_name": str,
        "file_size_mb": float,
        "page_count": int,
        "errors": list[str]
      }
    """
    report = {
        "is_valid": True,
        "file_name": pdf_path.name,
        "file_size_mb": 0.0,
        "page_count": 0,
        "errors": []
    }

    # Rule 1: Existence
    if not pdf_path.exists() or not pdf_path.is_file():
        report["is_valid"] = False
        report["errors"].append(f"File not found: {pdf_path}")
        return report

    # Rule 2: Non-zero size
    file_size_bytes = pdf_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    report["file_size_mb"] = file_size_mb

    if file_size_bytes == 0:
        report["is_valid"] = False
        report["errors"].append("File is empty (0 bytes)")
        return report

    # Rule 3: File Extension
    if pdf_path.suffix.lower() != ".pdf":
        report["is_valid"] = False
        report["errors"].append(f"Invalid extension '{pdf_path.suffix}'. Expected '.pdf'")

    # Rule 4: Maximum size threshold
    if file_size_mb > max_size_mb:
        report["is_valid"] = False
        report["errors"].append(f"File size ({file_size_mb} MB) exceeds maximum allowed ({max_size_mb} MB)")

    # Rule 5: PDF Magic Header Check (%PDF-)
    try:
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                report["is_valid"] = False
                report["errors"].append("Invalid PDF magic bytes header (not a valid PDF document)")
    except Exception as e:
        report["is_valid"] = False
        report["errors"].append(f"Cannot read file header: {e}")
        return report

    # Rule 6: PyMuPDF Open & Page Count Test
    try:
        doc = fitz.open(str(pdf_path))
        page_count = len(doc)
        report["page_count"] = page_count
        doc.close()

        if page_count < 1:
            report["is_valid"] = False
            report["errors"].append("PDF document contains 0 pages")
    except Exception as e:
        report["is_valid"] = False
        report["errors"].append(f"PyMuPDF failed to open PDF: {e}")

    return report


# ── 2. EXTRACTION QA AUDIT ────────────────────────────────────────────────────

def audit_extraction(json_path: Path) -> Dict[str, Any]:
    """
    Performs a 9-rule Quality Assurance (QA) audit on an extracted JSON file.

    Returns dict of test rule results:
      {
        "rule_1_valid_json": bool,
        "rule_2_metadata_present": bool,
        "rule_3_non_empty_blocks": bool,
        "rule_4_page_numbers_valid": bool,
        "rule_5_block_types_valid": bool,
        "rule_6_non_empty_text": bool,
        "rule_7_boilerplate_tagged": bool,
        "rule_8_table_structure_valid": bool,
        "rule_9_ner_populated": bool
      }
    """
    audit = {
        "rule_1_valid_json": False,
        "rule_2_metadata_present": False,
        "rule_3_non_empty_blocks": False,
        "rule_4_page_numbers_valid": False,
        "rule_5_block_types_valid": False,
        "rule_6_non_empty_text": False,
        "rule_7_boilerplate_tagged": False,
        "rule_8_table_structure_valid": False,
        "rule_9_ner_populated": False,
    }

    if not json_path.exists():
        logger.error(f"QA Audit: JSON file not found: {json_path}")
        return audit

    # Rule 1: Valid JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        audit["rule_1_valid_json"] = True
    except Exception as e:
        logger.error(f"QA Audit Rule 1 Failed: Invalid JSON: {e}")
        return audit

    # Rule 2: Metadata Present
    meta = data.get("document_metadata", {})
    if meta.get("file_name") and meta.get("extracted_at"):
        audit["rule_2_metadata_present"] = True

    # Rule 3: Non-empty blocks
    blocks = data.get("text_blocks", [])
    if isinstance(blocks, list) and len(blocks) > 0:
        audit["rule_3_non_empty_blocks"] = True

    if not blocks:
        return audit

    # Rule 4: Page numbers valid (>= 1)
    audit["rule_4_page_numbers_valid"] = all(
        isinstance(b.get("page_num"), int) and b.get("page_num") >= 1 for b in blocks
    )

    # Rule 5: Block types valid
    # Includes pipeline-generated structural types:
    #   blank_page        — emitted by block_cleaner for cover/boilerplate-only pages
    #   exercise_heading  — tagged by boilerplate_detector for "Check Your Progress" sections
    #   toc_page_number   — tagged by boilerplate_detector for ToC page-number tokens
    valid_types = {
        "heading", "paragraph", "list_item", "caption", "footnote",
        "header", "footer", "table",
        "blank_page", "exercise_heading", "toc_page_number"
    }
    audit["rule_5_block_types_valid"] = all(
        b.get("type") in valid_types for b in blocks
    )

    # Rule 6: Non-empty block text
    # blank_page markers intentionally have empty text — exempt them from this check.
    audit["rule_6_non_empty_text"] = all(
        b.get("type") == "blank_page" or
        (isinstance(b.get("text"), str) and len(b.get("text").strip()) > 0)
        for b in blocks
    )

    # Rule 7: Boilerplate tagged
    audit["rule_7_boilerplate_tagged"] = all(
        "is_boilerplate" in b for b in blocks
    )

    # Rule 8: Table structure valid
    tables = data.get("tables", [])
    table_ok = True
    for tbl in tables:
        if not (isinstance(tbl.get("table_id"), str) and isinstance(tbl.get("rows"), list)):
            table_ok = False
            break
    audit["rule_8_table_structure_valid"] = table_ok

    # Rule 9: NER populated
    audit["rule_9_ner_populated"] = all(
        "entities" in b for b in blocks
    )

    passed_count = sum(1 for k, v in audit.items() if v is True)
    total_rules = len(audit)
    logger.info(f"QA Audit Result for {json_path.name}: Passed {passed_count}/{total_rules} rules")

    return audit
