"""
smart_router.py
────────────────
Smart PDF extraction router.

Automatically detects whether an input PDF is digital or scanned,
then dispatches to the correct extraction engine:

    Digital PDF  →  extract_document()       (local Docling + Hybrid Fitz pipeline)
    Scanned PDF  →  extract_with_gemini_flash()  (Gemini 2.5 Flash Vision VLM)

══════════════════════════════════════════════════════════════════════
ZERO CHANGES TO EXISTING CODE:
  This file is the ONLY new entry point. All existing extraction/*.py
  files remain completely untouched. Drop-in replacement for callers
  that previously called extract_document() directly.
══════════════════════════════════════════════════════════════════════

Usage (replacing direct extract_document calls):

    # Before (direct):
    from extraction.docling_extractor import extract_document
    doc, data = extract_document(pdf_path, output_dir)

    # After (smart router — handles both digital AND scanned):
    from extraction.smart_router import route_extraction
    doc, data = route_extraction(pdf_path, output_dir)

The return value is IDENTICAL — (doc, extracted_data_dict).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("smart_router")


def route_extraction(
    pdf_path: Path,
    output_dir: Path,
    converter: Optional[Any] = None,
    force_engine: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Smart routing entry point — auto-detects PDF type and calls the correct engine.

    Args:
        pdf_path:      Path to the input PDF file.
        output_dir:    Directory to save extraction output.
        converter:     Optional pre-built Docling converter (digital path only).
        force_engine:  Override auto-detection. Pass "docling" or "gemini" to force.

    Returns:
        (doc_or_none, extracted_data_dict) — identical schema to extract_document().

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        RuntimeError: If Gemini engine is selected but API key is missing.
    """
    pdf_path   = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── Step 1: Determine which engine to use ─────────────────────────────
    if force_engine:
        use_gemini = force_engine.lower() == "gemini"
        logger.info(f"[SmartRouter] Engine FORCED to: {'gemini' if use_gemini else 'docling'}")
    else:
        from extraction.pdf_type_detector import is_scanned_pdf
        use_gemini = is_scanned_pdf(pdf_path)
        engine_name = "Gemini 2.5 Flash (Scanned)" if use_gemini else "Docling Hybrid (Digital)"
        logger.info(f"[SmartRouter] Auto-detected engine: {engine_name} → {pdf_path.name}")

    # ── Step 2: Dispatch to the selected engine ───────────────────────────
    if use_gemini:
        _log_gemini_header(pdf_path)
        from extraction.gemini_extractor import extract_with_gemini_flash
        return extract_with_gemini_flash(pdf_path, output_dir)

    else:
        _log_docling_header(pdf_path)
        from extraction.docling_extractor import extract_document
        return extract_document(pdf_path, output_dir, converter=converter)


# ─────────────────────────────────────────────────────────────────────────────
# Console banners
# ─────────────────────────────────────────────────────────────────────────────

def _log_gemini_header(pdf_path: Path) -> None:
    logger.info("=" * 60)
    logger.info("  EXTRACTION ENGINE : Gemini 2.5 Flash (Cloud VLM)")
    logger.info(f"  PDF               : {pdf_path.name}")
    logger.info("  REASON            : Scanned / image-only PDF detected")
    logger.info("=" * 60)


def _log_docling_header(pdf_path: Path) -> None:
    logger.info("=" * 60)
    logger.info("  EXTRACTION ENGINE : Docling + PyMuPDF Hybrid (Local)")
    logger.info(f"  PDF               : {pdf_path.name}")
    logger.info("  REASON            : Digital text layer detected")
    logger.info("=" * 60)
