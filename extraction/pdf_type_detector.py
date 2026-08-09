"""
pdf_type_detector.py
─────────────────────
Instant pre-extraction heuristic check using PyMuPDF (fitz) to auto-detect
whether a PDF is SCANNED (requires OCR) or DIGITAL (native text).
"""

import fitz  # PyMuPDF
import logging
from pathlib import Path

logger = logging.getLogger("pdf_type_detector")


def is_scanned_pdf(pdf_path: Path, sample_size: int = 8, char_threshold: float = 150.0) -> bool:
    """
    Heuristic scanned-vs-digital check.
    Samples pages spread across the WHOLE document (not just front matter)
    so cover/title/foreword pages don't skew the result.

    Note:
        Digital PDFs have 1,000 - 3,000+ chars/page.
        Scanned PDFs with watermark footers (e.g. 'freeupscmaterials.org')
        only have ~20 - 80 chars/page of watermark text.
        A threshold of 150.0 ensures watermarked scanned PDFs trigger OCR correctly.

    Args:
        pdf_path: Path to the input PDF document.
        sample_size: Number of pages to sample across the document.
        char_threshold: Average character count threshold below which a page is considered scanned.

    Returns:
        True if the PDF is determined to be SCANNED (needs OCR), False if DIGITAL.
    """
    try:
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        if n == 0:
            doc.close()
            return False

        sample_size = min(sample_size, n)
        step = max(1, n // sample_size)
        indices = list(range(step // 2, n, step))[:sample_size]

        total_chars = 0
        for i in indices:
            text = doc[i].get_text().strip()
            total_chars += len(text)
        doc.close()

        avg_chars = total_chars / len(indices) if indices else 0
        is_scanned = avg_chars < char_threshold

        logger.info(
            f"PDF Type Check for {pdf_path.name}: Sampled {len(indices)} pages "
            f"(indices: {indices}), avg_chars={avg_chars:.1f} -> {'SCANNED (OCR=ON)' if is_scanned else 'DIGITAL (OCR=OFF)'}"
        )
        return is_scanned
    except Exception as e:
        logger.warning(f"PDF type detection failed for {pdf_path.name}: {e}. Defaulting to SCANNED (do_ocr=True)")
        return True
