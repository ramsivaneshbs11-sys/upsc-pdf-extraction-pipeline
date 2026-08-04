import logging
from pathlib import Path

logger = logging.getLogger("pdf_type_detector")

def is_scanned_pdf(pdf_path: Path, sample_size: int = 8, threshold: int = 20) -> bool:
    try:
        import fitz
    except ImportError:
        logger.warning("[pdf_type_detector] PyMuPDF not installed. Defaulting to do_ocr=False.")
        return False
    try:
        doc = fitz.open(str(pdf_path))
        n = len(doc)
        if n == 0:
            doc.close()
            return False
        sample_size = min(sample_size, n)
        step = max(1, n // sample_size)
        indices = list(range(step // 2, n, step))[:sample_size]
        total_chars = sum(len(doc[i].get_text().strip()) for i in indices)
        doc.close()
        avg_chars = total_chars / len(indices)
        is_scanned = avg_chars < threshold
        logger.info(
            f"[pdf_type_detector] {pdf_path.name}: sampled pages={indices}, "
            f"avg_chars={avg_chars:.1f}, threshold={threshold} -> "
            f"{'SCANNED' if is_scanned else 'DIGITAL'}"
        )
        return is_scanned
    except Exception as e:
        logger.error(f"[pdf_type_detector] Failed to analyse {pdf_path.name}: {e}")
        return False
