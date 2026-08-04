"""
main.py
────────
Standalone CLI entry point for testing document extraction.

Usage:
  python main.py "C:\\path\\to\\document.pdf"
  python main.py "C:\\path\\to\\document.pdf" --output-dir "C:\\path\\to\\outputs"
"""

import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to sys.path so package resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.document_validator import validate_pdf, audit_extraction
from extraction.docling_extractor import extract_document
from extraction.json_builder import build_structured_json
from extraction.config import DEFAULT_OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("extraction_cli")


def process_single_pdf(pdf_path: Path, output_dir: Path) -> bool:
    """
    Executes the full pipeline for a single PDF document.
    """
    print("=" * 60)
    print(f"  PROCESSING: {pdf_path.name}")
    print("=" * 60)

    # 1. Validation
    logger.info("Step 1: Validating PDF file...")
    val_report = validate_pdf(pdf_path)
    if not val_report["is_valid"]:
        logger.error(f"Validation FAILED: {val_report['errors']}")
        return False
    logger.info(f"Validation PASSED (Pages: {val_report['page_count']}, Size: {val_report['file_size_mb']} MB)")

    # 2. Extraction & Post-processing
    logger.info("Step 2: Running Docling extraction + Postprocessing...")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc, extracted_data = extract_document(pdf_path, output_dir)

    # 3. Save JSON Output
    logger.info("Step 3: Building and saving structured JSON...")
    json_name = f"{pdf_path.stem}_extracted.json"
    json_out_path = output_dir / json_name
    build_structured_json(extracted_data, pdf_path, val_report, json_out_path)

    # 4. QA Audit
    logger.info("Step 4: Running QA Audit on extracted JSON...")
    audit_results = audit_extraction(json_out_path)
    passed_rules = sum(1 for v in audit_results.values() if v is True)

    print("\n" + "=" * 60)
    print(f"  EXTRACTION COMPLETE: {json_out_path.name}")
    print(f"  QA Audit Score: {passed_rules}/9 rules passed")
    print("=" * 60 + "\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="UPSC PDF Document Extractor")
    parser.add_argument("pdf_path", type=str, help="Path to input PDF file")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")

    args = parser.parse_args()
    pdf_path = Path(args.pdf_path)
    output_dir = Path(args.output_dir)

    if not pdf_path.exists():
        logger.error(f"Input PDF does not exist: {pdf_path}")
        sys.exit(1)

    success = process_single_pdf(pdf_path, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
