import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extraction.document_validator import validate_pdf, audit_extraction
from extraction.docling_extractor import extract_document
from extraction.json_builder import build_structured_json
from extraction.config import DEFAULT_OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("extraction_cli")

def process_single_pdf(pdf_path: Path, output_dir: Path) -> bool:
    val_report = validate_pdf(pdf_path)
    if not val_report["is_valid"]:
        logger.error(f"Validation FAILED: {val_report['errors']}")
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    doc, extracted_data = extract_document(pdf_path, output_dir)
    json_name = f"{pdf_path.stem}_extracted.json"
    json_out_path = output_dir / json_name
    build_structured_json(extracted_data, pdf_path, val_report, json_out_path)
    audit_results = audit_extraction(json_out_path)
    return True

def main():
    parser = argparse.ArgumentParser(description="UPSC PDF Document Extractor")
    parser.add_argument("pdf_path", type=str, help="Path to input PDF file")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists(): sys.exit(1)
    success = process_single_pdf(pdf_path, Path(args.output_dir))
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
