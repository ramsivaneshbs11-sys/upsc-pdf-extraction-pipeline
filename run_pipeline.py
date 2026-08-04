import argparse
import sys
import os
import logging
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from extraction.main import process_single_pdf
from preprocessing.main import process_single_json
from extraction.config import DEFAULT_OUTPUT_DIR as DEFAULT_EXTRACT_OUT
from preprocessing.config import DEFAULT_OUTPUT_DIR as DEFAULT_PREPROCESS_OUT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")

def run_pipeline(pdf_folder: Path, extract_out_dir: Path, preprocess_out_dir: Path, chunk_size: int, chunk_overlap: int):
    pdf_files = sorted(pdf_folder.rglob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in: {pdf_folder}")
        sys.exit(1)

    for idx, pdf_file in enumerate(pdf_files, start=1):
        file_extract_dir = extract_out_dir / pdf_file.stem
        ok = process_single_pdf(pdf_file, file_extract_dir)
        if not ok: continue
        json_path = file_extract_dir / f"{pdf_file.stem}_extracted.json"
        process_single_json(json_path=json_path, output_dir=preprocess_out_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline")
    parser.add_argument("pdf_folder", type=str, help="Path to PDF folder")
    parser.add_argument("--extract-output-dir", type=str, default=str(DEFAULT_EXTRACT_OUT))
    parser.add_argument("--preprocess-output-dir", type=str, default=str(DEFAULT_PREPROCESS_OUT))
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()
    pdf_folder = Path(args.pdf_folder)
    if not pdf_folder.is_dir(): sys.exit(1)
    run_pipeline(pdf_folder, Path(args.extract_output_dir), Path(args.preprocess_output_dir), args.chunk_size, args.chunk_overlap)

if __name__ == "__main__":
    main()
