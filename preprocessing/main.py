import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.text_cleaner import clean_extracted_json
from preprocessing.chunker import create_chunks
from preprocessing.config import DEFAULT_OUTPUT_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocessing_cli")

def process_single_json(json_path: Path, output_dir: Path, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> bool:
    logger.info(f"Preprocessing JSON file: {json_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_data = clean_extracted_json(json_path)
    result = create_chunks(clean_data, max_chunk_size=chunk_size, overlap=chunk_overlap)
    out_path = output_dir / f"{json_path.stem}_preprocessed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved preprocessed chunks ({result['chunk_count']} chunks) -> {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="UPSC Preprocessor & Chunker")
    parser.add_argument("json_path", type=str, help="Path to extracted JSON file")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()
    json_path = Path(args.json_path)
    if not json_path.exists(): sys.exit(1)
    process_single_json(json_path, Path(args.output_dir), args.chunk_size, args.chunk_overlap)

if __name__ == "__main__":
    main()
