"""
benchmark.py
──────────────
Benchmark script to measure text chunking performance and output quality.
"""

import time
import json
import logging
from pathlib import Path

from preprocessing.chunker import create_chunks
from preprocessing.text_cleaner import clean_extracted_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def run_benchmark(sample_json_path: Path):
    """
    Measures execution time and chunk quality metrics for a sample extracted JSON file.
    """
    if not sample_json_path.exists():
        logger.error(f"Sample file not found: {sample_json_path}")
        return

    logger.info(f"Benchmarking file: {sample_json_path.name}")

    start_time = time.time()

    # 1. Cleaning
    clean_data = clean_extracted_json(sample_json_path)
    clean_time = time.time() - start_time

    # 2. Chunking
    chunk_start = time.time()
    result = create_chunks(clean_data, max_chunk_size=1000, overlap=200)
    chunk_time = time.time() - chunk_start

    total_time = time.time() - start_time

    chunks = result.get("chunks", [])
    logger.info("=== BENCHMARK RESULTS ===")
    logger.info(f"Clean Time     : {clean_time*1000:.2f} ms")
    logger.info(f"Chunk Time     : {chunk_time*1000:.2f} ms")
    logger.info(f"Total Time     : {total_time*1000:.2f} ms")
    logger.info(f"Total Chunks   : {len(chunks)}")

    if chunks:
        sizes = [len(c["text"]) for c in chunks]
        avg_size = sum(sizes) / len(sizes)
        logger.info(f"Avg Chunk Size : {avg_size:.1f} chars")
        logger.info(f"Max Chunk Size : {max(sizes)} chars")
        logger.info(f"Min Chunk Size : {min(sizes)} chars")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_benchmark(Path(sys.argv[1]))
    else:
        print("Usage: python benchmark.py path/to/extracted.json")
