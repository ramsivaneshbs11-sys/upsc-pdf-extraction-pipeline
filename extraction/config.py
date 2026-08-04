import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = BASE_DIR / "inputs"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_CLASSIFICATIONS = [
    "M.A",
    "PATHSHALA"
]

MAX_FILE_SIZE_MB = 100

DOCLING_PIPELINE_OPTIONS = {
    "do_ocr": False,
    "do_table_structure": False,
    "generate_page_images": False,
    "generate_table_images": False,
}
