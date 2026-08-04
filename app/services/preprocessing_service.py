import logging
import sys
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from preprocessing.main import process_single_json
from app.core.config import PREPROCESSED_DIR

logger = logging.getLogger(__name__)

def run_preprocessing(file_id: str, extracted_json_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
    if not extracted_json_path.exists():
        return False, None, f"Extracted JSON does not exist: {extracted_json_path}"
    try:
        ok = process_single_json(json_path=extracted_json_path, output_dir=PREPROCESSED_DIR, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as exc:
        return False, None, f"Preprocessing raised exception: {exc}"
    if not ok:
        return False, None, "Preprocessing returned failure status"
    expected_output = PREPROCESSED_DIR / f"{extracted_json_path.stem}_preprocessed.json"
    if not expected_output.exists():
        return False, None, f"Output file not found: {expected_output}"
    return True, expected_output, None
