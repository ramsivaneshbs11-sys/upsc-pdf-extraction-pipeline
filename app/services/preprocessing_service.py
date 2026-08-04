"""
app/services/preprocessing_service.py
───────────────────────────────────────
Bridges the FastAPI endpoint with the existing preprocessing/chunking engine.
Calls process_single_json() from daily/preprocessing/main.py.

Saves output chunk JSON to:
    data/preprocessed/<file_id>_preprocessed.json

Returns:
    (success: bool, preprocessed_json_path: Path | None, error_message: str | None)
"""
import logging
import sys
from pathlib import Path

# Ensure the workspace root is on sys.path so the `preprocessing` package resolves
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent  # daily/
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from preprocessing.main import process_single_json
from app.core.config import PREPROCESSED_DIR

logger = logging.getLogger(__name__)


def run_preprocessing(
    file_id: str,
    extracted_json_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[bool, Path | None, str | None]:
    """
    Run text cleaning and layout-aware chunking on an extracted JSON file.

    Args:
        file_id:             UUID of the document record.
        extracted_json_path: Path to the data/extracted/<file_id>.json file.
        chunk_size:          Max characters per chunk (default 1000).
        chunk_overlap:       Overlap between chunks (default 200).

    Returns:
        (success, preprocessed_json_path, error_message)
    """
    logger.info(f"[{file_id}] Starting preprocessing & chunking for: {extracted_json_path.name}")

    if not extracted_json_path.exists():
        error_msg = f"Extracted JSON does not exist: {extracted_json_path}"
        logger.error(f"[{file_id}] {error_msg}")
        return False, None, error_msg

    try:
        ok = process_single_json(
            json_path=extracted_json_path,
            output_dir=PREPROCESSED_DIR,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except Exception as exc:
        error_msg = f"Preprocessing raised exception: {exc}"
        logger.exception(f"[{file_id}] {error_msg}")
        return False, None, error_msg

    if not ok:
        error_msg = "Preprocessing returned failure status"
        logger.error(f"[{file_id}] {error_msg}")
        return False, None, error_msg

    # Expected output path: data/preprocessed/<stem>_preprocessed.json
    expected_output = PREPROCESSED_DIR / f"{extracted_json_path.stem}_preprocessed.json"

    if not expected_output.exists():
        error_msg = f"Preprocessing completed but output file not found: {expected_output}"
        logger.error(f"[{file_id}] {error_msg}")
        return False, None, error_msg

    logger.info(f"[{file_id}] Preprocessing complete ✓ → {expected_output}")
    return True, expected_output, None
