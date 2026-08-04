import os
import shutil
from pathlib import Path
from app.core.config import UPLOAD_DIR

def save_uploaded_pdf(file_id: str, classification: str, temp_file_path: Path) -> Path:
    class_folder = classification.lower()
    target_dir = UPLOAD_DIR / class_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{file_id}.pdf"
    shutil.move(str(temp_file_path), str(target_path))
    return target_path
