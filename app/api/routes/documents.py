import uuid
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import ALLOWED_CLASSIFICATIONS
from app.database.session import get_db
from app.database import repository
from app.services.storage_service import save_uploaded_pdf
from app.services.extraction_service import run_extraction
from app.services.preprocessing_service import run_preprocessing
from app.services.embedding_service import run_embedding
from app.services.qdrant_service import run_qdrant_upsert

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["documents"])

@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def register_and_extract(
    file: UploadFile = File(...),
    classification: str = Form(...),
    db: Session = Depends(get_db),
):
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise HTTPException(status_code=400, detail=f"Invalid classification '{classification}'")

    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_id = str(uuid.uuid4())
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        saved_path = save_uploaded_pdf(file_id, classification, tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {exc}")

    doc = repository.create_document(db, file_id, filename, classification, str(saved_path))
    repository.update_document_status(db, file_id, status="extracting")
    success, json_path, error_msg = run_extraction(file_id, saved_path)

    if not success:
        return _build_response(repository.update_document_status(db, file_id, status="failed", error_message=error_msg))

    final_doc = repository.update_document_status(db, file_id, status="extracted", extracted_json_path=str(json_path))
    repository.update_document_status(db, file_id, status="preprocessing")
    pre_success, preprocessed_path, pre_error = run_preprocessing(file_id, json_path)

    if not pre_success:
        return _build_response(repository.update_document_status(db, file_id, status="failed", error_message=pre_error))

    final_doc = repository.update_document_status(db, file_id, status="preprocessed", preprocessed_json_path=str(preprocessed_path))
    repository.update_document_status(db, file_id, status="embedding")
    emb_success, embedded_chunks, emb_error = run_embedding(preprocessed_path)

    if not emb_success:
        return _build_response(repository.update_document_status(db, file_id, status="failed", error_message=emb_error))

    qdrant_success, qdrant_error = run_qdrant_upsert(file_id, classification, embedded_chunks)
    if not qdrant_success:
        return _build_response(repository.update_document_status(db, file_id, status="failed", error_message=qdrant_error))

    return _build_response(repository.update_document_status(db, file_id, status="ingested"))

def _build_response(doc) -> dict:
    return {
        "document_id": doc.id,
        "original_filename": doc.original_filename,
        "classification": doc.classification,
        "file_path": doc.file_path,
        "extracted_json_path": doc.extracted_json_path,
        "preprocessed_json_path": doc.preprocessed_json_path,
        "status": doc.status,
        "error_message": doc.error_message,
        "created_at": doc.created_at.isoformat(),
        "updated_at": doc.updated_at.isoformat(),
    }
