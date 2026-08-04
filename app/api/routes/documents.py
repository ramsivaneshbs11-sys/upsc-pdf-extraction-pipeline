"""
app/api/routes/documents.py
────────────────────────────
Single FastAPI endpoint: POST /api/v1/documents

Pipeline (fully sequential, all steps block before the next starts):

  1.  Validate classification & file extension
  2.  Generate UUID
  3.  Save PDF to uploads/<classification>/<uuid>.pdf
  4.  Register document in PostgreSQL        →  status = registered
  5.  Update status                          →  status = extracting
  6.  Run Docling extraction pipeline
  7a. Success → Update status               →  status = extracted
  7b. Failure → Update status               →  status = failed
  8.  Update status                          →  status = preprocessing
  8a. Run preprocessing + chunking
  8b. Success → Update status               →  status = preprocessed
  8c. Failure → Update status               →  status = failed
  9.  Update status                          →  status = embedding
  9a. Run BGE embedding
  9b. Failure → Update status               →  status = failed
  10. Run Qdrant upsert
  10a. Success → Update status              →  status = ingested
  10b. Failure → Update status              →  status = failed
  11. Return JSON response
"""
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
    file: UploadFile = File(..., description="PDF file to ingest"),
    classification: str = Form(..., description="Document classification: History or Anthropology"),
    db: Session = Depends(get_db),
):
    """
    Register a PDF document and run the full ingestion pipeline.

    Pipeline steps (all sequential, same endpoint):
    - **Validate** → **Save** → **Register** (PostgreSQL) → **Extract** (Docling)
    - **Preprocess + Chunk** → **Embed** (BGE) → **Upsert** (Qdrant)

    - **file**: PDF file (multipart/form-data)
    - **classification**: One of `History` or `Anthropology`

    Returns the final document record from PostgreSQL.
    """

    # ── Step 1: Validate inputs ───────────────────────────────────────────────
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid classification '{classification}'. Allowed: {ALLOWED_CLASSIFICATIONS}",
        )

    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # ── Step 2: Generate UUID ─────────────────────────────────────────────────
    file_id = str(uuid.uuid4())
    logger.info(f"New document request → id={file_id}, file={filename}, class={classification}")

    # ── Step 3: Save PDF to disk ──────────────────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        saved_path = save_uploaded_pdf(file_id, classification, tmp_path)
        logger.info(f"[{file_id}] PDF saved → {saved_path}")
    except Exception as exc:
        logger.exception(f"[{file_id}] Failed to save uploaded file: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {exc}")

    # ── Step 4: Register in PostgreSQL (status = registered) ──────────────────
    doc = repository.create_document(
        db=db,
        file_id=file_id,
        original_filename=filename,
        classification=classification,
        file_path=str(saved_path),
    )
    logger.info(f"[{file_id}] Registered in PostgreSQL — status=registered")

    # ── Step 5: Update status → extracting ───────────────────────────────────
    repository.update_document_status(db, file_id, status="extracting")
    logger.info(f"[{file_id}] Status updated → extracting")

    # ── Step 6: Run extraction ────────────────────────────────────────────────
    success, json_path, error_msg = run_extraction(
        file_id=file_id,
        pdf_path=saved_path,
    )

    # ── Steps 7a / 7b: Update extraction status ───────────────────────────────
    if not success:
        final_doc = repository.update_document_status(
            db,
            file_id,
            status="failed",
            error_message=error_msg,
        )
        logger.error(f"[{file_id}] Extraction failed → status=failed | {error_msg}")
        return _build_response(final_doc)

    final_doc = repository.update_document_status(
        db,
        file_id,
        status="extracted",
        extracted_json_path=str(json_path),
    )
    logger.info(f"[{file_id}] Extraction complete ✓ → status=extracted")

    # ── Step 8: Preprocessing + Chunking ─────────────────────────────────────
    repository.update_document_status(db, file_id, status="preprocessing")
    logger.info(f"[{file_id}] Status updated → preprocessing")

    pre_success, preprocessed_path, pre_error = run_preprocessing(
        file_id=file_id,
        extracted_json_path=json_path,
    )

    if not pre_success:
        final_doc = repository.update_document_status(
            db,
            file_id,
            status="failed",
            error_message=pre_error,
        )
        logger.error(f"[{file_id}] Preprocessing failed → status=failed | {pre_error}")
        return _build_response(final_doc)

    final_doc = repository.update_document_status(
        db,
        file_id,
        status="preprocessed",
        preprocessed_json_path=str(preprocessed_path),
    )
    logger.info(f"[{file_id}] Preprocessing complete ✓ → status=preprocessed")

    # ── Step 9: Embedding ─────────────────────────────────────────────────────
    repository.update_document_status(db, file_id, status="embedding")
    logger.info(f"[{file_id}] Status updated → embedding")

    emb_success, embedded_chunks, emb_error = run_embedding(
        preprocessed_json_path=preprocessed_path,
    )

    if not emb_success:
        final_doc = repository.update_document_status(
            db,
            file_id,
            status="failed",
            error_message=emb_error,
        )
        logger.error(f"[{file_id}] Embedding failed → status=failed | {emb_error}")
        return _build_response(final_doc)

    logger.info(f"[{file_id}] Embedding complete ✓ — {len(embedded_chunks)} vectors")

    # ── Step 10: Qdrant Upsert ────────────────────────────────────────────────
    qdrant_success, qdrant_error = run_qdrant_upsert(
        file_id=file_id,
        classification=classification,
        embedded_chunks=embedded_chunks,
    )

    if not qdrant_success:
        final_doc = repository.update_document_status(
            db,
            file_id,
            status="failed",
            error_message=qdrant_error,
        )
        logger.error(f"[{file_id}] Qdrant upsert failed → status=failed | {qdrant_error}")
        return _build_response(final_doc)

    # ── Step 11: Mark as fully ingested ──────────────────────────────────────
    final_doc = repository.update_document_status(db, file_id, status="ingested")
    logger.info(f"[{file_id}] Full pipeline complete ✓ → status=ingested")

    return _build_response(final_doc)


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_response(doc) -> dict:
    """Serialize a Document ORM object to a response dict."""
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
