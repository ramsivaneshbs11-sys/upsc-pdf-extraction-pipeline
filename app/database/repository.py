from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.database.models import Document

def create_document(db: Session, file_id: str, original_filename: str, classification: str, file_path: str) -> Document:
    doc = Document(id=file_id, original_filename=original_filename, classification=classification, file_path=file_path, status="registered")
    db.add(doc); db.commit(); db.refresh(doc)
    return doc

def update_document_status(db: Session, file_id: str, status: str, error_message: str | None = None, extracted_json_path: str | None = None, preprocessed_json_path: str | None = None) -> Document | None:
    doc = db.get(Document, file_id)
    if doc is None: return None
    doc.status = status
    doc.updated_at = datetime.now(timezone.utc)
    if error_message is not None: doc.error_message = error_message
    if extracted_json_path is not None: doc.extracted_json_path = extracted_json_path
    if preprocessed_json_path is not None: doc.preprocessed_json_path = preprocessed_json_path
    db.commit(); db.refresh(doc)
    return doc

def get_document_by_id(db: Session, file_id: str) -> Document | None:
    return db.get(Document, file_id)
