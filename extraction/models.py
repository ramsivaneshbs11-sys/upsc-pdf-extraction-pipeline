"""
models.py
──────────
Pydantic data models for extraction data structures.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class BoundingBox(BaseModel):
    left: float
    top: float
    right: float
    bottom: float


class EntityItem(BaseModel):
    text: str
    label: str
    start: int
    end: int


class TextBlock(BaseModel):
    block_id: str
    page_num: int = Field(ge=1)
    type: str = Field(description="heading | paragraph | list_item | caption | footnote | header | footer")
    text: str
    bbox: Optional[List[float]] = None
    is_boilerplate: bool = False
    boilerplate_type: Optional[str] = None
    was_corrected: bool = False
    raw_text: Optional[str] = None
    entities: List[EntityItem] = Field(default_factory=list)


class TableItem(BaseModel):
    table_id: str
    page_num: int = Field(ge=1)
    caption: str = ""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    row_count: int = 0
    column_count: int = 0


class DocumentMetadata(BaseModel):
    file_name: str
    file_path: str
    file_size_mb: float
    page_count: int
    extracted_at: str
    extractor_engine: str = "Docling v2.0"


class ExtractionSummary(BaseModel):
    total_blocks: int
    content_blocks: int
    boilerplate_blocks: int
    corrected_blocks: int
    table_count: int
    image_count: int
    total_ner_entities: int


class ExtractedDocument(BaseModel):
    document_metadata: DocumentMetadata
    extraction_summary: ExtractionSummary
    text_blocks: List[TextBlock]
    tables: List[TableItem] = Field(default_factory=list)
    page_images: List[Dict[str, Any]] = Field(default_factory=list)
