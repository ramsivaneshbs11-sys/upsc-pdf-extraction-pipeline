"""
models.py
──────────
Pydantic data models for preprocessing & chunking stage.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ChunkMetadata(BaseModel):
    file_name: str
    page_numbers: List[int]
    source_block_ids: List[str]


class ChunkItem(BaseModel):
    chunk_id: str
    text: str
    character_count: int
    metadata: ChunkMetadata


class PreprocessedOutput(BaseModel):
    metadata: Dict[str, Any]
    chunk_count: int
    chunks: List[ChunkItem]
