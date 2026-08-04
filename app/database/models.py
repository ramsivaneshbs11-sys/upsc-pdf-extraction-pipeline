from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.database.session import Base


class Document(Base):
    """
    Tracks every PDF that passes through the ingestion pipeline.

    Status flow:
        registered → extracting → extracted
                               → preprocessing → preprocessed
                               → embedding → ingested
                               ↘ failed  (any step can fail)
    """
    __tablename__ = "documents"

    # ── Primary key ───────────────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(String(36), primary_key=True)   # UUID

    # ── File info ─────────────────────────────────────────────────────────────
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    classification: Mapped[str] = mapped_column(String(50), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_json_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    preprocessed_json_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Pipeline status ───────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="registered")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
