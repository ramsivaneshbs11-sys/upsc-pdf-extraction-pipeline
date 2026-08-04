import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("json_builder")

def build_structured_json(extracted_data: Dict[str, Any], pdf_path: Path, validation_report: Dict[str, Any], output_json_path: Path) -> Path:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    text_blocks = extracted_data.get("text_blocks", [])
    tables = extracted_data.get("tables", [])
    page_images = extracted_data.get("page_images", [])
    total_blocks = len(text_blocks)
    boilerplate_count = sum(1 for b in text_blocks if b.get("is_boilerplate"))
    corrected_count   = sum(1 for b in text_blocks if b.get("was_corrected"))
    entities_count    = sum(len(b.get("entities", [])) for b in text_blocks)

    schema = {
        "document_metadata": {
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "file_size_mb": validation_report.get("file_size_mb", 0.0),
            "page_count": validation_report.get("page_count", 0),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "extractor_engine": "Docling v2.0",
        },
        "extraction_summary": {
            "total_blocks": total_blocks,
            "content_blocks": total_blocks - boilerplate_count,
            "boilerplate_blocks": boilerplate_count,
            "corrected_blocks": corrected_count,
            "table_count": len(tables),
            "image_count": len(page_images),
            "total_ner_entities": entities_count
        },
        "text_blocks": text_blocks,
        "tables": tables,
        "page_images": page_images
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    logger.info(f"Structured JSON saved successfully: {output_json_path}")
    return output_json_path
