"""
docling_extractor.py
─────────────────────
Core extraction engine using Docling to parse UPSC PDFs into structured representations.

Pipeline steps per document:
  1. PDF Loading & Layout Parsing (Docling DocumentConverter)
  2. Element Classification (Headings, Paragraphs, Lists, Tables)
  3. Page-Level Image Generation (if enabled)
  4. Post-processing Pipeline:
     - Boilerplate detection (headers, footers, page numbers)
     - Text content correction (hyphen joins, ligature fixes, abbreviation normalization)
     - Named Entity Recognition (UPSC key terms, dates, acts, articles)

Returns raw Docling document object and processed block list.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Disable HuggingFace symlink warning on Windows
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

logger = logging.getLogger("docling_extractor")

# Docling imports
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling library not installed. Install via: pip install docling docling-core")

# Import postprocessor modules
from extraction.boilerplate_detector import tag_boilerplate_blocks
from extraction.content_corrector import correct_extracted_blocks
from extraction.ner_extractor import enrich_blocks_with_ner
from extraction.config import DOCLING_PIPELINE_OPTIONS


# ── 1. DOCLING CONVERTER INITIALIZATION ───────────────────────────────────────

def get_docling_converter(
    do_ocr: bool = False,
    do_table_structure: bool = True,
    generate_page_images: bool = True
) -> Any:
    """
    Builds and configures a Docling DocumentConverter instance.
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling is not installed. Run: pip install docling")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure
    pipeline_options.generate_page_images = generate_page_images

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter


# ── 2. ELEMENT PARSER ─────────────────────────────────────────────────────────

def parse_docling_elements(doc: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extracts text blocks and tables from a converted Docling document object.

    Returns:
        (text_blocks, tables)
    """
    text_blocks: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    block_id_counter = 1

    # Extract text items
    if hasattr(doc, "texts"):
        for item in doc.texts:
            text = getattr(item, "text", "").strip()
            if not text:
                continue

            # Determine element label/type
            label = "paragraph"
            if hasattr(item, "label"):
                label_str = str(item.label).lower()
                if "heading" in label_str or "title" in label_str or "section" in label_str:
                    label = "heading"
                elif "list" in label_str:
                    label = "list_item"
                elif "caption" in label_str:
                    label = "caption"
                elif "footnote" in label_str:
                    label = "footnote"
                elif "header" in label_str:
                    label = "header"
                elif "footer" in label_str:
                    label = "footer"

            # Page number
            page_num = 1
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                if hasattr(prov, "page_no"):
                    page_num = prov.page_no

            # Bounding box
            bbox = None
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                if hasattr(prov, "bbox") and prov.bbox:
                    b = prov.bbox
                    if hasattr(b, "l"):
                        bbox = [b.l, b.t, b.r, b.b]
                    elif isinstance(b, (list, tuple)):
                        bbox = list(b)

            text_blocks.append({
                "block_id": f"blk_{block_id_counter:04d}",
                "page_num": page_num,
                "type": label,
                "text": text,
                "bbox": bbox
            })
            block_id_counter += 1

    # Extract tables
    if hasattr(doc, "tables"):
        for tbl_idx, table in enumerate(doc.tables, start=1):
            table_dict = _export_docling_table(table, tbl_idx)
            if table_dict:
                tables.append(table_dict)

    return text_blocks, tables


def _export_docling_table(table: Any, tbl_idx: int) -> Dict[str, Any]:
    """Helper to export a Docling table object into a structured dict."""
    page_num = 1
    if hasattr(table, "prov") and table.prov:
        prov = table.prov[0] if isinstance(table.prov, list) else table.prov
        if hasattr(prov, "page_no"):
            page_num = prov.page_no

    headers = []
    rows = []

    if hasattr(table, "export_to_dataframe"):
        try:
            df = table.export_to_dataframe()
            headers = [str(c) for c in df.columns]
            rows = df.astype(str).values.tolist()
        except Exception:
            pass

    caption = ""
    if hasattr(table, "caption"):
        caption = str(table.caption or "").strip()

    return {
        "table_id": f"tbl_{tbl_idx:03d}",
        "page_num": page_num,
        "caption": caption,
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
        "column_count": len(headers)
    }


# ── 3. PAGE IMAGE EXPORTER ────────────────────────────────────────────────────

def save_page_images(doc: Any, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Saves page images if generated by Docling.

    Returns:
        List of image metadata dicts [{"page_num": 1, "path": "..."}, ...]
    """
    image_meta = []

    if not hasattr(doc, "pages") or not doc.pages:
        return image_meta

    images_dir = output_dir / "page_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for page_no, page in doc.pages.items():
        if hasattr(page, "image") and page.image:
            try:
                img_filename = f"page_{page_no:03d}.png"
                img_path = images_dir / img_filename
                page.image.pil_image.save(str(img_path), "PNG")

                image_meta.append({
                    "page_num": page_no,
                    "filename": img_filename,
                    "path": str(img_path.relative_to(output_dir.parent))
                })
            except Exception as e:
                logger.warning(f"Failed to save image for page {page_no}: {e}")

    return image_meta


# ── 4. PRIMARY EXTRACTION FUNCTION ───────────────────────────────────────────

def extract_document(
    pdf_path: Path,
    output_dir: Path,
    converter: Optional[Any] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Full extraction pipeline for a single PDF file:
      1. Converts PDF via Docling
      2. Parses text blocks and tables
      3. Tags boilerplate elements (headers, footers, page numbers)
      4. Corrects text errors (hyphens, ligatures, domain terms)
      5. Enriches text blocks with UPSC Named Entity Recognition (NER)
      6. Saves page images (if generated)

    Args:
        pdf_path:   Path to the input PDF.
        output_dir: Output directory to save artifacts.
        converter:  Optional pre-initialized Docling DocumentConverter.

    Returns:
        (docling_doc_object, processed_data_dict)
    """
    logger.info(f"Extracting document: {pdf_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if converter is None:
        opts = DOCLING_PIPELINE_OPTIONS
        converter = get_docling_converter(
            do_ocr=opts.get("do_ocr", False),
            do_table_structure=opts.get("do_table_structure", True),
            generate_page_images=opts.get("generate_page_images", True)
        )

    # 1. Conversion
    conv_result = converter.convert(str(pdf_path))
    doc = conv_result.document

    # 2. Parse elements
    text_blocks, tables = parse_docling_elements(doc)
    logger.info(f"Parsed {len(text_blocks)} text blocks, {len(tables)} tables")

    # 3. Postprocessor Pass 1: Tag Boilerplate
    text_blocks = tag_boilerplate_blocks(text_blocks)

    # 4. Postprocessor Pass 2: Text Correction
    text_blocks = correct_extracted_blocks(text_blocks)

    # 5. Postprocessor Pass 3: NER Enrichment
    text_blocks = enrich_blocks_with_ner(text_blocks)

    # 6. Save images
    page_images = save_page_images(doc, output_dir)

    extracted_data = {
        "text_blocks": text_blocks,
        "tables": tables,
        "page_images": page_images,
        "block_count": len(text_blocks),
        "table_count": len(tables)
    }

    return doc, extracted_data
