import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

logger = logging.getLogger("docling_extractor")

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling library not installed.")

from extraction.boilerplate_detector import tag_boilerplate_blocks
from extraction.content_corrector import correct_extracted_blocks
from extraction.ner_extractor import enrich_blocks_with_ner
from extraction.block_cleaner import clean_blocks
from extraction.pdf_type_detector import is_scanned_pdf
from extraction.extraction_validator import validate_extracted_data, print_validation_report
from extraction.config import DOCLING_PIPELINE_OPTIONS

def get_docling_converter(do_ocr=False, do_table_structure=False, generate_page_images=False):
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling is not installed.")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure
    pipeline_options.generate_page_images = generate_page_images
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

def parse_docling_elements(doc):
    text_blocks, tables = [], []
    block_id_counter = 1
    if hasattr(doc, "texts"):
        for item in doc.texts:
            text = getattr(item, "text", "").strip()
            if not text:
                continue
            label = "paragraph"
            if hasattr(item, "label"):
                label_str = str(item.label).lower()
                if "heading" in label_str or "title" in label_str or "section" in label_str: label = "heading"
                elif "list" in label_str: label = "list_item"
                elif "caption" in label_str: label = "caption"
                elif "footnote" in label_str: label = "footnote"
                elif "header" in label_str: label = "header"
                elif "footer" in label_str: label = "footer"
            page_num = 1
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                if hasattr(prov, "page_no"): page_num = prov.page_no
            bbox = None
            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0] if isinstance(item.prov, list) else item.prov
                if hasattr(prov, "bbox") and prov.bbox:
                    b = prov.bbox
                    bbox = [b.l, b.t, b.r, b.b] if hasattr(b, "l") else list(b)
            text_blocks.append({"block_id": f"blk_{block_id_counter:04d}", "page_num": page_num, "type": label, "text": text, "bbox": bbox})
            block_id_counter += 1
    if hasattr(doc, "tables"):
        for tbl_idx, table in enumerate(doc.tables, start=1):
            table_dict = _export_docling_table(table, tbl_idx)
            if table_dict: tables.append(table_dict)
    return text_blocks, tables

def _export_docling_table(table, tbl_idx):
    page_num = 1
    if hasattr(table, "prov") and table.prov:
        prov = table.prov[0] if isinstance(table.prov, list) else table.prov
        if hasattr(prov, "page_no"): page_num = prov.page_no
    headers, rows = [], []
    if hasattr(table, "export_to_dataframe"):
        try:
            df = table.export_to_dataframe()
            headers = [str(c) for c in df.columns]
            rows = df.astype(str).values.tolist()
        except Exception: pass
    if rows and all("<!-- rich cell -->" in cell for row in rows for cell in row):
        return None
    caption = str(getattr(table, "caption", "") or "").strip()
    return {"table_id": f"tbl_{tbl_idx:03d}", "page_num": page_num, "caption": caption, "headers": headers, "rows": rows, "row_count": len(rows), "column_count": len(headers)}

def _fill_missing_pages_via_fitz(pdf_path, text_blocks):
    try: import fitz
    except ImportError: return text_blocks, 0
    try:
        fitz_doc = fitz.open(str(pdf_path))
        total_pages = len(fitz_doc)
        covered_pages = set(b.get("page_num") for b in text_blocks if b.get("text", "").strip())
        missing_pages = [p for p in range(1, total_pages + 1) if p not in covered_pages]
        if not missing_pages:
            fitz_doc.close()
            return text_blocks, 0
        added_blocks = 0
        block_id_start = len(text_blocks) + 1
        for p_num in missing_pages:
            page = fitz_doc[p_num - 1]
            for b in page.get_text("blocks"):
                raw_text = b[4].strip()
                if not raw_text: continue
                bbox = [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)]
                is_heading = (len(raw_text) < 60 and ("CHAPTER" in raw_text.upper() or raw_text.isupper() or raw_text.startswith("Let's")))
                text_blocks.append({"block_id": f"blk_{block_id_start:04d}", "page_num": p_num, "type": "heading" if is_heading else "paragraph", "text": raw_text, "bbox": bbox})
                block_id_start += 1
                added_blocks += 1
        fitz_doc.close()
        return text_blocks, len(missing_pages)
    except Exception: return text_blocks, 0

def save_page_images(doc, output_dir):
    image_meta = []
    if not hasattr(doc, "pages") or not doc.pages: return image_meta
    images_dir = output_dir / "page_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for page_no, page in doc.pages.items():
        if hasattr(page, "image") and page.image:
            try:
                img_filename = f"page_{page_no:03d}.png"
                img_path = images_dir / img_filename
                page.image.pil_image.save(str(img_path), "PNG")
                image_meta.append({"page_num": page_no, "filename": img_filename, "path": str(img_path.relative_to(output_dir.parent))})
            except Exception: pass
    return image_meta

def extract_document(pdf_path: Path, output_dir: Path, converter: Optional[Any] = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = pdf_path.resolve()
    opts = DOCLING_PIPELINE_OPTIONS
    if converter is None:
        do_ocr = is_scanned_pdf(resolved_path)
        converter = get_docling_converter(do_ocr=do_ocr, do_table_structure=opts.get("do_table_structure", False), generate_page_images=opts.get("generate_page_images", False))
    else: do_ocr = opts.get("do_ocr", False)
    conv_result = converter.convert(resolved_path)
    doc = conv_result.document
    text_blocks, tables = parse_docling_elements(doc)
    text_blocks, _ = _fill_missing_pages_via_fitz(resolved_path, text_blocks)
    if len(text_blocks) == 0 and not do_ocr:
        fallback_converter = get_docling_converter(do_ocr=True, do_table_structure=opts.get("do_table_structure", False), generate_page_images=opts.get("generate_page_images", False))
        conv_result = fallback_converter.convert(resolved_path)
        doc = conv_result.document
        text_blocks, tables = parse_docling_elements(doc)
        text_blocks, _ = _fill_missing_pages_via_fitz(resolved_path, text_blocks)
        do_ocr = True
    text_blocks = tag_boilerplate_blocks(text_blocks)
    text_blocks = correct_extracted_blocks(text_blocks)
    text_blocks = enrich_blocks_with_ner(text_blocks)
    text_blocks = clean_blocks(text_blocks, sort_by_reading_order=True)
    total_pdf_pages = None
    try:
        import fitz
        fitz_doc = fitz.open(str(resolved_path))
        total_pdf_pages = len(fitz_doc)
        fitz_doc.close()
    except Exception: pass
    validation_report = validate_extracted_data({"text_blocks": text_blocks, "tables": tables}, total_pdf_pages=total_pdf_pages, run_contamination_check=True)
    print_validation_report(validation_report)
    page_images = save_page_images(doc, output_dir)
    return doc, {"text_blocks": text_blocks, "tables": tables, "page_images": page_images, "block_count": len(text_blocks), "table_count": len(tables), "ocr_used": do_ocr, "validation": validation_report}
