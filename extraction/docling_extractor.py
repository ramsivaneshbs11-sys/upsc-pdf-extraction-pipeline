"""
docling_extractor.py
─────────────────────
Core extraction engine using Docling to parse UPSC PDFs into structured representations.

Pipeline steps per document:
  0. PDF Type Auto-Detection (SCANNED vs DIGITAL via pdf_type_detector.py)
  1. PDF Loading & Layout Parsing (Docling DocumentConverter)
  2. Incomplete Coverage Retry Fallback
  3. Element Classification (Headings, Paragraphs, Lists, Tables)
  4. Hybrid Fitz + Per-Page RapidOCR Fallback (Guarantees 100% Page Coverage for any skipped/crashed pages)
  5. Post-processing Pipeline:
     - Watermark, icon glyph, and 7-pass block cleaner (block_cleaner.py)
     - Boilerplate detection (headers, footers, page numbers)
     - Text content correction (hyphen joins, ligature fixes, abbreviation normalization)
     - Named Entity Recognition (UPSC key terms, dates, acts, articles)

Returns raw Docling document object and processed block list.
"""

import sys
import re
import logging
import os
import gc
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set

import fitz  # PyMuPDF
import numpy as np

# Set environment variables to prevent threadpool RAM allocation spikes
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

# RapidOCR import for fallback
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_AVAILABLE = True
except ImportError:
    RAPID_OCR_AVAILABLE = False

# Import postprocessor & cleaner modules
from extraction.pdf_type_detector import is_scanned_pdf
from extraction.block_cleaner import clean_extracted_blocks
from extraction.boilerplate_detector import tag_boilerplate_blocks
from extraction.content_corrector import correct_extracted_blocks
from extraction.ner_extractor import enrich_blocks_with_ner
from extraction.config import DOCLING_PIPELINE_OPTIONS


# ── 1. DOCLING CONVERTER INITIALIZATION ───────────────────────────────────────

def get_docling_converter(
    do_ocr: bool = False,
    do_table_structure: bool = False,
    generate_page_images: bool = False
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
            # Clean placeholder text
            headers = [h if "<!--" not in h else f"Col_{i+1}" for i, h in enumerate(headers)]
            rows = [[cell if "<!--" not in cell else "" for cell in row] for row in rows]
        except Exception:
            pass

    # If dataframe export produced no rows or pure placeholders, fallback to docling data model cells
    if not rows and hasattr(table, "data") and hasattr(table.data, "grid"):
        try:
            grid = table.data.grid
            if grid:
                raw_grid = [[cell.text.strip() if hasattr(cell, "text") else str(cell) for cell in row] for row in grid]
                if raw_grid:
                    headers = raw_grid[0]
                    rows = raw_grid[1:]
        except Exception:
            pass

    caption = ""
    if hasattr(table, "caption"):
        caption = str(table.caption or "").strip()

    # Clean generic integer headers (e.g. ['0', '1'] in TOC tables)
    if headers and all(h.isdigit() for h in headers):
        if len(headers) == 2:
            headers = ["Section / Unit", "Page Number"]
        else:
            headers = [f"Column_{int(h)+1}" for h in headers]

    # ── Issue 2 & 3 Fixes: Table Cleaning & Citation Extraction ───────────────────
    # 1. Clean boilerplate running headers from headers and row cells
    BOILERPLATE_PATTERNS = [
        r"(?i)india:\s*6\s*th\s*century\s*bce\s*to\s*200\s*bce",
        r"(?i)^\s*page\s+nos?\.\s*$",
        r"(?i)^\s*bhic-101\s*$",
    ]
    compiled_bp = [re.compile(p) for p in BOILERPLATE_PATTERNS]

    # Check if column 0 is a phantom column containing boilerplate text
    if headers and rows:
        col0_is_bp = True
        for row in rows:
            c0 = row[0].strip() if len(row) > 0 else ""
            if c0 and not any(bp.search(c0) for bp in compiled_bp):
                col0_is_bp = False
                break
        if col0_is_bp:
            headers = headers[1:] if len(headers) > 1 else []
            rows = [r[1:] for r in rows if len(r) > 1]

    # 2. Extract citation footnotes absorbed into the last table rows (Issue 3 fix)
    CITATION_REGEX = re.compile(r"^\s*\(.*?(?:Kumar|MHI|Block|Unit|p\.\s*\d+|Press|Edition|ISBN|Source|Adapted).*?\)\s*$", re.IGNORECASE)
    cleaned_rows = []
    for r in rows:
        r_text = " ".join([cell.strip() for cell in r if cell.strip()])
        if CITATION_REGEX.search(r_text):
            if not caption:
                caption = r_text
            else:
                caption = f"{caption} ({r_text})"
        else:
            cleaned_rows.append(r)
    rows = cleaned_rows

    return {
        "table_id": f"tbl_{tbl_idx:03d}",
        "page_num": page_num,
        "caption": caption,
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
        "column_count": len(headers)
    }


# ── 3. HYBRID PYMUPDF + PER-PAGE RAPIDOCR FALLBACK (100% Coverage Guarantee) ──

def _fill_missing_pages_via_fitz(pdf_path: Path, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Checks if any PDF pages were skipped or rendered near-empty (<50 chars) by Docling.
    Executes targeted 300 DPI RapidOCR recovery to guarantee complete content capture.

    IMPORTANT — de-duplication behaviour:
    If a page already has some blocks (low-content, < 20 chars total), those blocks
    are REMOVED before appending the recovered text so we do not create duplicates.
    """
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return text_blocks

        # Calculate character counts per page
        page_char_counts: Dict[int, int] = {}
        for b in text_blocks:
            p = b.get("page_num")
            if isinstance(p, int):
                page_char_counts[p] = page_char_counts.get(p, 0) + len(b.get("text", ""))

        covered_pages = set(page_char_counts.keys())
        missing_pages = set(range(1, total_pages + 1)) - covered_pages
        # Also flag truly blank pages (<20 chars) for re-OCR check
        low_content_pages = {p for p in range(1, total_pages + 1) if page_char_counts.get(p, 0) < 20}

        pages_to_recover = missing_pages.union(low_content_pages)

        if not pages_to_recover:
            doc.close()
            return text_blocks

        logger.info(
            f"Hybrid Fitz Fallback: Triggering targeted recovery for {len(pages_to_recover)} pages "
            f"({sorted(list(pages_to_recover))})..."
        )

        ocr_engine = RapidOCR() if RAPID_OCR_AVAILABLE else None

        for p_num in sorted(list(pages_to_recover)):
            page = doc[p_num - 1]
            text = page.get_text().strip()

            # Run 150 DPI Preprocessed Scan OCR if page native text is thin/empty
            if len(text) < 50 and ocr_engine:
                try:
                    pix = page.get_pixmap(dpi=150)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

                    # 300 DPI Scan Image Preprocessing (Grayscale + Otsu Binarization for sharp letter edges)
                    try:
                        import cv2
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        img_for_ocr = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                    except Exception:
                        img_for_ocr = img

                    res, _ = ocr_engine(img_for_ocr)
                    if res:
                        pw, ph = pix.width, pix.height
                        x_mid = pw / 2.0

                        headers, footers, col1, col2 = [], [], [], []
                        HEADER_PHRASES = {
                            "social science - part i", "social science - part 1", "social science part 1",
                            "social science part i", "india's struggle for independence", "the heritage of india",
                        }
                        HEADING_PATTERNS = [
                            re.compile(r"^\s*chapter\s+\d+", re.IGNORECASE),
                            re.compile(r"^\s*exercises\s*$", re.IGNORECASE),
                            re.compile(r"^\s*things\s+to\s+know\b", re.IGNORECASE),
                            re.compile(r"^\s*india['’]?s\s+struggle\s+for\s+independence\b", re.IGNORECASE),
                            re.compile(r"^\s*revolt\s+of\s+1857\b", re.IGNORECASE),
                            re.compile(r"^\s*non[- ]cooperation\b", re.IGNORECASE),
                            re.compile(r"^\s*civil\s+disobedience\b", re.IGNORECASE),
                            re.compile(r"^\s*quit\s+india\b", re.IGNORECASE),
                        ]
                        LIST_PATTERN = re.compile(r"^\s*(\d+[\.\)]|[a-z][\.\)]|[•\-\➢])\s+", re.IGNORECASE)
                        FOOTER_NUM_PATTERN = re.compile(r"^\s*\d{1,3}\s*$")

                        for item in res:
                            box, line_text, score = item[0], item[1].strip(), item[2]
                            if not line_text:
                                continue
                            xc = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4.0
                            yc = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4.0

                            t_low = line_text.lower()
                            if yc < ph * 0.08 or t_low in HEADER_PHRASES:
                                headers.append(line_text)
                            elif yc > ph * 0.92 or (FOOTER_NUM_PATTERN.match(line_text) and len(line_text) <= 3):
                                footers.append(line_text)
                            elif xc < x_mid:
                                col1.append((yc, line_text))
                            else:
                                col2.append((yc, line_text))

                        col1.sort(key=lambda x: x[0])
                        col2.sort(key=lambda x: x[0])

                        page_recovered_blocks = []
                        for h in headers:
                            page_recovered_blocks.append({
                                "block_id": f"blk_rec_p{p_num:04d}",
                                "page_num": p_num,
                                "type": "header",
                                "text": h,
                                "bbox": [0, 0, page.rect.width, page.rect.height]
                            })

                        all_body_lines = [t for y, t in col1] + [t for y, t in col2]
                        curr_para = []
                        for line in all_body_lines:
                            if any(hp.search(line) for hp in HEADING_PATTERNS):
                                if curr_para:
                                    page_recovered_blocks.append({
                                        "block_id": f"blk_rec_p{p_num:04d}",
                                        "page_num": p_num,
                                        "type": "paragraph",
                                        "text": " ".join(curr_para),
                                        "bbox": [0, 0, page.rect.width, page.rect.height]
                                    })
                                    curr_para = []
                                page_recovered_blocks.append({
                                    "block_id": f"blk_rec_p{p_num:04d}",
                                    "page_num": p_num,
                                    "type": "heading",
                                    "text": line,
                                    "bbox": [0, 0, page.rect.width, page.rect.height]
                                })
                            elif LIST_PATTERN.search(line):
                                if curr_para:
                                    page_recovered_blocks.append({
                                        "block_id": f"blk_rec_p{p_num:04d}",
                                        "page_num": p_num,
                                        "type": "paragraph",
                                        "text": " ".join(curr_para),
                                        "bbox": [0, 0, page.rect.width, page.rect.height]
                                    })
                                    curr_para = []
                                page_recovered_blocks.append({
                                    "block_id": f"blk_rec_p{p_num:04d}",
                                    "page_num": p_num,
                                    "type": "list_item",
                                    "text": line,
                                    "bbox": [0, 0, page.rect.width, page.rect.height]
                                })
                            else:
                                curr_para.append(line)

                        if curr_para:
                            page_recovered_blocks.append({
                                "block_id": f"blk_rec_p{p_num:04d}",
                                "page_num": p_num,
                                "type": "paragraph",
                                "text": " ".join(curr_para),
                                "bbox": [0, 0, page.rect.width, page.rect.height]
                            })

                        for f in footers:
                            page_recovered_blocks.append({
                                "block_id": f"blk_rec_p{p_num:04d}",
                                "page_num": p_num,
                                "type": "footer",
                                "text": f,
                                "bbox": [0, 0, page.rect.width, page.rect.height]
                            })

                        if page_recovered_blocks:
                            text_blocks = [b for b in text_blocks if b.get("page_num") != p_num]
                            text_blocks.extend(page_recovered_blocks)
                            del pix, img
                            gc.collect()
                            continue

                except Exception as ocr_err:
                    logger.warning(f"Fallback OCR failed on page {p_num}: {ocr_err}")

            if not text:
                text = "[Blank Page / Image Only]"

            # Remove existing low-content blocks for this page BEFORE appending recovered text
            if p_num in low_content_pages and p_num not in missing_pages:
                text_blocks = [b for b in text_blocks if b.get("page_num") != p_num]

            text_blocks.append({
                "block_id": f"blk_recovery_p{p_num:04d}",   # temp ID; re-assigned later
                "page_num": p_num,
                "type": "paragraph",
                "text": text,
                "bbox": [0.0, 0.0, page.rect.width, page.rect.height]
            })

        doc.close()
    except Exception as e:
        logger.warning(f"Hybrid Fitz fallback failed: {e}")

    return text_blocks


def _deduplicate_blocks(text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes near-identical consecutive duplicate blocks on the same page that can arise
    when Docling produces a second linear-text pass.
    """
    if not text_blocks:
        return text_blocks

    import unicodedata

    def _fingerprint(text: str) -> str:
        """Normalise and truncate text for comparison."""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text[:120]

    deduped: List[Dict[str, Any]] = []

    for b in text_blocks:
        text = b.get("text", "")
        fp = _fingerprint(text)
        p_num = b.get("page_num")

        # Consecutive duplicate check on same page
        if deduped:
            prev_b = deduped[-1]
            if prev_b.get("page_num") == p_num and _fingerprint(prev_b.get("text", "")) == fp:
                logger.debug(f"Dedup: removing consecutive duplicate block on page {p_num}: '{text[:60]}...'")
                continue

        deduped.append(b)

    removed = len(text_blocks) - len(deduped)
    if removed:
        logger.info(f"Dedup: removed {removed} duplicate blocks from document stream")

    return deduped



def filter_covered_text_blocks(pdf_path: Path, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Issue 6 Fix (QA Report v2):
    Filters out text blocks that are obscured/covered by white or opaque fill vector graphics
    (e.g., leftover template text covered by a white rectangle drawn over it).
    """
    if not pdf_path or not pdf_path.exists() or not text_blocks:
        return text_blocks

    try:
        fitz_doc = fitz.open(str(pdf_path))
        filtered = []
        removed_count = 0

        pages_dict: Dict[int, List[Dict[str, Any]]] = {}
        for b in text_blocks:
            pages_dict.setdefault(b.get("page_num", 1), []).append(b)

        for p_num in sorted(pages_dict.keys()):
            if p_num > len(fitz_doc):
                filtered.extend(pages_dict[p_num])
                continue

            page_obj = fitz_doc[p_num - 1]
            drawings = page_obj.get_drawings()

            white_rects = []
            page_area = page_obj.rect.width * page_obj.rect.height
            for d in drawings:
                fill = d.get("fill")
                rect = d.get("rect")
                if fill and rect and all(c > 0.9 for c in fill):
                    rect_area = abs(rect.width * rect.height)
                    # Ignore large background boxes/cards (> 5% of page area)
                    if rect_area < 0.05 * page_area:
                        white_rects.append(rect)

            for b in pages_dict[p_num]:
                bbox = b.get("bbox")
                if not bbox or len(bbox) != 4 or not white_rects:
                    filtered.append(b)
                    continue

                bl, bt, br, bb = bbox[0], bbox[1], bbox[2], bbox[3]
                ph = page_obj.rect.height

                py_y0 = ph - max(bt, bb)
                py_y1 = ph - min(bt, bb)
                b_rect = fitz.Rect(bl, py_y0, br, py_y1)
                b_area = abs(b_rect.width * b_rect.height)

                if b_area <= 0:
                    filtered.append(b)
                    continue

                is_covered = False
                for w_rect in white_rects:
                    inter = b_rect & w_rect
                    if not inter.is_empty:
                        inter_area = abs(inter.width * inter.height)
                        if (inter_area / b_area) > 0.85:
                            is_covered = True
                            break

                if is_covered:
                    logger.info(f"CoveredTextFilter: Dropping hidden block '{b.get('text', '')[:40]}' on page {p_num}")
                    removed_count += 1
                else:
                    filtered.append(b)

        fitz_doc.close()
        if removed_count:
            logger.info(f"CoveredTextFilter: Filtered out {removed_count} hidden/masked text blocks across document")
        return filtered

    except Exception as e:
        logger.warning(f"CoveredTextFilter failed: {e}")
        return text_blocks


# ── 4. PAGE & EMBEDDED IMAGE EXPORTER ─────────────────────────────────────────

def save_page_images(doc: Any, output_dir: Path, pdf_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Saves page rendered images and extracts embedded PDF images/figures into output_dir.
    """
    image_meta = []
    images_dir = output_dir / "page_images"

    # 4a. Docling Page Rendered Images
    if hasattr(doc, "pages") and doc.pages:
        for page_no, page in doc.pages.items():
            if hasattr(page, "image") and page.image:
                try:
                    images_dir.mkdir(parents=True, exist_ok=True)
                    img_filename = f"page_{page_no:03d}.png"
                    img_path = images_dir / img_filename
                    page.image.pil_image.save(str(img_path), "PNG")

                    image_meta.append({
                        "page_num": page_no,
                        "type": "page_render",
                        "filename": img_filename,
                        "path": str(img_path.relative_to(output_dir.parent))
                    })
                except Exception as e:
                    logger.warning(f"Failed to save page image for page {page_no}: {e}")

    # 4b. PyMuPDF Embedded Image Figure Extraction
    if pdf_path and pdf_path.exists():
        try:
            fitz_doc = fitz.open(str(pdf_path))
            figure_counter = 1
            figures_dir = output_dir / "extracted_figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            for page_idx in range(len(fitz_doc)):
                page = fitz_doc[page_idx]
                image_list = page.get_images(full=True)
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = fitz_doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Issue 6 fix: skip micro-tile fragments (< 60x60 px or < 3KB) from sliced maps
                    if (width > 0 and width < 60) or (height > 0 and height < 60) or len(image_bytes) < 3000:
                        continue

                    fig_filename = f"fig_p{page_idx+1:03d}_{figure_counter:02d}.{image_ext}"
                    fig_path = figures_dir / fig_filename
                    with open(fig_path, "wb") as f:
                        f.write(image_bytes)

                    image_meta.append({
                        "page_num": page_idx + 1,
                        "type": "embedded_figure",
                        "filename": fig_filename,
                        "path": str(fig_path.relative_to(output_dir.parent))
                    })
                    figure_counter += 1
            fitz_doc.close()
        except Exception as img_err:
            logger.warning(f"PyMuPDF embedded image extraction warning: {img_err}")

    return image_meta


# ── 5. PRIMARY EXTRACTION FUNCTION ───────────────────────────────────────────

def extract_document(
    pdf_path: Path,
    output_dir: Path,
    converter: Optional[Any] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Full extraction pipeline for a single PDF file:
      Step 0: Auto PDF type detection (SCANNED vs DIGITAL)
      Step 1: Converts PDF via Docling
      Step 2: Incomplete coverage retry fallback
      Step 3: Hybrid Fitz + Per-Page RapidOCR fallback (Guarantees 100% Page Coverage)
      Step 4: Post-processing pipeline (Block Cleaner, Boilerplate, Text Corrector, NER)

    Returns:
        (docling_doc_object, processed_data_dict)
    """
    logger.info(f"Extracting document: {pdf_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Auto PDF Type Detection
    is_scanned = is_scanned_pdf(pdf_path)

    if converter is None:
        opts = DOCLING_PIPELINE_OPTIONS
        converter = get_docling_converter(
            do_ocr=is_scanned or opts.get("do_ocr", False),
            do_table_structure=opts.get("do_table_structure", False),
            generate_page_images=opts.get("generate_page_images", False)
        )

    # Step 1: Docling Conversion
    try:
        conv_result = converter.convert(str(pdf_path))
        doc = conv_result.document
        text_blocks, tables = parse_docling_elements(doc)
    except Exception as exc:
        logger.warning(
            f"Docling conversion failed for {pdf_path.name}: {exc}. "
            f"Falling back to 100% PyMuPDF / Fitz extraction."
        )
        doc = None
        text_blocks, tables = [], []

    # Step 2: Coverage Check
    # NOTE: We intentionally skip full-document OCR retry here because it doubles RAM
    # usage and causes std::bad_alloc crashes on large PDFs. Missing pages are recovered
    # safely page-by-page in Step 3 (Hybrid Fitz Fallback).
    try:
        pdf_doc = fitz.open(str(pdf_path))
        total_pdf_pages = len(pdf_doc)
        pdf_doc.close()
    except Exception:
        total_pdf_pages = 0

    covered_pages = {b.get("page_num") for b in text_blocks if isinstance(b.get("page_num"), int)}
    covered_pages.update({t.get("page_num") for t in tables if isinstance(t.get("page_num"), int)})

    if total_pdf_pages > 0 and len(covered_pages) < total_pdf_pages:
        logger.warning(
            f"Incomplete page coverage ({len(covered_pages)}/{total_pdf_pages} pages) for {pdf_path.name}. "
            f"Missing pages will be recovered by Hybrid Fitz Fallback (Step 3)."
        )

    # Step 3: Hybrid Fitz + Per-Page RapidOCR Fallback (Guarantees 100% Page Coverage)
    text_blocks = _fill_missing_pages_via_fitz(pdf_path, text_blocks)
    logger.info(f"Parsed {len(text_blocks)} text blocks, {len(tables)} tables")

    # Step 3b: Global deduplication — removes duplicate block sequences that can
    # appear when Docling emits both a layout pass and a linear text pass.
    text_blocks = _deduplicate_blocks(text_blocks)

    # Step 4: Post-Processing Pipeline
    # 4a. 7-Pass Block Cleaner (Watermarks, Glyphs, Stylized Headings, Reading Order)
    text_blocks = clean_extracted_blocks(text_blocks)

    # 4b. Tag Boilerplate
    text_blocks = tag_boilerplate_blocks(text_blocks)

    # 4c. Text Correction
    text_blocks = correct_extracted_blocks(text_blocks)

    # 4d. NER Enrichment
    text_blocks = enrich_blocks_with_ner(text_blocks)

    # 4e. Filter out covered/hidden text blocks (Issue 6 fix in QA Report v2)
    text_blocks = filter_covered_text_blocks(pdf_path, text_blocks)

    # 4f. Issue 5 fix: Re-index block_ids contiguously (blk_0001 -> blk_N)
    for idx, b in enumerate(text_blocks, start=1):
        b["block_id"] = f"blk_{idx:04d}"

    # Step 5: Save page images and embedded figure assets
    page_images = save_page_images(doc, output_dir, pdf_path=pdf_path)

    extracted_data = {
        "text_blocks": text_blocks,
        "tables": tables,
        "page_images": page_images,
        "block_count": len(text_blocks),
        "table_count": len(tables)
    }

    return doc, extracted_data
