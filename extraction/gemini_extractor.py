"""
gemini_extractor.py
────────────────────
Gemini 2.5 Flash Vision-Language Model (VLM) extractor for scanned PDFs.

This module is 100% ADDITIVE — it does NOT modify any existing extraction code.
It is used only when is_scanned_pdf() returns True (i.e., the PDF has no native
text layer).

Architecture position:
    Smart Router (smart_router.py)
        ├── is_scanned_pdf() → False  →  existing extract_document()  [UNCHANGED]
        └── is_scanned_pdf() → True   →  extract_with_gemini_flash()  [THIS FILE]

Output format:
    Matches the exact same dict schema as extract_document() so json_builder.py,
    audit_extraction(), and all downstream consumers work identically for both paths.

Requirements:
    pip install google-genai pillow PyMuPDF python-dotenv

API Key Setup:
    Add GEMINI_API_KEY=your_key to your .env file.
    Free-tier: 1,500 requests/day via Google AI Studio → https://aistudio.google.com
"""

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF — page-to-image renderer

logger = logging.getLogger("gemini_extractor")

# ── Gemini SDK import (optional dependency) ───────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning(
        "google-genai SDK not installed. "
        "Run: pip install google-genai\n"
        "Gemini Flash extraction will not be available."
    )

# ── PIL for image encoding ────────────────────────────────────────────────────
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Model to use — gemini-3.5-flash-lite has optimal accuracy and standard 1,500 daily requests free-tier quota
GEMINI_MODEL = "models/gemini-3.5-flash-lite"

# Rate limiting — Free tier allows 15 requests/minute
FREE_TIER_RPM        = 10                     # Effective safe rate across all keys
REQUEST_DELAY_SECS   = 6.0                    # 6s gap = 10 pages/min safely within quota


# Page render resolution — 150 DPI is optimal: clear text without huge file size
RENDER_DPI = 150

# Maximum image bytes per request (Gemini limit: 20 MB inline)
MAX_IMAGE_BYTES = 18 * 1024 * 1024  # 18 MB safety margin

# Structured extraction prompt — tells Gemini exactly what JSON format to output
EXTRACTION_PROMPT = """
You are a document extraction engine for UPSC (India's civil service exam) study materials.

Extract ALL text from this PDF page image into structured JSON. Follow these exact rules:

1. Identify each distinct visual block on the page (headings, paragraphs, bullet lists,
   callout boxes, tables, captions, headers, footers).
2. For EACH block, output a JSON object with:
   - "type": one of ["heading", "paragraph", "list_item", "caption", "footer",
                     "header", "table", "pyq_question"]
   - "text": the exact extracted text (preserve bullet markers like z, ●, ▶)
   - "page_num": the page number you are given
   - "is_boilerplate": true ONLY for running page headers/footers and page numbers
3. For tables: output a single block with type "table" and text as a markdown table.
4. Maintain LEFT-COLUMN FIRST, then RIGHT-COLUMN reading order for two-column layouts.
5. UPSC practice questions (starting with "Q.", MCQ options (a)/(b)/(c)/(d), or
   "PREVIOUS YEAR QUESTION" banners) must use type "pyq_question".
6. Do NOT summarize. Extract every word exactly as printed.

Return ONLY a valid JSON array of block objects. No prose, no markdown fences.

Example output:
[
  {"type": "heading", "text": "1.1 Architecture in India", "page_num": 10, "is_boilerplate": false},
  {"type": "list_item", "text": "z The Indus Valley Civilization used standardized bricks.", "page_num": 10, "is_boilerplate": false},
  {"type": "footer", "text": "Indian Art & Culture 10 UPSC WALLAH", "page_num": 10, "is_boilerplate": true}
]
"""


# ─────────────────────────────────────────────────────────────────────────────
# 1. API KEY ROTATOR
# ─────────────────────────────────────────────────────────────────────────────

class KeyRotator:
    """
    Manages a pool of Gemini API keys and rotates to the next key instantly
    when a 429 Rate Limit or quota error is encountered.

    Supports two .env configurations:
        # Single key (original):
        GEMINI_API_KEY=AIzaSyKey1

        # Multiple keys (rotator — 10x daily capacity, zero rate-limit pauses):
        GEMINI_API_KEYS=AIzaSyKey1,AIzaSyKey2,AIzaSyKey3

    Usage:
        rotator = KeyRotator.from_env()
        client  = rotator.current_client()   # Get active client
        rotator.rotate()                      # Switch to next key on 429
    """

    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("At least one Gemini API key is required.")
        self._keys   = api_keys
        self._index  = 0
        self._clients: List[Any] = []
        # Pre-build all clients
        for key in self._keys:
            try:
                self._clients.append(genai.Client(api_key=key))
            except Exception as e:
                logger.warning(f"Could not init client for key ...{key[-4:]}: {e}")
        if not self._clients:
            raise RuntimeError("No valid Gemini clients could be initialized.")
        logger.info(
            f"KeyRotator initialized with {len(self._clients)} API key(s). "
            f"Daily capacity: {len(self._clients) * 1500:,} pages."
        )

    @classmethod
    def from_env(cls) -> "KeyRotator":
        """Load API keys from .env / environment and return a KeyRotator."""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai SDK not installed. Run: pip install google-genai")

        # Load .env
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).resolve().parent.parent / ".env"
            load_dotenv(env_path, override=True)
        except ImportError:
            pass

        # Priority 1: GEMINI_API_KEYS (comma-separated list)
        keys_str = os.environ.get("GEMINI_API_KEYS", "").strip()
        if keys_str:
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            logger.info(f"Loaded {len(keys)} API key(s) from GEMINI_API_KEYS.")
            return cls(keys)

        # Priority 2: Single GEMINI_API_KEY (backward compatible)
        single_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if single_key and single_key != "your_gemini_api_key_here":
            logger.info("Loaded 1 API key from GEMINI_API_KEY.")
            return cls([single_key])

        raise RuntimeError(
            "No Gemini API key found. Add to .env:\n"
            "  Single key : GEMINI_API_KEY=AIzaSy...\n"
            "  Multi key  : GEMINI_API_KEYS=AIzaSyKey1,AIzaSyKey2,AIzaSyKey3"
        )

    def current_client(self) -> Any:
        """Return the currently active Gemini client."""
        return self._clients[self._index]

    def rotate(self) -> bool:
        """
        Rotate to the next available API key.
        Returns True if a new key is available, False if all keys are exhausted.
        """
        next_index = (self._index + 1) % len(self._clients)
        if next_index == self._index:
            return False  # Only one key available, can't rotate
        self._index = next_index
        logger.warning(
            f"🔄 Rotated to API Key #{self._index + 1}/{len(self._clients)} "
            f"(...{self._keys[self._index][-4:]})"
        )
        return True

    @property
    def key_count(self) -> int:
        return len(self._clients)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PAGE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_page_to_bytes(page: Any, dpi: int = RENDER_DPI) -> Optional[bytes]:
    """
    Renders a single PyMuPDF page to a JPEG byte string for sending to Gemini API.

    Args:
        page: fitz.Page object
        dpi:  Render resolution (150 DPI balances quality vs. file size)

    Returns:
        JPEG bytes, or None on failure.
    """
    if not PIL_AVAILABLE:
        logger.error("Pillow not installed. Run: pip install pillow")
        return None

    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("jpeg")

        # Compress further if over limit
        if len(img_bytes) > MAX_IMAGE_BYTES:
            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=75, optimize=True)
            img_bytes = buf.getvalue()

        del pix
        gc.collect()
        return img_bytes

    except Exception as e:
        logger.warning(f"Page render failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. SINGLE PAGE EXTRACTOR (Key-Rotation Aware)
# ─────────────────────────────────────────────────────────────────────────────

def _local_fallback_extraction(page_num: int, pdf_path: Path) -> List[Dict]:
    """
    Fallback extraction using PyMuPDF native text or local RapidOCR
    if Gemini API blocks the page (due to false-positive recitation/safety filters).
    """
    logger.warning(f"  Page {page_num}: Running local fallback extraction...")
    try:
        doc = fitz.open(str(pdf_path))
        page = doc[page_num - 1]
        
        # 1. Try PyMuPDF native text layer
        native_text = page.get_text().strip()
        text = native_text
        is_native = bool(native_text)
        
        # 2. Try RapidOCR if native text is empty
        if not text:
            try:
                from rapidocr_onnxruntime import RapidOCR
                import numpy as np
                import cv2
                
                ocr_engine = RapidOCR()
                pix = page.get_pixmap(dpi=150)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                
                # Preprocessing
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img_for_ocr = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                
                res, _ = ocr_engine(img_for_ocr)
                if res:
                    pw = pix.width
                    x_mid = pw / 2.0
                    col1, col2 = [], []
                    
                    for item in res:
                        box, line_text = item[0], item[1].strip()
                        if not line_text:
                            continue
                        xc = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4.0
                        if xc < x_mid:
                            col1.append(line_text)
                        else:
                            col2.append(line_text)
                            
                    text = "\n".join(col1) + "\n\n" + "\n".join(col2)
            except Exception as e:
                logger.warning(f"  Page {page_num}: RapidOCR fallback failed: {e}")
                text = ""
                
        doc.close()
        
        if not text:
            text = f"[Empty Page: Page {page_num} has no text or image content]"

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]
            
        blocks = []
        for idx, para in enumerate(paragraphs, start=1):
            blocks.append({
                "block_id":         f"blk_fallback_p{page_num:04d}_{idx:03d}",
                "page_num":         page_num,
                "type":             "paragraph",
                "text":             para,
                "bbox":             [0.0, 0.0, 612.0, 792.0],
                "is_boilerplate":   False,
                "boilerplate_type": None,
                "was_corrected":    True,
                "entities":         [],
                "source":           "local_pymupdf_fallback" if is_native else "local_rapidocr_fallback",
            })
        logger.info(f"  Page {page_num}: Successfully recovered {len(blocks)} blocks via local fallback")
        return blocks
    except Exception as e:
        logger.error(f"  Page {page_num}: Local fallback failed: {e}")
        return [{
            "block_id":         f"blk_fallback_p{page_num:04d}_000",
            "page_num":         page_num,
            "type":             "paragraph",
            "text":             f"[Extraction completely blocked/failed for page {page_num}]",
            "bbox":             [0.0, 0.0, 612.0, 792.0],
            "is_boilerplate":   False,
            "boilerplate_type": None,
            "was_corrected":    False,
            "entities":         [],
            "source":           "failed_page_placeholder",
        }]


def _extract_page_with_gemini(
    rotator: "KeyRotator",
    page_image_bytes: bytes,
    page_num: int,
    pdf_path: Path,
    retries: int = 10,
) -> List[Dict]:

    """
    Sends a single rendered page image to Gemini and parses the structured JSON.
    On any 429 / quota error, instantly rotates to the next API key and retries
    immediately — zero waiting time when multiple keys are configured.

    Args:
        rotator:          KeyRotator instance managing API key pool.
        page_image_bytes: JPEG bytes of the rendered page.
        page_num:         1-indexed page number.
        retries:          Total attempts across all keys.

    Returns:
        List of block dicts matching the pipeline schema.
    """
    for attempt in range(1, retries + 1):
        client = rotator.current_client()
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    genai_types.Part.from_bytes(
                        data=page_image_bytes,
                        mime_type="image/jpeg",
                    ),
                    EXTRACTION_PROMPT + f"\n\nThis is page number: {page_num}",
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8192,
                    safety_settings=[
                        genai_types.SafetySetting(
                            category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        genai_types.SafetySetting(
                            category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        genai_types.SafetySetting(
                            category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        genai_types.SafetySetting(
                            category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]
                ),
            )

            if not response.text:
                raise ValueError("Response text is None or empty. The request might have been blocked by safety filters.")
            raw_text = response.text.strip()

            # Strip markdown code fences if Gemini wraps response
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1]
            if raw_text.endswith("```"):
                raw_text = raw_text.rsplit("```", 1)[0]

            blocks = json.loads(raw_text.strip())
            if not isinstance(blocks, list):
                blocks = [blocks]

            normalized: List[Dict] = []
            for idx, blk in enumerate(blocks, start=1):
                if not isinstance(blk, dict):
                    continue
                text = str(blk.get("text", "")).strip()
                if not text:
                    continue
                normalized.append({
                    "block_id":         f"blk_gemini_p{page_num:04d}_{idx:03d}",
                    "page_num":         page_num,
                    "type":             blk.get("type", "paragraph"),
                    "text":             text,
                    "bbox":             blk.get("bbox", [0.0, 0.0, 612.0, 792.0]),
                    "is_boilerplate":   bool(blk.get("is_boilerplate", False)),
                    "boilerplate_type": blk.get("type") if blk.get("is_boilerplate") else None,
                    "was_corrected":    False,
                    "entities":         [],
                    "source":           "gemini_flash",
                })
            # Check for LLM summary contamination (refusals to copy verbatim)
            summary_markers = [
                'based on the provided text',
                'here is a summary',
                'in summary',
                'to summarize',
                'summary of the main historical facts',
                'summary of the provided text',
            ]
            is_contaminated = False
            for b in normalized:
                t_low = b.get("text", "").lower()
                if any(m in t_low for m in summary_markers):
                    is_contaminated = True
                    break
            
            if not is_contaminated and len(normalized) <= 2:
                import re
                md_pattern = re.compile(r'\*\s{2,3}\*\*|^\s*-\s\*\*', re.M)
                for b in normalized:
                    if md_pattern.search(b.get("text", "")):
                        is_contaminated = True
                        break

            if is_contaminated:
                raise ValueError("Contaminated LLM summary output detected. Triggering local fallback.")

            key_idx = rotator._index + 1
            logger.info(
                f"  Page {page_num}: Extracted {len(normalized)} blocks "
                f"[Key #{key_idx}/{rotator.key_count}] (attempt {attempt}/{retries})"
            )
            return normalized

        except json.JSONDecodeError as e:
            logger.warning(f"  Page {page_num}: JSON parse error on attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(2)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                # ── KEY ROTATION WITH BRIEF COOLDOWN ─────────────────────────
                rotated = rotator.rotate()
                if rotated:
                    logger.warning(
                        f"  Page {page_num}: 429 on Key #{rotator._index}/{rotator.key_count} "
                        f"→ switched to Key #{rotator._index + 1}. Waiting 3s cooldown..."
                    )
                    time.sleep(3)  # Wait 3s to let project/IP rate limit reset
                    # Retry now

                else:
                    # Only one key — fall back to timed wait
                    wait_secs = 60 * attempt
                    logger.warning(
                        f"  Page {page_num}: Rate limited (single key). "
                        f"Waiting {wait_secs}s before retry..."
                    )
                    time.sleep(wait_secs)
            elif isinstance(e, (TypeError, ValueError, AttributeError)):
                logger.error(f"  Page {page_num}: Fatal logic/programming error: {e}. Skipping retries.")
                break
            else:
                logger.error(f"  Page {page_num}: API/Network error on attempt {attempt}: {e}")
                if attempt < retries:
                    sleep_secs = min(10, 2 ** attempt)
                    time.sleep(sleep_secs)

    # All retries failed — run the local fallback recovery
    logger.error(f"  Page {page_num}: All {retries} attempts failed/blocked. Running local fallback.")
    return _local_fallback_extraction(page_num, pdf_path)



# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN EXTRACTION FUNCTION (Public API)
# ─────────────────────────────────────────────────────────────────────────────

def extract_with_gemini_flash(
    pdf_path: Path,
    output_dir: Path,
    page_delay_secs: float = REQUEST_DELAY_SECS,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> Tuple[None, Dict[str, Any]]:
    """
    Full Gemini Flash extraction pipeline for scanned/image-only PDFs.

    Matches the EXACT same return signature as extract_document() so that
    json_builder, audit tools, and all downstream consumers work identically.

    Args:
        pdf_path:         Path to the scanned PDF.
        output_dir:       Directory to save page images and output files.
        page_delay_secs:  Delay between API calls (respects free-tier rate limits).
        start_page:       First page to extract (1-indexed, default: 1).
        end_page:         Last page to extract (inclusive, default: all pages).

    Returns:
        (None, extracted_data_dict) — None because there is no Docling doc object
        for scanned PDFs. The dict matches extract_document()'s output schema.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not GEMINI_AVAILABLE:
        raise ImportError(
            "google-genai SDK not installed.\n"
            "Run: pip install google-genai\n"
            "Then add GEMINI_API_KEY to your .env file."
        )

    rotator = KeyRotator.from_env()

    logger.info(f"[Gemini Flash] Starting scanned PDF extraction: {pdf_path.name}")
    logger.info(f"[Gemini Flash] API Keys available: {rotator.key_count} | Daily capacity: {rotator.key_count * 1500:,} pages")

    fitz_doc = fitz.open(str(pdf_path))
    total_pages = len(fitz_doc)
    end_page    = min(end_page or total_pages, total_pages)
    pages_range = range(start_page - 1, end_page)  # 0-indexed for fitz

    logger.info(
        f"[Gemini Flash] Pages to extract: {start_page}–{end_page} "
        f"({len(pages_range)} pages, ~{len(pages_range) * page_delay_secs:.0f}s total)"
    )

    all_text_blocks: List[Dict] = []
    page_images:     List[Dict] = []

    for page_idx in pages_range:
        page_num   = page_idx + 1
        fitz_page  = fitz_doc[page_idx]

        # ── a. Render page to JPEG bytes ──────────────────────────────────
        img_bytes = _render_page_to_bytes(fitz_page)
        if img_bytes is None:
            logger.warning(f"  Page {page_num}: Render failed, skipping.")
            continue

        # ── b. Page image disk-save DISABLED ──────────────────────────────
        # img_bytes are kept in RAM only for the Gemini API call below.
        # Writing .jpg files to disk is skipped to save disk space and I/O.
        # (page_images list is intentionally left empty — no audit images)

        # ── c. Send to Gemini API (with key rotation on 429 & local fallback) ────
        blocks = _extract_page_with_gemini(rotator, img_bytes, page_num, pdf_path)
        all_text_blocks.extend(blocks)


        del img_bytes
        gc.collect()

        # ── d. Rate-limit delay (respect free-tier 15 RPM) ───────────────
        if page_idx < pages_range[-1]:
            time.sleep(page_delay_secs)

    fitz_doc.close()

    # ── Final block ID assignment ─────────────────────────────────────────
    for i, blk in enumerate(all_text_blocks, start=1):
        blk["block_id"] = f"blk_{i:04d}"

    extracted_data = {
        "text_blocks":   all_text_blocks,
        "tables":        [],            # Tables are embedded in text_blocks as "table" type
        "page_images":   page_images,
        "block_count":   len(all_text_blocks),
        "table_count":   sum(1 for b in all_text_blocks if b.get("type") == "table"),
        "extractor":       "gemini_flash",
        "model":           GEMINI_MODEL,
        "pages_extracted": len(pages_range),
        "api_keys_used":   rotator.key_count,
    }

    logger.info(
        f"[Gemini Flash] Extraction complete: "
        f"{len(all_text_blocks)} blocks from {len(pages_range)} pages."
    )
    return None, extracted_data
