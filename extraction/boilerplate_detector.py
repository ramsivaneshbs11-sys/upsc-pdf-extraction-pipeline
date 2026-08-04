"""
boilerplate_detector.py
─────────────────────────
Detects and tags boilerplate text in UPSC study material.

Boilerplate types detected:
  1. Header / Running Title
     - Standard header phrases: "UPSC Civil Services", "General Studies", "IGNOU", etc.
     - Document title repetitions on every page.
  2. Footer / Page Number
     - "Page X of Y", "Page X", bare numbers at top/bottom margin.
  3. Publisher / Copyright Notice
     - "All Rights Reserved", "For internal use only", website URLs, phone numbers.
  4. Table of Contents / Index Line
     - Lines of dots leading to a page number (e.g. "Chapter 1 ........ 12").
  5. Watermark / Background Text
     - Common overlay phrases: "DRAFT", "CONFIDENTIAL", "SAMPLE ONLY", coaching institute names.

Output field added to block dicts:
  "is_boilerplate": bool
  "boilerplate_type": str | None   ("header" | "footer" | "copyright" | "toc" | "watermark")
"""

import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("boilerplate_detector")

# ── 1. PATTERN DEFINITIONS ────────────────────────────────────────────────────

HEADER_PATTERNS = [
    r"(?i)^\s*upsc\s+(civil\s+services|prelims|mains|examination)",
    r"(?i)^\s*general\s+studies\s*[-–:]?\s*paper\s+[i|v|x\d]+",
    r"(?i)^\s*egyankosh\s*\|\s*ignou",
    r"(?i)^\s*subject\s*:\s*(history|geography|polity|economy|anthropology)",
    r"(?i)^\s*module\s+\d+\s*[-–:]?\s*",
    r"(?i)^\s*chapter\s+\d+\s*[-–:]?\s*$",
    r"(?i)^\s*vision\s+ias\s*",
    r"(?i)^\s*drishti\s+ias\s*",
    r"(?i)^\s*insights\s+on\s+india\s*",
    r"(?i)^\s*forum\s+ias\s*",
    r"(?i)^\s*byju'?s\s+classes\s*",
]

FOOTER_PATTERNS = [
    r"(?i)^\s*page\s+\d+\s*(of\s+\d+)?\s*$",
    r"^\s*\d+\s*/\s*\d+\s*$",
    r"^\s*-\s*\d+\s*-\s*$",
    r"^\s*\[\s*\d+\s*\]\s*$",
    r"^\s*\d+\s*$",                            # bare page number line
    r"(?i)^\s*continued\s+on\s+next\s+page",
    r"(?i)^\s*turn\s+over\s*$",
    r"(?i)^\s*pto\s*$",
]

COPYRIGHT_PATTERNS = [
    r"(?i)all\s+rights?\s+reserved",
    r"(?i)copyright\s*(©|\(c\))?\s*\d{4}",
    r"(?i)for\s+(internal|personal)\s+use\s+only",
    r"(?i)do\s+not\s+(copy|duplicate|reproduce|distribute)",
    r"(?i)no\s+part\s+of\s+this\s+(publication|document)\s+may\s+be",
    r"(?i)https?://[^\s]+",
    r"(?i)www\.[a-z0-9\-]+\.[a-z]{2,}",
    r"(?i)email\s*:\s*[^\s]+@[^\s]+",
    r"(?i)call\s*/?\s*whatsapp\s*:\s*\+?\d[\d\s\-]{8,}",
    r"(?i)contact\s*:\s*\+?\d[\d\s\-]{8,}",
]

TOC_PATTERNS = [
    r"\.{4,}\s*\d+\s*$",                       # Dots leading to page number: "Intro ...... 5"
    r"_{4,}\s*\d+\s*$",                        # Underscores leading to number
    r"(?i)^\s*contents\s*$",
    r"(?i)^\s*table\s+of\s+contents\s*$",
]

WATERMARK_PATTERNS = [
    r"(?i)^\s*draft\s*$",
    r"(?i)^\s*confidential\s*$",
    r"(?i)^\s*sample\s+only\s*$",
    r"(?i)^\s*for\s+review\s+only\s*$",
    r"(?i)^\s*not\s+for\s+sale\s*$",
]


# ── 2. COMPILED REGEXES ───────────────────────────────────────────────────────

COMPILED_HEADER    = [re.compile(p) for p in HEADER_PATTERNS]
COMPILED_FOOTER    = [re.compile(p) for p in FOOTER_PATTERNS]
COMPILED_COPYRIGHT = [re.compile(p) for p in COPYRIGHT_PATTERNS]
COMPILED_TOC       = [re.compile(p) for p in TOC_PATTERNS]
COMPILED_WATERMARK = [re.compile(p) for p in WATERMARK_PATTERNS]


# ── 3. DETECTOR CLASS ─────────────────────────────────────────────────────────

class BoilerplateDetector:
    """
    Detects boilerplate elements in structured document blocks.
    Can operate in single-pass mode or multi-page frequency analysis mode.
    """

    def __init__(self, page_height: float = 842.0):
        """
        Args:
            page_height: Typical A4 page height in points (842 pt = 297mm).
                         Used for margin-relative position checks.
        """
        self.page_height = page_height
        self._page_top_margin    = page_height * 0.08   # Top 8% of page
        self._page_bottom_margin = page_height * 0.92   # Bottom 8% of page

    def detect_block(self, block: Dict[str, Any], page_height: Optional[float] = None) -> Dict[str, Any]:
        """
        Examines a single block and adds boilerplate metadata fields:
          "is_boilerplate": bool
          "boilerplate_type": str | None

        Returns the modified block dict.
        """
        text = block.get("text", "").strip()
        if not text:
            block["is_boilerplate"] = False
            block["boilerplate_type"] = None
            return block

        # Use block bbox if available for position-based detection
        bbox = block.get("bbox")
        top_y = bbox[1] if (bbox and len(bbox) >= 4) else None
        bottom_y = bbox[3] if (bbox and len(bbox) >= 4) else None
        ph = page_height or self.page_height

        b_type = self._check_text_and_position(text, top_y, bottom_y, ph)

        if b_type:
            block["is_boilerplate"] = True
            block["boilerplate_type"] = b_type
        else:
            block["is_boilerplate"] = False
            block["boilerplate_type"] = None

        return block

    def process_document(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Two-pass detection on full document block list:
          Pass 1: Pattern + position detection per block.
          Pass 2: Frequency analysis across pages (repeating lines on 3+ pages = header/footer).
        """
        # Pass 1: Pattern check
        for block in blocks:
            self.detect_block(block)

        # Pass 2: Repeating line detection (frequency analysis across pages)
        line_page_map: Dict[str, set] = {}
        for block in blocks:
            text = block.get("text", "").strip()
            page = block.get("page_num", 1)
            if text and len(text) < 120:         # Only track short-to-medium lines
                line_page_map.setdefault(text, set()).add(page)

        # Lines appearing identically on 3 or more distinct pages are likely boilerplate
        repeating_texts = {text for text, pages in line_page_map.items() if len(pages) >= 3}

        for block in blocks:
            if not block.get("is_boilerplate", False):
                text = block.get("text", "").strip()
                if text in repeating_texts:
                    block["is_boilerplate"] = True
                    block["boilerplate_type"] = "repeating_header_footer"

        return blocks

    # ── PRIVATE HELPERS ───────────────────────────────────────────────────────

    def _check_text_and_position(
        self,
        text: str,
        top_y: Optional[float],
        bottom_y: Optional[float],
        page_height: float
    ) -> Optional[str]:

        # 1. Check Page Number / Bare Footer
        for rx in COMPILED_FOOTER:
            if rx.search(text):
                return "footer"

        # 2. Check Header Patterns
        for rx in COMPILED_HEADER:
            if rx.search(text):
                return "header"

        # 3. Position-assisted Top Margin Header Check
        if top_y is not None and top_y < (page_height * 0.07) and len(text) < 80:
            if any(word in text.lower() for word in ["upsc", "paper", "module", "chapter", "notes"]):
                return "header"

        # 4. Position-assisted Bottom Margin Footer Check
        if bottom_y is not None and bottom_y > (page_height * 0.93) and len(text) < 60:
            return "footer"

        # 5. Copyright / Contact info
        for rx in COMPILED_COPYRIGHT:
            if rx.search(text):
                return "copyright"

        # 6. Table of Contents line
        for rx in COMPILED_TOC:
            if rx.search(text):
                return "toc"

        # 7. Watermark
        for rx in COMPILED_WATERMARK:
            if rx.search(text):
                return "watermark"

        return None


# ── 4. CONVENIENCE FUNCTION ───────────────────────────────────────────────────

def tag_boilerplate_blocks(
    blocks: List[Dict[str, Any]],
    page_height: float = 842.0
) -> List[Dict[str, Any]]:
    """
    Tags all blocks in a document with 'is_boilerplate' and 'boilerplate_type'.

    Args:
        blocks: List of block dictionaries.
        page_height: Height of page in points (default: A4 = 842 pt).

    Returns:
        List of blocks with boilerplate metadata populated.
    """
    detector = BoilerplateDetector(page_height=page_height)
    return detector.process_document(blocks)
