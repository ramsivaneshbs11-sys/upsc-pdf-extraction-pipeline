import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("boilerplate_detector")

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
    r"^\s*\d+\s*$",
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
    r"\.{4,}\s*\d+\s*$",
    r"_{4,}\s*\d+\s*$",
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

COMPILED_HEADER    = [re.compile(p) for p in HEADER_PATTERNS]
COMPILED_FOOTER    = [re.compile(p) for p in FOOTER_PATTERNS]
COMPILED_COPYRIGHT = [re.compile(p) for p in COPYRIGHT_PATTERNS]
COMPILED_TOC       = [re.compile(p) for p in TOC_PATTERNS]
COMPILED_WATERMARK = [re.compile(p) for p in WATERMARK_PATTERNS]

class BoilerplateDetector:
    def __init__(self, page_height: float = 842.0):
        self.page_height = page_height

    def detect_block(self, block: Dict[str, Any], page_height: Optional[float] = None) -> Dict[str, Any]:
        text = block.get("text", "").strip()
        if not text:
            block["is_boilerplate"] = False
            block["boilerplate_type"] = None
            return block
        bbox = block.get("bbox")
        top_y = bbox[1] if (bbox and len(bbox) >= 4) else None
        bottom_y = bbox[3] if (bbox and len(bbox) >= 4) else None
        ph = page_height or self.page_height
        b_type = self._check_text_and_position(text, top_y, bottom_y, ph)
        block["is_boilerplate"] = bool(b_type)
        block["boilerplate_type"] = b_type
        return block

    def process_document(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for block in blocks:
            self.detect_block(block)
        line_page_map: Dict[str, set] = {}
        for block in blocks:
            text = block.get("text", "").strip()
            page = block.get("page_num", 1)
            if text and len(text) < 120:
                line_page_map.setdefault(text, set()).add(page)
        repeating_texts = {text for text, pages in line_page_map.items() if len(pages) >= 3}
        for block in blocks:
            if not block.get("is_boilerplate", False):
                text = block.get("text", "").strip()
                if text in repeating_texts:
                    block["is_boilerplate"] = True
                    block["boilerplate_type"] = "repeating_header_footer"
        return blocks

    def _check_text_and_position(self, text, top_y, bottom_y, page_height):
        for rx in COMPILED_FOOTER:
            if rx.search(text): return "footer"
        for rx in COMPILED_HEADER:
            if rx.search(text): return "header"
        if top_y is not None and top_y < (page_height * 0.07) and len(text) < 80:
            if any(w in text.lower() for w in ["upsc", "paper", "module", "chapter", "notes"]):
                return "header"
        if bottom_y is not None and bottom_y > (page_height * 0.93) and len(text) < 60:
            return "footer"
        for rx in COMPILED_COPYRIGHT:
            if rx.search(text): return "copyright"
        for rx in COMPILED_TOC:
            if rx.search(text): return "toc"
        for rx in COMPILED_WATERMARK:
            if rx.search(text): return "watermark"
        return None

def tag_boilerplate_blocks(blocks: List[Dict[str, Any]], page_height: float = 842.0) -> List[Dict[str, Any]]:
    detector = BoilerplateDetector(page_height=page_height)
    return detector.process_document(blocks)
