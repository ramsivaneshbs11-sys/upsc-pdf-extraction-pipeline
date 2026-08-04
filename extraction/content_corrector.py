import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger("content_corrector")

DOMAIN_TERM_MAP = {
    r"\bU\.P\.S\.C\.\b": "UPSC",
    r"\bI\.A\.S\.\b": "IAS",
    r"\bI\.P\.S\.\b": "IPS",
    r"\bI\.F\.S\.\b": "IFS",
    r"\bC\.S\.A\.T\.\b": "CSAT",
    r"\bN\.C\.E\.R\.T\.\b": "NCERT",
    r"\bM\.L\.A\.\b": "MLA",
    r"\bM\.P\.\b": "MP",
    r"\bB\.C\.E\.\b": "BCE",
    r"\bC\.E\.\b": "CE",
}

CHAR_REPLACEMENTS = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "–": "-", "…": "...", "\xa0": " ",
    "\u200b": "", "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi",
}

COMPILED_DOMAIN_TERMS = [(re.compile(pattern), repl) for pattern, repl in DOMAIN_TERM_MAP.items()]
HYPHEN_LINEBREAK_REGEX = re.compile(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)")
REPEATED_WORD_REGEX   = re.compile(r"\b([a-zA-Z]{3,})\s+\1\b", re.IGNORECASE)
MULTIPLE_SPACES_REGEX = re.compile(r"[ \t]{2,}")
MULTIPLE_NEWLINES_REGEX = re.compile(r"\n{3,}")

class ContentCorrector:
    def correct_text(self, text: str) -> tuple[str, bool]:
        if not text:
            return "", False
        original = text
        corrected = original
        for char, repl in CHAR_REPLACEMENTS.items():
            if char in corrected:
                corrected = corrected.replace(char, repl)
        corrected = HYPHEN_LINEBREAK_REGEX.sub(r"\1\2", corrected)
        for regex, replacement in COMPILED_DOMAIN_TERMS:
            corrected = regex.sub(replacement, corrected)
        corrected = REPEATED_WORD_REGEX.sub(r"\1", corrected)
        corrected = MULTIPLE_SPACES_REGEX.sub(" ", corrected)
        corrected = MULTIPLE_NEWLINES_REGEX.sub("\n\n", corrected)
        corrected = corrected.strip()
        return corrected, (corrected != original.strip())

    def correct_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        raw = block.get("text", "")
        corrected, changed = self.correct_text(raw)
        if changed:
            block["raw_text"] = raw
            block["text"] = corrected
            block["was_corrected"] = True
        else:
            block["was_corrected"] = False
        return block

    def correct_document(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        corrected_count = sum(1 for block in blocks if self.correct_block(block).get("was_corrected"))
        logger.info(f"ContentCorrector: Corrected {corrected_count}/{len(blocks)} blocks")
        return blocks

def correct_extracted_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    corrector = ContentCorrector()
    return corrector.correct_document(blocks)
