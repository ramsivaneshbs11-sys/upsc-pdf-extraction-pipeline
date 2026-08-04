"""
content_corrector.py
──────────────────────
Fixes common OCR/Docling extraction errors in UPSC study material text.

Corrections applied:
  1. Hyphenated word break joining across line breaks ("gov- \n ernment" -> "government")
  2. OCR artifact replacements (ligatures, odd unicode quotes/dashes)
  3. UPSC domain term normalization ("U.P.S.C." -> "UPSC", "I.A.S." -> "IAS")
  4. De-duplication of accidental repeated words ("the the" -> "the")
  5. Whitespace normalization (collapsing multiple spaces/newlines)

Output field added/updated in block dicts:
  "text": corrected text string
  "was_corrected": bool
"""

import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger("content_corrector")

# ── 1. DOMAIN REPLACEMENTS MAP ────────────────────────────────────────────────

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

# OCR Character fixes
CHAR_REPLACEMENTS = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "—": "-",
    "–": "-",
    "…": "...",
    "\xa0": " ",       # Non-breaking space
    "\u200b": "",      # Zero-width space
    "ﬁ": "fi",         # Ligature fi
    "ﬂ": "fl",         # Ligature fl
    "ﬀ": "ff",         # Ligature ff
    "ﬃ": "ffi",        # Ligature ffi
}

COMPILED_DOMAIN_TERMS = [(re.compile(pattern), repl) for pattern, repl in DOMAIN_TERM_MAP.items()]
HYPHEN_LINEBREAK_REGEX = re.compile(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)")
REPEATED_WORD_REGEX   = re.compile(r"\b([a-zA-Z]{3,})\s+\1\b", re.IGNORECASE)
MULTIPLE_SPACES_REGEX = re.compile(r"[ \t]{2,}")
MULTIPLE_NEWLINES_REGEX = re.compile(r"\n{3,}")


# ── 2. CONTENT CORRECTOR CLASS ────────────────────────────────────────────────

class ContentCorrector:
    """
    Applies text cleaning and normalization rules to raw extracted text.
    """

    def correct_text(self, text: str) -> tuple[str, bool]:
        """
        Corrects raw text string.

        Returns:
            (corrected_text: str, was_changed: bool)
        """
        if not text:
            return "", False

        original = text
        corrected = original

        # Rule 1: Character & Ligature Replacement
        for char, repl in CHAR_REPLACEMENTS.items():
            if char in corrected:
                corrected = corrected.replace(char, repl)

        # Rule 2: Join Hyphenated Words across line breaks ("gov-\nernment" -> "government")
        corrected = HYPHEN_LINEBREAK_REGEX.sub(r"\1\2", corrected)

        # Rule 3: UPSC Domain Abbreviation Normalization ("U.P.S.C." -> "UPSC")
        for regex, replacement in COMPILED_DOMAIN_TERMS:
            corrected = regex.sub(replacement, corrected)

        # Rule 4: Remove accidental duplicate words ("the the" -> "the")
        corrected = REPEATED_WORD_REGEX.sub(r"\1", corrected)

        # Rule 5: Whitespace Normalization
        corrected = MULTIPLE_SPACES_REGEX.sub(" ", corrected)
        corrected = MULTIPLE_NEWLINES_REGEX.sub("\n\n", corrected)

        corrected = corrected.strip()
        was_changed = (corrected != original.strip())

        return corrected, was_changed

    def correct_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies text corrections to a block dict in-place.
        Adds metadata fields:
          "was_corrected": bool
          "raw_text": str (saved if corrections were made)
        """
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
        """
        Applies text corrections across all blocks in a document list.
        """
        corrected_count = 0
        for block in blocks:
            self.correct_block(block)
            if block.get("was_corrected"):
                corrected_count += 1

        logger.info(f"ContentCorrector: Corrected {corrected_count}/{len(blocks)} blocks")
        return blocks


# ── 3. CONVENIENCE FUNCTION ───────────────────────────────────────────────────

def correct_extracted_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to apply content corrections to a list of block dicts.
    """
    corrector = ContentCorrector()
    return corrector.correct_document(blocks)
