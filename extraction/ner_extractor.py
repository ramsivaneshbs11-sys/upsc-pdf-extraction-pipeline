"""
ner_extractor.py
──────────────────
UPSC domain-specific Named Entity Recognition (NER).

Categories Extracted:
  1. CONSTITUTIONAL_ARTICLE - Article numbers (e.g. "Article 370", "Art. 21A")
  2. ACT_POLICY            - Legislation & Policies (e.g. "Environment Protection Act 1986")
  3. GOVT_BODY             - Bodies & Committees (e.g. "NITI Aayog", "Finance Commission")
  4. HISTORICAL_EVENT      - Wars, Movements, Treaties (e.g. "Quit India Movement 1942")
  5. SCHEME               - Govt Welfare Schemes (e.g. "PM-KISAN", "MGNREGA")
  6. DATE_YEAR             - Historical Years & Specific Dates (e.g. "1947", "15th August 1947")

Output field added to block dicts:
  "entities": list of entity dicts [{"text": "...", "label": "...", "start": 0, "end": 10}, ...]
"""

import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger("ner_extractor")

# ── 1. UPSC DOMAIN ENTITY PATTERNS ───────────────────────────────────────────

NER_PATTERNS = [
    # Constitutional Articles & Amendments
    (
        "CONSTITUTIONAL_ARTICLE",
        r"(?i)\b(article|art\.)\s+\d+[a-z]?(\(\d+\))?|\b\d+(st|nd|rd|th)\s+constitutional\s+amendment(\s+act)?\b"
    ),

    # Acts, Legislation & Policies
    (
        "ACT_POLICY",
        r"(?i)\b([a-z\s]+)\s+act,?\s+\d{4}\b|\b([a-z\s]+)\s+policy,?\s+\d{4}\b|\bcode\s+of\s+[a-z\s]+\d{4}\b"
    ),

    # Government Bodies, Commissions & Committees
    (
        "GOVT_BODY",
        r"(?i)\b(niti\s+aayog|election\s+commission|finance\s+commission|upsc|spsc|cag|law\s+commission|national\s+green\s+tribunal|ngt|supreme\s+court|high\s+court|rbi|reserve\b[a-z\s]*bank)\b"
    ),

    # Government Schemes & Initiatives
    (
        "SCHEME",
        r"(?i)\b(pm-?[a-z]+|mgnrega|ayushman\s+bharat|poshan\s+abhiyaan|swachh\s+bharat|make\s+in\s+india|digital\s+india|atmanirbhar\s+bharat)\b"
    ),

    # Historical Events, Movements & Treaties
    (
        "HISTORICAL_EVENT",
        r"(?i)\b(battle\s+of\s+[a-z]+|quit\s+india\s+movement|non-?cooperation\s+movement|civil\s+disobedience\s+movement|swadeshi\s+movement|treaty\s+of\s+[a-z]+|revolt\s+of\s+1857|sepoy\s+mutiny)\b"
    ),

    # Historical Years (1000 AD to 2099 AD) & Dates
    (
        "DATE_YEAR",
        r"\b(1[0-9]{3}|20[0-2][0-9])\b|\b\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b"
    )
]

COMPILED_NER = [(label, re.compile(pattern)) for label, pattern in NER_PATTERNS]


# ── 2. NER EXTRACTOR CLASS ────────────────────────────────────────────────────

class UPSCNERExtractor:
    """
    Extracts UPSC-relevant domain entities from text strings.
    """

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts entities from text string.

        Returns:
            List of entity dicts:
            [{"text": "Article 21", "label": "CONSTITUTIONAL_ARTICLE", "start": 0, "end": 10}, ...]
        """
        if not text:
            return []

        entities = []
        seen_spans = set()

        for label, regex in COMPILED_NER:
            for match in regex.finditer(text):
                start, end = match.span()
                matched_text = match.group(0).strip()

                # Avoid overlapping spans
                if any(start < s_end and end > s_start for s_start, s_end in seen_spans):
                    continue

                seen_spans.add((start, end))
                entities.append({
                    "text": matched_text,
                    "label": label,
                    "start": start,
                    "end": end
                })

        # Sort entities by character position
        entities.sort(key=lambda x: x["start"])
        return entities

    def enrich_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches a block dict with an 'entities' field.
        """
        text = block.get("text", "")
        block["entities"] = self.extract_entities(text)
        return block

    def enrich_document(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enriches all blocks in a document list with NER entities.
        """
        total_entities = 0
        for block in blocks:
            self.enrich_block(block)
            total_entities += len(block.get("entities", []))

        logger.info(f"UPSCNERExtractor: Extracted {total_entities} entities across {len(blocks)} blocks")
        return blocks


# ── 3. CONVENIENCE FUNCTION ───────────────────────────────────────────────────

def enrich_blocks_with_ner(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience wrapper to apply NER entity extraction to a list of block dicts.
    """
    extractor = UPSCNERExtractor()
    return extractor.enrich_document(blocks)
