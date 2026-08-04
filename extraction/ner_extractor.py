import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger("ner_extractor")

NER_PATTERNS = [
    ("CONSTITUTIONAL_ARTICLE", r"(?i)\b(article|art\.)\s+\d+[a-z]?(\(\d+\))?|\b\d+(st|nd|rd|th)\s+constitutional\s+amendment(\s+act)?\b"),
    ("ACT_POLICY", r"(?i)\b([a-z\s]+)\s+act,?\s+\d{4}\b|\b([a-z\s]+)\s+policy,?\s+\d{4}\b|\bcode\s+of\s+[a-z\s]+\d{4}\b"),
    ("GOVT_BODY", r"(?i)\b(niti\s+aayog|election\s+commission|finance\s+commission|upsc|spsc|cag|law\s+commission|national\s+green\s+tribunal|ngt|supreme\s+court|high\s+court|rbi|reserve\b[a-z\s]*bank)\b"),
    ("SCHEME", r"(?i)\b(pm-?[a-z]+|mgnrega|ayushman\s+bharat|poshan\s+abhiyaan|swachh\s+bharat|make\s+in\s+india|digital\s+india|atmanirbhar\s+bharat)\b"),
    ("HISTORICAL_EVENT", r"(?i)\b(battle\s+of\s+[a-z]+|quit\s+india\s+movement|non-?cooperation\s+movement|civil\s+disobedience\s+movement|swadeshi\s+movement|treaty\s+of\s+[a-z]+|revolt\s+of\s+1857|sepoy\s+mutiny)\b"),
    ("DATE_YEAR", r"\b(1[0-9]{3}|20[0-2][0-9])\b|\b\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b")
]

COMPILED_NER = [(label, re.compile(pattern)) for label, pattern in NER_PATTERNS]

class UPSCNERExtractor:
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if not text: return []
        entities, seen_spans = [], set()
        for label, regex in COMPILED_NER:
            for match in regex.finditer(text):
                start, end = match.span()
                matched_text = match.group(0).strip()
                if any(start < s_end and end > s_start for s_start, s_end in seen_spans): continue
                seen_spans.add((start, end))
                entities.append({"text": matched_text, "label": label, "start": start, "end": end})
        entities.sort(key=lambda x: x["start"])
        return entities

    def enrich_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        block["entities"] = self.extract_entities(block.get("text", ""))
        return block

    def enrich_document(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for block in blocks: self.enrich_block(block)
        return blocks

def enrich_blocks_with_ner(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return UPSCNERExtractor().enrich_document(blocks)
