"""
postprocessor.py
──────────────────
Consolidated post-processor combining Boilerplate Detection, Text Correction, and NER.
"""

from typing import List, Dict, Any
from extraction.boilerplate_detector import tag_boilerplate_blocks
from extraction.content_corrector import correct_extracted_blocks
from extraction.ner_extractor import enrich_blocks_with_ner


def postprocess_extracted_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies full 3-step post-processing to text blocks:
      1. Boilerplate tagging
      2. Content correction
      3. Named Entity Recognition (NER)
    """
    blocks = tag_boilerplate_blocks(blocks)
    blocks = correct_extracted_blocks(blocks)
    blocks = enrich_blocks_with_ner(blocks)
    return blocks
