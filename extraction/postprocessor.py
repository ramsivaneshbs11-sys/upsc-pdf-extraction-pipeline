from typing import List, Dict, Any
from extraction.boilerplate_detector import tag_boilerplate_blocks
from extraction.content_corrector import correct_extracted_blocks
from extraction.ner_extractor import enrich_blocks_with_ner

def postprocess_extracted_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks = tag_boilerplate_blocks(blocks)
    blocks = correct_extracted_blocks(blocks)
    blocks = enrich_blocks_with_ner(blocks)
    return blocks
