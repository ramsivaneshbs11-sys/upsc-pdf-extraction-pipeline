import sys
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from extraction.block_cleaner import _deduplicate_pages

json_path = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

blocks = data.get("text_blocks", [])

print(f"Initial block count: {len(blocks)}")
p_nums_before = set(b.get("page_num") for b in blocks)
print("Pages present before dedup:", len(p_nums_before))

deduped = _deduplicate_pages(blocks)
p_nums_after = set(b.get("page_num") for b in deduped)

print("Pages present after dedup:", len(p_nums_after))
print("Dropped pages:", sorted(list(p_nums_before - p_nums_after)))
