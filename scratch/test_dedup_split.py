import json
import re
from pathlib import Path

json_path = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

blocks = data.get("text_blocks", [])

p20_text = " ".join([b.get("text", "") for b in blocks if b.get("page_num") == 20])
p22_text = " ".join([b.get("text", "") for b in blocks if b.get("page_num") == 22])

print("P20 text sample:", p20_text[:120])
print("P22 text sample:", p22_text[:120])

w20 = set(re.sub(r"\s+", " ", p20_text).strip().lower().split())
w22 = set(re.sub(r"\s+", " ", p22_text).strip().lower().split())

inter = len(w20 & w22)
union = len(w20 | w22)
print(f"P20 vs P22 Jaccard word similarity: {inter/union:.3f}")

# Check Page 68 text
p68_blks = [b for b in blocks if b.get("page_num") == 68]
print(f"\nPage 68 block count: {len(p68_blks)}")
if p68_blks:
    print("Page 68 type:", p68_blks[0].get("type"))
    print("Page 68 text (first 300 chars):")
    print(repr(p68_blks[0].get("text")[:300]))
    print("Does text contain '\\n'?", "\n" in p68_blks[0].get("text"))
