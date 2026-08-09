import json
from pathlib import Path

json_path = Path(r"outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")
if not json_path.exists():
    print("File not found:", json_path)
    exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

blocks = data.get("text_blocks", [])
tables = data.get("tables", [])

print(f"Total blocks: {len(blocks)}, Total tables: {len(tables)}")

# Check pages present
pages_present = set(b.get("page_num") for b in blocks if isinstance(b.get("page_num"), int))
print(f"Pages present: {min(pages_present)} to {max(pages_present)}, count: {len(pages_present)}")

missing = set(range(1, max(pages_present)+1)) - pages_present
print(f"Missing pages in JSON: {sorted(list(missing))}")

# Check pages 66 to 93
p66_93_blocks = [b for b in blocks if 66 <= b.get("page_num", 0) <= 93]
print(f"Number of blocks for pages 66-93: {len(p66_93_blocks)}")
p66_93_by_page = {}
for b in p66_93_blocks:
    p = b.get("page_num")
    p66_93_by_page.setdefault(p, []).append(b)

for p in range(66, 94):
    p_blks = p66_93_by_page.get(p, [])
    types = [b.get("type") for b in p_blks]
    print(f"Page {p}: {len(p_blks)} blocks, types: {types}")

# Check pages 20, 21, 22, 23
for p in [20, 21, 22, 23]:
    p_blks = [b for b in blocks if b.get("page_num") == p]
    txt = " ".join([b.get("text", "") for b in p_blks[:2]])
    print(f"Page {p} ({len(p_blks)} blocks): {txt[:100]}...")

# Check table on page 7
p7_tables = [t for t in tables if t.get("page_num") == 7]
p7_blocks = [b for b in blocks if b.get("page_num") == 7]
print(f"Page 7 tables: {len(p7_tables)}, blocks: {len(p7_blocks)}")
if p7_tables:
    print("Page 7 table headers:", p7_tables[0].get("headers"))
    print("Page 7 table rows sample:", p7_tables[0].get("rows")[:2])
