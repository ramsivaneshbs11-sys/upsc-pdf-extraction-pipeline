import json
from pathlib import Path

json_path = Path(r"outputs/story_fixed/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")

if not json_path.exists():
    print("Output JSON does not exist at:", json_path)
    exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

blocks = data.get("text_blocks", [])
tables = data.get("tables", [])

print("=" * 60)
print(f"VERIFICATION REPORT FOR: {json_path.name}")
print("=" * 60)
print(f"Total extracted blocks: {len(blocks)}")
print(f"Total extracted tables: {len(tables)}")

# 1. Missing Pages Check (Pages 81, 85, 92)
pages_present = set(b.get("page_num") for b in blocks if isinstance(b.get("page_num"), int))
print(f"\n1. Page Coverage Check: {len(pages_present)} pages covered (min={min(pages_present)}, max={max(pages_present)})")
for p in [81, 85, 92]:
    p_blks = [b for b in blocks if b.get("page_num") == p]
    print(f"   - Page {p}: {len(p_blks)} blocks extracted")

# 2. Pages 66-93 Structural Breakdown & 2-column check
p66_93 = [b for b in blocks if 66 <= b.get("page_num", 0) <= 93]
types_66_93 = {}
for b in p66_93:
    t = b.get("type", "unknown")
    types_66_93[t] = types_66_93.get(t, 0) + 1
print(f"\n2. Pages 66-93 Structural Breakdown ({len(p66_93)} total blocks across 28 pages):")
for t, count in types_66_93.items():
    print(f"   - {t}: {count} blocks")

# 3. Page 68 Blocks
p68 = [b for b in blocks if b.get("page_num") == 68]
print(f"\n3. Page 68 Blocks ({len(p68)} blocks):")
for b in p68[:5]:
    print(f"   [{b.get('type')}] {b.get('text')[:100]}...")

# 4. Duplicate Page Check (Pages 20, 21, 22, 23)
for p in [20, 21, 22, 23]:
    p_blks = [b for b in blocks if b.get("page_num") == p]
    print(f"   - Page {p} blocks count: {len(p_blks)}")

# 5. Table Header Check
print(f"\n5. Table Header Check ({len(tables)} tables):")
for tbl in tables:
    print(f"   - Table ID: {tbl.get('table_id')}, Page: {tbl.get('page_num')}, Headers: {tbl.get('headers')}")

# 6. Character & OCR Correction Check
full_text = " ".join([b.get("text", "") for b in blocks])
print(f"\n6. OCR Error Check:")
print(f"   - Contains 'aud': {'aud' in full_text.split()}")
print(f"   - Contains 'Thiugs': {'Thiugs' in full_text}")
print(f"   - Contains 'Preachiug': {'Preachiug' in full_text}")
print(f"   - Contains 'Índia': {'Índia' in full_text}")
print("=" * 60)
