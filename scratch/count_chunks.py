import json
from pathlib import Path

prep_dir = Path("data/preprocessed")
input_pdfs = [
    "146001247033ET",
    "1504074018P12-M07-UrbanizationinNorthernIndia-ET",
    "1510564017P12-M20-Cholas-Banking,TaxationandCoinage-ET",
    "289-10-22-19-43-18",
    "290-10-22-0-57-21",
    "6 APPROACHES AND THEMES IN INDIAN HISTORIOGRAPHY-II",
    "Block-8",
    "THEME-II"
]

print("=" * 80)
print(f"{'INPUT PDF NAME':<55} | {'CHUNKS':<10}")
print("=" * 80)

total_chunks_8_pdfs = 0

for stem in input_pdfs:
    file_path = prep_dir / f"{stem}_extracted_preprocessed.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks = data.get("chunks", [])
            chunk_count = len(chunks)
            total_chunks_8_pdfs += chunk_count
            print(f"{stem:<55} | {chunk_count:<10}")
    else:
        print(f"{stem:<55} | NOT FOUND")

print("-" * 80)
print(f"TOTAL CHUNKS FOR 8 INPUT PDFs: {total_chunks_8_pdfs}")
print("=" * 80)

# Total overall chunks across all 24 files in data/preprocessed
all_files = list(prep_dir.glob("*.json"))
total_all = 0
for f in all_files:
    try:
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
            total_all += len(d.get("chunks", []))
    except Exception:
        pass

print(f"\nTotal Chunks across all {len(all_files)} files in data/preprocessed: {total_all}")
