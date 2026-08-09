"""
qa_check_all.py
───────────────
Deep QA check on all extracted + preprocessed outputs for P-01 and Semester-V.
Checks:
  1. Extraction JSON exists
  2. Preprocessing JSON exists
  3. JSON is valid (parseable)
  4. Has non-zero blocks
  5. Has non-zero content blocks
  6. No missing page coverage
  7. No empty blocks
  8. No invalid block types
  9. All blocks have is_boilerplate tag
  10. All blocks have NER entities field
  11. Preprocessed JSON has chunks
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

outputs = Path("outputs")
preprocess_out = Path("data/preprocessed")

p01_pdfs = list(Path("inputs/P-01").rglob("*.pdf"))
sem5_pdfs = list(Path("inputs/Semester-V").rglob("*.pdf"))

all_pdfs = [("P-01", p) for p in sorted(p01_pdfs)] + [("Semester-V", p) for p in sorted(sem5_pdfs)]

# Pipeline-generated structural block types (in addition to Docling baseline types)
VALID_TYPES = {
    "heading", "paragraph", "list_item", "caption", "footnote",
    "header", "footer", "table",
    "blank_page", "exercise_heading", "toc_page_number"
}

issues = []
ok_count = 0
stats = {}

for folder, pdf in all_pdfs:
    stem = pdf.stem.strip()
    json_file = outputs / stem / f"{stem}_extracted.json"
    pre_file = preprocess_out / f"{stem}_extracted_preprocessed.json"

    file_issues = []

    # Check 1: Extraction JSON exists
    if not json_file.exists():
        file_issues.append("EXTRACTION JSON MISSING")
        issues.append(f"  [{folder}] {pdf.name}: EXTRACTION JSON MISSING")
        continue

    # Check 2: Preprocessing JSON exists
    if not pre_file.exists():
        file_issues.append("PREPROCESSED JSON MISSING")

    # Check 3: JSON is parseable
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        file_issues.append(f"CORRUPT EXTRACTION JSON: {e}")
        issues.append(f"  [{folder}] {pdf.name}: " + " | ".join(file_issues))
        continue

    blocks = data.get("text_blocks", [])
    tables = data.get("tables", [])
    meta = data.get("document_metadata", {})
    summary = data.get("extraction_summary", {})

    page_count = meta.get("page_count", 0)
    total_blocks = len(blocks)
    content_blocks = summary.get("content_blocks", 0)
    boilerplate_blocks = summary.get("boilerplate_blocks", 0)

    # Check 4: Non-zero blocks
    if total_blocks == 0:
        file_issues.append("NO BLOCKS EXTRACTED")

    # Check 5: Non-zero content blocks
    if content_blocks == 0 and total_blocks > 0:
        file_issues.append("ALL BLOCKS ARE BOILERPLATE (0 content blocks)")

    # Check 6: Page coverage
    if page_count > 0:
        covered_pages = set(b.get("page_num") for b in blocks if isinstance(b.get("page_num"), int))
        missing_pages = set(range(1, page_count + 1)) - covered_pages
        if missing_pages:
            file_issues.append(f"MISSING PAGE COVERAGE: pages {sorted(missing_pages)[:10]}")

    # Check 7: Empty blocks
    # - blank_page markers intentionally have empty text (structural markers)
    # - boilerplate blocks (is_boilerplate=True) are filtered out in preprocessing — exempt them too
    empty_blocks = [
        b for b in blocks
        if b.get("type") != "blank_page"
        and not b.get("is_boilerplate", False)
        and not b.get("text", "").strip()
    ]
    if empty_blocks:
        file_issues.append(f"{len(empty_blocks)} EMPTY TEXT BLOCKS")

    # Check 8: Invalid block types
    invalid_type_blocks = [b for b in blocks if b.get("type") not in VALID_TYPES]
    if invalid_type_blocks:
        bad_types = set(b.get("type") for b in invalid_type_blocks)
        file_issues.append(f"{len(invalid_type_blocks)} INVALID BLOCK TYPES: {bad_types}")

    # Check 9: Missing is_boilerplate tag
    no_bp_tag = [b for b in blocks if "is_boilerplate" not in b]
    if no_bp_tag:
        file_issues.append(f"{len(no_bp_tag)} BLOCKS MISSING is_boilerplate TAG")

    # Check 10: Missing NER entities field
    no_ner = [b for b in blocks if "entities" not in b]
    if no_ner:
        file_issues.append(f"{len(no_ner)} BLOCKS MISSING entities FIELD")

    # Check 11: Preprocessed JSON has chunks
    if pre_file.exists():
        try:
            with open(pre_file, "r", encoding="utf-8") as f:
                pre_data = json.load(f)
            chunk_count = pre_data.get("chunk_count", 0)
            if chunk_count == 0:
                file_issues.append("PREPROCESSED JSON HAS 0 CHUNKS")
        except Exception as e:
            file_issues.append(f"CORRUPT PREPROCESSED JSON: {e}")

    if file_issues:
        for iss in file_issues:
            issues.append(f"  [{folder}] {pdf.name}: {iss}")
        stats[pdf.name] = {
            "folder": folder,
            "total_blocks": total_blocks,
            "content_blocks": content_blocks,
            "page_count": page_count,
            "issues": file_issues
        }
    else:
        ok_count += 1
        stats[pdf.name] = {
            "folder": folder,
            "total_blocks": total_blocks,
            "content_blocks": content_blocks,
            "page_count": page_count,
            "issues": []
        }

print("=" * 70)
print("  BATCH QA AUDIT RESULTS")
print("=" * 70)
print(f"  Total PDFs checked : {len(all_pdfs)}")
print(f"  Fully OK           : {ok_count}")
print(f"  With issues        : {len(all_pdfs) - ok_count}")
print("=" * 70)

if issues:
    print(f"\nISSUES FOUND ({len(issues)} total):\n")
    for iss in issues:
        print(iss)
else:
    print("\n  ALL OUTPUTS PASSED QA - No issues found!")

# Summary table
print("\n" + "=" * 70)
print(f"  {'PDF':<55} {'Blks':>5} {'Cont':>5} {'Pgs':>4}")
print("  " + "-" * 68)
for name, s in stats.items():
    flag = " [!]" if s["issues"] else ""
    short_name = name[:52] + "..." if len(name) > 52 else name
    print(f"  {short_name:<55} {s['total_blocks']:>5} {s['content_blocks']:>5} {s['page_count']:>4}{flag}")
print("=" * 70)
