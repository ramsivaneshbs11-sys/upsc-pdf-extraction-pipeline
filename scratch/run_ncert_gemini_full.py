"""
Full NCERT extraction via Gemini 3.5 Flash (Smart Router)
+ Accuracy comparison vs old Docling extraction (if available)
"""
import sys, json, logging, time, re, collections
from pathlib import Path

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ncert_extraction")

PDF_PATH   = Path("inputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org.pdf")
OUTPUT_DIR = Path("outputs")
JSON_OUT   = OUTPUT_DIR / "[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json"

print("=" * 65)
print("  NCERT — FULL GEMINI 3.5 FLASH EXTRACTION")
print(f"  PDF   : {PDF_PATH.name[:55]}")
print("=" * 65)

start_time = time.time()

# ── Run smart router extraction ───────────────────────────────────────────────
from extraction.smart_router import route_extraction
from extraction.json_builder import build_structured_json
from extraction.document_validator import validate_pdf, audit_extraction
from extraction.extraction_validator import audit_extraction_coverage_and_quality

val_report = validate_pdf(PDF_PATH)
doc, extracted_data = route_extraction(PDF_PATH, OUTPUT_DIR)

elapsed = time.time() - start_time
print(f"\nExtraction done in {elapsed:.1f}s")

# ── Save JSON ─────────────────────────────────────────────────────────────────
build_structured_json(extracted_data, PDF_PATH, val_report, JSON_OUT)
print(f"Saved: {JSON_OUT.name}")

# ── QA Audit ──────────────────────────────────────────────────────────────────
audit_results  = audit_extraction(JSON_OUT)
coverage_report = audit_extraction_coverage_and_quality(JSON_OUT, val_report["page_count"])
passed = sum(1 for v in audit_results.values() if v is True)

print(f"\nQA Score   : {passed}/9 rules passed")
print(f"Coverage   : {coverage_report['covered_pages_count']}/{val_report['page_count']} pages ({coverage_report['coverage_percentage']}%)")

# ── Accuracy Comparison Report ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ACCURACY COMPARISON: Docling (Old) vs Gemini 3.5 Flash (New)")
print("=" * 65)

blocks = extracted_data["text_blocks"]
type_counts = collections.Counter(b.get("type") for b in blocks)

# Docling baseline (from previous RapidOCR/Docling run — approximate known values)
DOCLING_BASELINE = {
    "total_blocks"   : 0,      # Was 0 — Docling produced NO output (all scanned pages)
    "coverage_pct"   : 0.0,    # 0% — all pages missed
    "headings"       : 0,
    "paragraphs"     : 0,
    "list_items"     : 0,
    "tables"         : 0,
    "qa_score"       : "N/A",  # Never ran successfully
}

gemini_stats = {
    "total_blocks"  : len(blocks),
    "coverage_pct"  : float(coverage_report["coverage_percentage"]),
    "headings"      : type_counts.get("heading", 0),
    "paragraphs"    : type_counts.get("paragraph", 0),
    "list_items"    : type_counts.get("list_item", 0),
    "tables"        : type_counts.get("table", 0),
    "pyq_questions" : type_counts.get("pyq_question", 0),
    "qa_score"      : f"{passed}/9",
    "elapsed_secs"  : round(elapsed, 1),
}

print(f"\n{'Metric':<28} {'Docling (Old)':>16} {'Gemini 3.5 Flash':>18}")
print("-" * 64)
print(f"{'Total text blocks':<28} {'0 (all failed)':>16} {gemini_stats['total_blocks']:>18,}")
print(f"{'Page coverage':<28} {'0%':>16} {gemini_stats['coverage_pct']:>17.1f}%")
print(f"{'Heading blocks':<28} {'0':>16} {gemini_stats['headings']:>18,}")
print(f"{'Paragraph blocks':<28} {'0':>16} {gemini_stats['paragraphs']:>18,}")
print(f"{'List item blocks':<28} {'0':>16} {gemini_stats['list_items']:>18,}")
print(f"{'Table blocks':<28} {'0':>16} {gemini_stats['tables']:>18,}")
print(f"{'PYQ question blocks':<28} {'0':>16} {gemini_stats['pyq_questions']:>18,}")
print(f"{'QA Score':<28} {'N/A (failed)':>16} {gemini_stats['qa_score']:>18}")
print(f"{'Processing time':<28} {'OOM crash':>16} {gemini_stats['elapsed_secs']:>15.1f}s")
print("-" * 64)

# Text quality sample
print("\n── Sample Extracted Content (Gemini 3.5 Flash) ──")
content_blocks = [b for b in blocks if not b.get("is_boilerplate") and len(b.get("text","")) > 30]
for b in content_blocks[5:12]:
    pg   = b.get("page_num", "?")
    typ  = b.get("type", "?")
    text = b.get("text", "")[:75].replace('\n', ' ')
    print(f"  [{typ:12s}] p{pg:03d}: {text!r}")

print(f"""
Summary:
  Docling result  : ❌ FAILED (0 pages extracted — scanned PDF)
  Gemini result   : ✅ {gemini_stats['total_blocks']} blocks, {gemini_stats['coverage_pct']}% page coverage
  Improvement     : ∞ (from 0 to {gemini_stats['total_blocks']} blocks)
  Ready for AI    : {'Yes' if gemini_stats['coverage_pct'] >= 90 else 'Partial — some pages may need review'}
""")
