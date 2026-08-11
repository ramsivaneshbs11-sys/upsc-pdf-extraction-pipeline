"""
Healer script to re-extract and patch failed pages in the NCERT JSON.
Only runs Gemini for pages marked with 'gemini_flash_failed'.
"""
import sys, json, logging, time
from pathlib import Path

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("extraction_healer")

JSON_PATH = Path("outputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json")
PDF_PATH  = Path("inputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org.pdf")

if not JSON_PATH.exists():
    logger.error(f"JSON file not found: {JSON_PATH}")
    sys.exit(1)

with open(JSON_PATH, encoding='utf-8') as f:
    data = json.load(f)

blocks = data.get('text_blocks', [])
failed_pages = [54, 79]
# Clean up any existing placeholders for these pages in the original json
blocks = [b for b in blocks if b.get('page_num') not in failed_pages]
data['text_blocks'] = blocks


logger.info(f"Found {len(failed_pages)} failed pages: {failed_pages}")

# ── Re-extract failed pages using smart_router with a safe cooldown ────────────
from extraction.smart_router import route_extraction

# Temp output directory for healer
temp_dir = Path("outputs/healer_temp")
temp_dir.mkdir(parents=True, exist_ok=True)

patched_blocks = []
from extraction.gemini_extractor import extract_with_gemini_flash

# Process failed pages one by one with safe 10-second gap
for page_num in failed_pages:
    logger.info(f"Re-extracting page {page_num}...")
    # Delay to avoid triggering project quota
    time.sleep(10)
    try:
        _, page_data = extract_with_gemini_flash(
            PDF_PATH,
            temp_dir,
            page_delay_secs=10.0,
            start_page=page_num,
            end_page=page_num
        )
        page_blocks = page_data.get('text_blocks', [])
        # Ensure we didn't fail again
        if page_blocks and page_blocks[0].get('source') != 'gemini_flash_failed':
            patched_blocks.extend(page_blocks)
            logger.info(f"  Successfully recovered page {page_num} ({len(page_blocks)} blocks)")
        else:
            logger.warning(f"  Page {page_num} recovery returned fallback block.")
    except Exception as e:
        logger.error(f"  Failed to recover page {page_num}: {e}")

if not patched_blocks:
    logger.error("No pages could be recovered. Quota limit might still be active.")
    sys.exit(1)

# ── Patch the original JSON ──────────────────────────────────────────────────
# Remove the old failed placeholders
cleaned_blocks = [b for b in blocks if b.get('source') != 'gemini_flash_failed']

# Add the new recovered blocks
all_blocks = cleaned_blocks + patched_blocks

# Sort blocks by page_num then block ID to maintain order
all_blocks.sort(key=lambda x: (x.get('page_num', 0), x.get('block_id', '')))

# Re-assign sequential block IDs
for idx, b in enumerate(all_blocks, 1):
    b['block_id'] = f"blk_{idx:04d}"

data['text_blocks'] = all_blocks
data['extraction_summary']['total_blocks'] = len(all_blocks)
data['extraction_summary']['content_blocks'] = len([b for b in all_blocks if not b.get('is_boilerplate')])

with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

logger.info(f"Patched JSON saved. New total blocks: {len(all_blocks)}")

# ── Re-run audit ─────────────────────────────────────────────────────────────
from extraction.document_validator import audit_extraction
from extraction.extraction_validator import audit_extraction_coverage_and_quality

audit_results = audit_extraction(JSON_PATH)
coverage_report = audit_extraction_coverage_and_quality(JSON_PATH, data['document_metadata']['page_count'])
passed = sum(1 for v in audit_results.values() if v is True)

print("\n" + "="*50)
print("  HEALER RUN COMPLETE")
print(f"  New Coverage: {coverage_report['covered_pages_count']}/{data['document_metadata']['page_count']} pages ({coverage_report['coverage_percentage']}%)")
print(f"  QA Passed   : {passed}/9 rules")
print("="*50)
