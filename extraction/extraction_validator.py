import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("extraction_validator")

_NER_SPOT_CHECKS = [
    ("Nicholas I", "Nicholas II", "Tsar name: 'I' vs 'II' - high-risk factual error"),
    ("Gandhi i", "Gandhi", "Corrupted proper noun: space-split OCR error"),
]
_GLYPH_TOKENS = {"headright", "boxshadowdwn"}
_WATERMARK_SINGLE = re.compile(r"^[0-9\-]$")
_WATERMARK_FOOTER = re.compile(r"^2019-?200?$")
_CORRUPTED_HEADING_RE = re.compile(
    r"Let'?s\s+(recal\s+l|imagin\s+e|discus\s+s|d\s+o|lear\s+n|thin\s+k)",
    re.IGNORECASE,
)

def validate_extracted_data(extracted_data, total_pdf_pages=None, run_contamination_check=True):
    blocks = extracted_data.get("text_blocks", [])
    tables = extracted_data.get("tables", [])
    report = {
        "overall_pass": True,
        "coverage": _check_coverage(blocks, total_pdf_pages),
        "contamination": _check_contamination(blocks) if run_contamination_check else {"skipped": True},
        "quality": _check_structural_quality(blocks),
        "ner_spotcheck": _check_ner_spotcheck(blocks),
    }
    report["overall_pass"] = report["coverage"].get("pass", True) and report["quality"].get("pass", True)
    return report

def _check_coverage(blocks, total_pdf_pages):
    if not blocks:
        return {"pass": False, "reason": "No text blocks extracted at all"}
    pages_with_text = set(b["page_num"] for b in blocks if b.get("text", "").strip())
    result = {"pages_with_text": len(pages_with_text)}
    if total_pdf_pages is not None:
        result["total_pdf_pages"] = total_pdf_pages
        result["coverage_pct"] = round(len(pages_with_text) / total_pdf_pages * 100, 1)
        zero_content_pages = total_pdf_pages - len(pages_with_text)
        result["zero_content_pages"] = zero_content_pages
        missing_pages = sorted(p for p in range(1, total_pdf_pages + 1) if p not in pages_with_text)
        result["missing_pages"] = missing_pages[:20]
        result["missing_page_count"] = len(missing_pages)
        result["pass"] = zero_content_pages <= 1
        if not result["pass"]:
            result["reason"] = f"{zero_content_pages} pages have no content"
    else:
        result["pass"] = True
    return result

def _check_contamination(blocks):
    text_to_pages = {}
    for b in blocks:
        text = b.get("text", "").strip()
        page = b.get("page_num")
        if len(text) > 40 and page:
            text_to_pages.setdefault(text[:120], set()).add(page)
    findings = []
    for text_snippet, pages in text_to_pages.items():
        if len(pages) >= 3:
            continue
        if len(pages) == 2:
            pages_sorted = sorted(pages)
            findings.append({"text_snippet": text_snippet[:100], "found_on_pages": pages_sorted,
                             "primary_page": pages_sorted[0], "also_on": pages_sorted[1:]})
    return {"pass": True, "warning": len(findings) > 0, "findings": len(findings), "examples": findings[:5]}

def _check_structural_quality(blocks):
    glyph_remaining, watermark_remaining, corrupted_headings = [], [], []
    for b in blocks:
        text = b.get("text", "").strip()
        words = set(text.lower().split())
        if words & _GLYPH_TOKENS:
            glyph_remaining.append({"block_id": b.get("block_id"), "text": text[:80]})
        if _WATERMARK_SINGLE.match(text) or _WATERMARK_FOOTER.match(text):
            watermark_remaining.append({"block_id": b.get("block_id"), "text": text})
        if b.get("type") == "heading" and _CORRUPTED_HEADING_RE.search(text):
            corrupted_headings.append({"block_id": b.get("block_id"), "text": text[:80]})
    passed = len(glyph_remaining) == 0 and len(watermark_remaining) == 0 and len(corrupted_headings) == 0
    return {"pass": passed, "glyph_tokens_remaining": len(glyph_remaining),
            "watermark_blocks_remaining": len(watermark_remaining),
            "corrupted_headings": len(corrupted_headings),
            "glyph_examples": glyph_remaining[:3], "watermark_examples": watermark_remaining[:3],
            "corrupted_heading_examples": corrupted_headings[:3]}

def _check_ner_spotcheck(blocks):
    warnings = []
    all_text = " ".join(b.get("text", "") for b in blocks)
    for wrong, expected, description in _NER_SPOT_CHECKS:
        if wrong in all_text and expected not in all_text:
            warnings.append({"found": wrong, "expected": expected, "description": description})
    return {"pass": True, "warnings": warnings}

def print_validation_report(report):
    sep = "-" * 60
    print(f"\n{sep}")
    print("  EXTRACTION VALIDATION REPORT")
    print(sep)
    cov = report.get("coverage", {})
    cont = report.get("contamination", {})
    qual = report.get("quality", {})
    ner = report.get("ner_spotcheck", {})
    print(f"  Overall: {'[PASS]' if report.get('overall_pass') else '[FAIL]'}")
    print(sep)
    print(f"\n[Coverage] {'[PASS]' if cov.get('pass', True) else '[FAIL]'}")
    if cov.get("total_pdf_pages"):
        print(f"  PDF pages        : {cov['total_pdf_pages']}")
        print(f"  Pages with text  : {cov['pages_with_text']} ({cov['coverage_pct']}%)")
        print(f"  Zero-content pages: {cov.get('zero_content_pages', 0)}")
    if not cont.get("skipped"):
        print(f"\n[Cross-page contamination] {'[WARN]' if cont.get('warning') else '[PASS]'}")
        print(f"  Findings: {cont.get('findings', 0)}")
    print(f"\n[Structural quality] {'[PASS]' if qual.get('pass', True) else '[FAIL]'}")
    print(f"  Glyph tokens remaining   : {qual.get('glyph_tokens_remaining', 0)}")
    print(f"  Watermark blocks remaining: {qual.get('watermark_blocks_remaining', 0)}")
    print(f"  Corrupted headings        : {qual.get('corrupted_headings', 0)}")
    ner_warns = ner.get("warnings", [])
    print(f"\n[Named-entity spot-check] {'[WARN]' if ner_warns else '[PASS]'}")
    for w in ner_warns:
        print(f"  [WARN] Found \"{w['found']}\" -- expected \"{w['expected']}\"")
    print(f"\n{sep}\n")
