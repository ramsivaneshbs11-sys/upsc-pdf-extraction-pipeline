"""
extraction_validator.py
─────────────────────────
Quality Assurance & Audit validator to verify:
  1. 100% Page Coverage (no missing pages)
  2. Cross-page text contamination (detecting blocks repeated across different pages)
  3. Quality & completeness audit metrics
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set

logger = logging.getLogger("extraction_validator")


def audit_extraction_coverage_and_quality(json_path: Path, expected_page_count: int) -> Dict[str, Any]:
    """
    Performs runtime validation on extracted JSON to guarantee 100% page coverage and check quality.
    """
    report = {
        "expected_pages": expected_page_count,
        "covered_pages_count": 0,
        "missing_pages": [],
        "coverage_percentage": 0.0,
        "cross_page_contamination_found": False,
        "contaminated_blocks_count": 0,
        "is_coverage_100_percent": False
    }

    if not json_path.exists():
        logger.error(f"Extraction validator error: File not found {json_path}")
        return report

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        blocks = data.get("text_blocks", [])
        tables = data.get("tables", [])
        covered_pages: Set[int] = {b.get("page_num") for b in blocks if isinstance(b.get("page_num"), int)}
        covered_pages.update({t.get("page_num") for t in tables if isinstance(t.get("page_num"), int)})

        all_expected = set(range(1, expected_page_count + 1))
        missing = sorted(list(all_expected - covered_pages))

        report["covered_pages_count"] = len(covered_pages)
        report["missing_pages"] = missing
        report["coverage_percentage"] = round((len(covered_pages) / expected_page_count * 100), 2) if expected_page_count > 0 else 0.0
        report["is_coverage_100_percent"] = len(missing) == 0

        # Cross-page contamination check
        text_page_map: Dict[str, Set[int]] = {}
        for b in blocks:
            txt = b.get("text", "").strip()
            p = b.get("page_num")
            # Only check long sentences (>40 chars) to avoid common headers/footers
            if len(txt) > 40 and p:
                text_page_map.setdefault(txt, set()).add(p)

        cross_page_dups = {txt: pages for txt, pages in text_page_map.items() if len(pages) > 1}
        if cross_page_dups:
            report["cross_page_contamination_found"] = True
            report["contaminated_blocks_count"] = len(cross_page_dups)
            logger.warning(f"Cross-page contamination detected in {len(cross_page_dups)} blocks!")

        logger.info(
            f"Extraction Coverage Audit: {report['covered_pages_count']}/{expected_page_count} pages covered "
            f"({report['coverage_percentage']}%) | 100% Coverage: {report['is_coverage_100_percent']}"
        )
    except Exception as e:
        logger.error(f"Failed to run extraction validator on {json_path}: {e}")

    return report
