"""
callout_detector.py
────────────────────
Fix 4: Detect and tag colored callout / highlight boxes in UPSC study material.

Problem
-------
Books like _Indian Art and Culture_ (Nitin Singhania) use light-pink shaded boxes
to mark curated "important facts" lists.  Docling's text layer carries only glyph
positions — it has no concept of the page's background fill color.  As a result,
callout boxes are extracted as ordinary ``paragraph`` / ``list_item`` blocks,
indistinguishable from regular body text.

Fix strategy
------------
Cross-reference each text block's ``bbox`` against the already-rendered page image
(produced by Docling when ``generate_page_images=True``) to sample the background
fill color behind the block.  When a non-white background is detected, two new
fields are added:

- ``highlight_type``: ``"pink_callout"`` | ``"yellow_callout"`` | ``"blue_callout"``
- ``box_id``:         a string like ``"box_p59_001"`` that is **shared** across all
                      consecutive same-color blocks on the same page, so the whole
                      logical callout can be retrieved/chunked as a unit.

Acceptance test
---------------
Re-run against ``Indian_Art_and_Culture_-_Nitin_Singhania_2nd_1_.pdf`` pages 59–60.
All 11 site bullets plus the intro paragraph should carry
``highlight_type: "pink_callout"`` and the same ``box_id``.
Surrounding body text should be untagged.

Dependency
----------
Requires ``Pillow`` and ``numpy``.  Both are typically present in the extraction
environment (numpy is already used by the OCR fallback path).  If either is missing
the module degrades gracefully — all blocks are returned unmodified.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("callout_detector")

# ── Optional dependency guard ─────────────────────────────────────────────────

try:
    from PIL import Image
    import numpy as np
    _IMAGING_AVAILABLE = True
except ImportError:
    _IMAGING_AVAILABLE = False
    logger.debug(
        "callout_detector: Pillow / numpy not available — "
        "Fix 4 (callout box tagging) will be skipped."
    )


# ── 1. COLOR SAMPLING ─────────────────────────────────────────────────────────

def get_box_background_color(
    page_image_path: Path,
    bbox: List[float],
    page_width_pts: float,
    page_height_pts: float,
    pad: int = 6,
) -> Optional[Tuple[int, int, int]]:
    """
    Sample the background fill color behind a text block's ``bbox``.

    Rather than sampling the text glyphs themselves (which would pick up ink
    pixels), we sample a thin padding strip **to the left** of the block's
    left edge — this strip is inside the shaded box but outside the glyph area.

    Args:
        page_image_path:  Absolute path to the rendered page PNG.
        bbox:             ``[x0, y0, x1, y1]`` in PDF points.
                          In Docling's coordinate system HIGH y = top of page.
        page_width_pts:   Page width in points (for pixel scaling).
        page_height_pts:  Page height in points (for pixel scaling).
        pad:              Width in pixels of the sampling strip.

    Returns:
        Median RGB as ``(r, g, b)`` tuple, or ``None`` if sampling failed.
    """
    if not _IMAGING_AVAILABLE:
        return None

    try:
        img = Image.open(page_image_path).convert("RGB")
        img_w, img_h = img.size

        if page_width_pts <= 0 or page_height_pts <= 0:
            return None

        sx = img_w / page_width_pts
        sy = img_h / page_height_pts

        x0_pt, x1_pt = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
        y_min_pt, y_max_pt = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])

        px_x0 = int(x0_pt * sx)
        px_y0 = max(0, int(img_h - y_max_pt * sy))  # top of block (smaller Y in PIL)
        px_y1 = min(img_h, int(img_h - y_min_pt * sy))  # bottom of block (larger Y in PIL)

        # Sample the left-padding strip inside the shaded box margin
        strip_x0 = max(0, px_x0 - 15)
        strip_x1 = max(0, px_x0 - 2)

        if strip_x1 <= strip_x0 or px_y1 <= px_y0:
            # Fallback: sample a thin interior strip near left edge
            strip_x0 = px_x0
            strip_x1 = min(px_x0 + pad, img_w)

        if strip_x1 <= strip_x0 or px_y1 <= px_y0:
            return None

        strip = img.crop((strip_x0, px_y0, strip_x1, px_y1))
        arr = np.array(strip).reshape(-1, 3)
        if len(arr) == 0:
            return None

        median_rgb = tuple(np.median(arr, axis=0).astype(int))
        return median_rgb  # type: ignore[return-value]

    except Exception as exc:
        logger.debug(f"get_box_background_color failed for {page_image_path}: {exc}")
        return None


# ── 2. COLOR CLASSIFICATION ───────────────────────────────────────────────────

def classify_highlight_color(
    rgb: Optional[Tuple[int, int, int]],
    whiteness_threshold: int = 230,
) -> Optional[str]:
    """
    Map a sampled RGB value to a semantic highlight label.

    Returns ``None`` for white / near-white backgrounds (no highlight).

    Color heuristics (tunable):
    - **pink_callout**:   R > 200, B > 150, R - G > 20, B - G > 10
    - **yellow_callout**: R > 200, G > 180, B < 150, R - B > 60
    - **blue_callout**:   B > 180, B - R > 30, B - G > 20
    """
    if rgb is None:
        return None

    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

    # Near-white → no highlight
    if r >= whiteness_threshold and g >= whiteness_threshold and b >= whiteness_threshold:
        return None

    if r > 200 and b > 150 and (r - g) > 20 and (b - g) > 10:
        return "pink_callout"

    if r > 200 and g > 180 and b < 150 and (r - b) > 60:
        return "yellow_callout"

    if b > 180 and (b - r) > 30 and (b - g) > 20:
        return "blue_callout"

    return None


# ── 3. PAGE-IMAGE LOOKUP ──────────────────────────────────────────────────────

def _build_page_image_index(
    page_images: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[int, Path]:
    """
    Returns a ``{page_num: absolute_path_to_page_image}`` mapping built from
    the ``page_images`` metadata list produced by :func:`save_page_images`.

    Only ``type == "page_render"`` entries are indexed (not embedded figures).
    """
    index: Dict[int, Path] = {}
    for entry in page_images:
        if entry.get("type") != "page_render":
            continue
        p_num = entry.get("page_num")
        rel_path = entry.get("path")
        if p_num is None or not rel_path:
            continue
        # "path" is stored relative to output_dir.parent
        abs_path = output_dir.parent / rel_path
        if abs_path.exists():
            index[p_num] = abs_path
    return index


# ── 4. PAGE DIMENSION LOOKUP ──────────────────────────────────────────────────

def _get_page_dimensions_from_fitz(pdf_path: Path) -> Dict[int, Tuple[float, float]]:
    """
    Returns ``{page_num: (width_pts, height_pts)}`` using PyMuPDF.
    Falls back to an empty dict if fitz is unavailable.
    """
    dims: Dict[int, Tuple[float, float]] = {}
    try:
        import fitz  # type: ignore
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc):
            dims[i + 1] = (page.rect.width, page.rect.height)
        doc.close()
    except Exception as exc:
        logger.debug(f"_get_page_dimensions_from_fitz failed: {exc}")
    return dims


# ── 5. MAIN TAGGER ────────────────────────────────────────────────────────────

def tag_callout_blocks(
    text_blocks: List[Dict[str, Any]],
    page_images: List[Dict[str, Any]],
    output_dir: Path,
    pdf_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Fix 4: Tags blocks that sit on a colored (non-white) background with
    ``highlight_type`` and a shared ``box_id`` across consecutive same-color
    blocks on the same page.

    This function is a **no-op** when:
    - Pillow or numpy are not installed.
    - ``page_images`` is empty (i.e. ``generate_page_images=False`` in config).

    Args:
        text_blocks:  Full document block list (after reordering and reindexing).
        page_images:  Metadata list from :func:`save_page_images`.
        output_dir:   Extraction output directory (used to resolve image paths).
        pdf_path:     Source PDF path (used for page dimension lookup via fitz).

    Returns:
        Block list with ``highlight_type`` and ``box_id`` fields added where
        applicable.  Blocks without a callout background are unchanged.
    """
    if not _IMAGING_AVAILABLE:
        logger.debug("tag_callout_blocks: skipped — Pillow/numpy not available.")
        return text_blocks

    page_render_index = _build_page_image_index(page_images, output_dir)
    if not page_render_index:
        logger.info(
            "tag_callout_blocks: no page renders available "
            "(generate_page_images=False?). Fix 4 skipped."
        )
        return text_blocks

    # Fetch page dimensions for coordinate scaling
    page_dims: Dict[int, Tuple[float, float]] = {}
    if pdf_path and pdf_path.exists():
        page_dims = _get_page_dimensions_from_fitz(pdf_path)

    tagged = 0
    box_counter: Dict[int, int] = {}       # page_num → box count on that page
    prev_highlight: Optional[str] = None   # highlight type of the previous block
    prev_page: Optional[int] = None        # page of the previous block
    current_box_id: Optional[str] = None   # box_id assigned to the current run

    for block in text_blocks:
        bbox = block.get("bbox")
        p_num = block.get("page_num", 1)

        # Blocks without a valid bbox (e.g. re-emitted TOC rows) can't be sampled
        if not bbox or len(bbox) != 4:
            prev_highlight = None
            prev_page = p_num
            current_box_id = None
            continue

        page_img_path = page_render_index.get(p_num)
        if not page_img_path:
            prev_highlight = None
            prev_page = p_num
            current_box_id = None
            continue

        w_pts, h_pts = page_dims.get(p_num, (0.0, 0.0))
        if w_pts <= 0 or h_pts <= 0:
            prev_highlight = None
            prev_page = p_num
            current_box_id = None
            continue

        rgb = get_box_background_color(page_img_path, bbox, w_pts, h_pts)
        highlight = classify_highlight_color(rgb)

        if highlight:
            # Continue existing box_id if same color on same page
            if highlight == prev_highlight and p_num == prev_page and current_box_id:
                block["highlight_type"] = highlight
                block["box_id"] = current_box_id
            else:
                # Start a new callout group
                box_counter[p_num] = box_counter.get(p_num, 0) + 1
                current_box_id = f"box_p{p_num}_{box_counter[p_num]:03d}"
                block["highlight_type"] = highlight
                block["box_id"] = current_box_id
            tagged += 1
            prev_highlight = highlight
        else:
            prev_highlight = None
            current_box_id = None

        prev_page = p_num

    if tagged:
        logger.info(
            f"tag_callout_blocks: tagged {tagged} block(s) across "
            f"{len(box_counter)} page(s) with callout highlight metadata."
        )

    return text_blocks
