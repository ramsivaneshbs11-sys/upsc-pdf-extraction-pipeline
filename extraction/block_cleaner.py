import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("block_cleaner")

_WATERMARK_SINGLE_CHAR_REGEX = re.compile(r"^[0-9\-]$")
_WATERMARK_FOOTER_REGEX = re.compile(r"^2019-?200?$")
_ICON_GLYPH_TOKENS = {"headright", "boxshadowdwn"}
_GLYPH_PREFIX_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(t) for t in ["headright", "boxshadowdwn"]) + r")\s+",
    re.IGNORECASE,
)
_STYLIZED_HEADING_FIXES: Dict[str, str] = {
    "Let's recal l": "Let's recall",
    "Let's recall l": "Let's recall",
    "Let's imagin e": "Let's imagine",
    "Let's imagine e": "Let's imagine",
    "Let's discus s": "Let's discuss",
    "di scus s": "Let's discuss",
    "Let's di scus s": "Let's discuss",
    "Let's d o": "Let's do",
    "Let's chang e": "Let's change",
    "Let's explor e": "Let's explore",
    "Let's fin d": "Let's find",
    "Let's lear n": "Let's learn",
    "Let's thin k": "Let's think",
}
_STYLIZED_HEADING_PATTERNS = [
    (re.compile(re.escape(bad), re.IGNORECASE), good)
    for bad, good in _STYLIZED_HEADING_FIXES.items()
]

def _is_watermark_block(block):
    text = block.get("text", "").strip()
    return bool(_WATERMARK_SINGLE_CHAR_REGEX.match(text) or _WATERMARK_FOOTER_REGEX.match(text))

def _is_icon_glyph_block(block):
    text = block.get("text", "").strip().lower()
    words = set(text.split())
    return bool(words and words.issubset(_ICON_GLYPH_TOKENS))

def _strip_glyph_prefix(block):
    text = block.get("text", "")
    new_text = text
    for token in _ICON_GLYPH_TOKENS:
        new_text = re.sub(r"(?i)\b" + re.escape(token) + r"\b", "", new_text)
    new_text = re.sub(r"  +", " ", new_text).strip()
    if new_text != text.strip():
        block["text"] = new_text
        block["was_cleaned"] = True
        return True
    return False

def _fix_stylized_headings(text):
    for pattern, replacement in _STYLIZED_HEADING_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def _deduplicate_headings(blocks):
    seen_headings = set()
    prev_page = None
    for block in blocks:
        if block.get("is_filtered") or block.get("type") != "heading":
            continue
        page_num = block.get("page_num")
        if page_num != prev_page:
            seen_headings.clear()
            prev_page = page_num
        heading_key = (page_num, block.get("text", "").strip())
        if heading_key in seen_headings:
            block["is_filtered"] = True
            block["filter_reason"] = "duplicate_heading"
        else:
            seen_headings.add(heading_key)
        prev_page = page_num
    return blocks

def _sort_blocks_by_reading_order(blocks):
    from itertools import groupby
    result = []
    indexed = list(enumerate(blocks))
    for page_num, page_group in groupby(indexed, key=lambda x: x[1].get("page_num", 0)):
        page_items = list(page_group)
        with_bbox = [(idx, blk) for idx, blk in page_items if blk.get("bbox")]
        without_bbox = [(idx, blk) for idx, blk in page_items if not blk.get("bbox")]
        def bbox_sort_key(item):
            bbox = item[1]["bbox"]
            top_y = max(bbox[1], bbox[3])
            left_x = bbox[0]
            return (-top_y, left_x)
        with_bbox_sorted = sorted(with_bbox, key=bbox_sort_key)
        page_sorted = [blk for _, blk in with_bbox_sorted] + [blk for _, blk in without_bbox]
        result.extend(page_sorted)
    return result

def _rejoin_split_captions(blocks):
    _SENTENCE_END = re.compile(r"[.?!]\s*$")
    VERTICAL_GAP_THRESHOLD = 15
    i = 0
    while i < len(blocks) - 1:
        curr = blocks[i]
        nxt = blocks[i + 1]
        if (curr.get("type") == "caption" and nxt.get("type") == "caption"
                and curr.get("page_num") == nxt.get("page_num")
                and not curr.get("is_filtered") and not nxt.get("is_filtered")
                and not _SENTENCE_END.search(curr.get("text", ""))):
            curr_bbox = curr.get("bbox")
            nxt_bbox = nxt.get("bbox")
            should_merge = False
            if curr_bbox and nxt_bbox:
                curr_bottom = min(curr_bbox[1], curr_bbox[3])
                nxt_top = max(nxt_bbox[1], nxt_bbox[3])
                vertical_gap = abs(nxt_top - curr_bottom)
                should_merge = vertical_gap <= VERTICAL_GAP_THRESHOLD
            else:
                should_merge = True
            if should_merge:
                curr["text"] = curr["text"].rstrip() + " " + nxt["text"].lstrip()
                curr["was_cleaned"] = True
                if curr_bbox and nxt_bbox:
                    curr["bbox"] = [min(curr_bbox[0], nxt_bbox[0]), min(curr_bbox[1], nxt_bbox[1]),
                                    max(curr_bbox[2], nxt_bbox[2]), max(curr_bbox[3], nxt_bbox[3])]
                nxt["is_filtered"] = True
                nxt["filter_reason"] = "caption_fragment_merged"
        i += 1
    return blocks

def _deduplicate_toc(blocks):
    _TOC_FRAGMENT_RE = re.compile(r"^\s*(\d+\.?|[ivxlcdmIVXLCDM]+\.?|.{1,40})\s*$")
    _PURE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")
    pages = {}
    for idx, block in enumerate(blocks):
        p = block.get("page_num", 0)
        pages.setdefault(p, []).append(idx)
    for page_num, idxs in pages.items():
        page_blocks = [blocks[i] for i in idxs if not blocks[i].get("is_filtered")]
        if not page_blocks:
            continue
        all_short = all(_TOC_FRAGMENT_RE.match(b.get("text", "")) for b in page_blocks)
        has_pure_number = any(_PURE_NUMBER_RE.match(b.get("text", "")) for b in page_blocks)
        if all_short and has_pure_number and len(page_blocks) >= 5:
            for i in idxs:
                b = blocks[i]
                if b.get("type") == "paragraph" and not b.get("is_filtered"):
                    b["is_filtered"] = True
                    b["filter_reason"] = "toc_fragment_duplicate"
    return blocks

def clean_blocks(blocks, sort_by_reading_order=True):
    watermark_count = 0
    glyph_filtered_count = 0
    glyph_stripped_count = 0
    heading_fix_count = 0
    for block in blocks:
        if block.get("is_filtered"):
            continue
        if _is_watermark_block(block):
            block["is_filtered"] = True
            block["filter_reason"] = "watermark_noise"
            watermark_count += 1
            continue
        if _is_icon_glyph_block(block):
            block["is_filtered"] = True
            block["filter_reason"] = "icon_glyph"
            glyph_filtered_count += 1
            continue
        if _strip_glyph_prefix(block):
            glyph_stripped_count += 1
        original_text = block.get("text", "")
        fixed_text = _fix_stylized_headings(original_text)
        if fixed_text != original_text:
            block["text"] = fixed_text
            block["was_cleaned"] = True
            heading_fix_count += 1
    blocks = _deduplicate_headings(blocks)
    blocks = _deduplicate_toc(blocks)
    blocks = _rejoin_split_captions(blocks)
    if sort_by_reading_order:
        blocks = _sort_blocks_by_reading_order(blocks)
    before_count = len(blocks)
    blocks = [b for b in blocks if not b.get("is_filtered")]
    filtered_count = before_count - len(blocks)
    logger.info(
        f"[block_cleaner] Removed {filtered_count} blocks total "
        f"({watermark_count} watermark, {glyph_filtered_count} glyph-filtered, "
        f"{filtered_count - watermark_count - glyph_filtered_count} other). "
        f"Cleaned {glyph_stripped_count} glyph-prefixed blocks. "
        f"Fixed {heading_fix_count} stylized headings."
    )
    return blocks
