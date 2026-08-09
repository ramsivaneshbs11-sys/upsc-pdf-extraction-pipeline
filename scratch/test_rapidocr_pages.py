import fitz
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from pathlib import Path

pdf_path = Path(r"inputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org.pdf")
doc = fitz.open(str(pdf_path))
ocr = RapidOCR()

# Test missing pages 81, 85, 92
print("=== MISSING PAGES (81, 85, 92) OCR TEST ===")
for p_num in [81, 85, 92]:
    page = doc[p_num - 1]
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    res, _ = ocr(img)
    print(f"\n--- Page {p_num} RapidOCR lines: {len(res) if res else 0} ---")
    if res:
        for line in res[:5]:
            print(f"  Box: {line[0]} Text: {line[1]}")

# Test corrupted pages 66, 68
print("\n=== CORRUPTED PAGES (66, 68) OCR TEST ===")
for p_num in [66, 68]:
    page = doc[p_num - 1]
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    res, _ = ocr(img)
    print(f"\n--- Page {p_num} RapidOCR lines: {len(res) if res else 0} ---")
    if res:
        for line in res[:8]:
            print(f"  Box: {line[0]} Text: {line[1]}")

# Test pages 20, 21, 22, 23
print("\n=== PAGES 20, 21, 22, 23 OCR TEST ===")
for p_num in [20, 21, 22, 23]:
    page = doc[p_num - 1]
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    res, _ = ocr(img)
    print(f"\n--- Page {p_num} RapidOCR lines: {len(res) if res else 0} ---")
    if res:
        print("First line:", res[0][1] if len(res) > 0 else "")
        print("Last line:", res[-1][1] if len(res) > 0 else "")

doc.close()
