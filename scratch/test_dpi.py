import fitz
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from pathlib import Path

pdf_path = Path(r"inputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org.pdf")
doc = fitz.open(str(pdf_path))
ocr = RapidOCR()

out = []
for p_num in [81, 85, 92, 66, 68]:
    page = doc[p_num - 1]
    pix = page.get_pixmap(dpi=150)   # 150 DPI is fast and safe
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    res, _ = ocr(img)
    out.append(f"\n--- Page {p_num} (150 DPI) RapidOCR lines: {len(res) if res else 0} ---")
    if res:
        for line in res[:6]:
            out.append(f"  Text: {line[1]}")

doc.close()

print("\n".join(out))
