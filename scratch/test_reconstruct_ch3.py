import fitz
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from pathlib import Path

pdf_path = Path(r"inputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org.pdf")
doc = fitz.open(str(pdf_path))
ocr = RapidOCR()

page = doc[67] # Page 68
pix = page.get_pixmap(dpi=150)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
res, _ = ocr(img)

page_width = pix.width
x_mid = page_width / 2.0

col1_lines = []
col2_lines = []
header_lines = []
footer_lines = []

for item in res:
    box, text, score = item[0], item[1].strip(), item[2]
    x_center = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4.0
    y_center = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4.0

    if y_center < pix.height * 0.10:
        header_lines.append((y_center, text))
    elif y_center > pix.height * 0.90:
        footer_lines.append((y_center, text))
    elif x_center < x_mid:
        col1_lines.append((y_center, text))
    else:
        col2_lines.append((y_center, text))

col1_lines.sort(key=lambda x: x[0])
col2_lines.sort(key=lambda x: x[0])

print("--- PAGE 68 RECONSTRUCTED ---")
print("Headers:", [t for y, t in header_lines])
print("\nColumn 1 Text:")
print(" ".join([t for y, t in col1_lines]))
print("\nColumn 2 Text:")
print(" ".join([t for y, t in col2_lines]))
print("Footers:", [t for y, t in footer_lines])

doc.close()
