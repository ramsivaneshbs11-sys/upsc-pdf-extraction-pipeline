import fitz
from pathlib import Path

pdf_path = Path(r"inputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org.pdf")
doc = fitz.open(str(pdf_path))

print(f"Total Pages: {len(doc)}")
page_stats = []

for i in range(len(doc)):
    page = doc[i]
    text = page.get_text().strip()
    images = page.get_images()
    page_stats.append((i+1, len(text), len(images)))

scanned_pages = [p for p, t_len, img_cnt in page_stats if t_len < 30]
digital_pages = [p for p, t_len, img_cnt in page_stats if t_len >= 30]

print(f"Digital pages count: {len(digital_pages)}")
print(f"Scanned / low-text pages count: {len(scanned_pages)}")
print(f"Scanned pages list: {scanned_pages}")

doc.close()
