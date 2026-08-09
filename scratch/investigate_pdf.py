import fitz
from pathlib import Path

pdf_path = Path(r"inputs/[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org.pdf")
doc = fitz.open(str(pdf_path))

print(f"PDF Total Pages: {len(doc)}")

# 1. Check pages 81, 85, 92 native text
for p_num in [81, 85, 92]:
    page = doc[p_num - 1]
    text = page.get_text()
    blocks = page.get_text("blocks")
    print(f"--- Page {p_num} ---")
    print(f"Text len: {len(text)}, blocks count: {len(blocks)}")
    print("Sample text:", text[:200].replace('\n', ' '))

# 2. Check page 68 layout & columns
p68 = doc[67]
print("\n--- Page 68 Blocks (fitz) ---")
for b in p68.get_text("blocks")[:10]:
    # (x0, y0, x1, y1, text, block_no, block_type)
    print(f"Bbox: ({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}) Text: {b[4][:60].replace('\n', ' ')}")

# 3. Check pages 20, 21 vs 22, 23
print("\n--- Page 20 vs 22 in PDF ---")
text_p20 = doc[19].get_text()
text_p22 = doc[21].get_text()
print(f"Page 20 text sample: {text_p20[:150].replace('\n', ' ')}")
print(f"Page 22 text sample: {text_p22[:150].replace('\n', ' ')}")
print(f"Are pages 20 and 22 identical in source PDF? {text_p20.strip() == text_p22.strip()}")

# 4. Check page 66-93 text vs RapidOCR vs Fitz
print("\n--- Page 66 text (fitz) ---")
print(doc[65].get_text()[:300].replace('\n', ' '))

doc.close()
