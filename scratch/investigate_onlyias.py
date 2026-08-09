import sys
from pathlib import Path
import fitz

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

pdf_path = ROOT_DIR / "inputs" / "OnlyIAS_Post Independence India_Updated 2023 www.upscpdf.com.pdf"

print("Checking PDF existence:", pdf_path.exists())
if pdf_path.exists():
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    try:
        doc = fitz.open(str(pdf_path))
        print("Page count:", len(doc))
        print("Is encrypted:", doc.is_encrypted)
        print("Needs password:", doc.needs_pass)
        
        from extraction.document_validator import validate_pdf
        val_report = validate_pdf(pdf_path)
        print("\nValidator report:", val_report)

        from extraction.pdf_type_detector import is_scanned_pdf
        is_scanned = is_scanned_pdf(pdf_path)
        print(f"\nPDF Type Check: is_scanned = {is_scanned}")
        
        doc.close()
    except Exception as exc:
        print("Error during investigation:", exc)
