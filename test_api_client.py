import sys
import requests
from pathlib import Path

API_URL = "http://localhost:8000/api/v1/documents"

def test_upload():
    workspace_dir = Path(__file__).resolve().parent
    pdf_files = list(workspace_dir.rglob("*.pdf"))
    if not pdf_files:
        print("[ERROR] No PDF files found in workspace.")
        sys.exit(1)

    test_pdf = pdf_files[0]
    print(f"[INFO] Found test PDF: {test_pdf}")
    classification = "History"
    with open(test_pdf, "rb") as f:
        files = {"file": (test_pdf.name, f, "application/pdf")}
        data = {"classification": classification}
        try:
            response = requests.post(API_URL, files=files, data=data)
            print(f"[INFO] Status Code: {response.status_code}")
            if response.status_code in [200, 201]:
                import json
                print("[SUCCESS] API responded successfully!")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"[FAIL] Error response: {response.text}")
        except Exception as e:
            print(f"[ERROR] Failed to connect to server: {e}")

if __name__ == "__main__":
    test_upload()
