import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
from extraction.extraction_validator import audit_extraction_coverage_and_quality

json_path = ROOT_DIR / "outputs" / "[NCERT] The Story of Civilization Part I (Arjun Dev) freeupscmaterials.org_extracted.json"
print("JSON File Path:", json_path)

if json_path.exists():
    report = audit_extraction_coverage_and_quality(json_path, 107)
    print("CURRENT REPORT:", json.dumps(report, indent=2))
