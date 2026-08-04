# UPSC AI Study Platform

An intelligent PDF extraction, preprocessing, and RAG (Retrieval-Augmented Generation) pipeline designed for UPSC study materials — NCERT textbooks, ePathshala modules, and UPSC reference documents.

---

## 🚀 Features

- **Automatic PDF Type Detection** — Detects scanned vs digital PDFs instantly using PyMuPDF before running Docling
- **Hybrid Extraction Engine** — Docling as primary extractor + PyMuPDF fallback for any pages Docling misses (e.g. std::bad_alloc crashes) — guarantees 100% page coverage
- **Zero-Block OCR Retry** — If extraction returns no content, automatically retries with OCR enabled
- **7-Pass Block Cleaner** — Removes watermarks, icon-glyph garbage tokens, fixes cursive heading OCR corruption, deduplicates headings/TOC, rejoins split captions, and re-sorts blocks by PDF reading order
- **Runtime QA Validator** — Page coverage report, cross-page contamination detection, structural quality checks, and named-entity factual spot-checks
- **UPSC NER Enrichment** — Extracts and tags UPSC-relevant named entities (dates, acts, articles, historical figures)
- **Chunking & Preprocessing** — Structured JSON → cleaned chunks ready for vector embedding
- **RAM Crash Prevention** — Configurable Docling pipeline options to prevent `std::bad_alloc` on 100+ page PDFs

---

## 📁 Project Structure

```
upsc-pdf-extraction-pipeline/
├── app/                         # FastAPI application
│   ├── core/                    # Configuration & database
│   ├── models/                  # SQLAlchemy models
│   ├── routes/                  # API endpoints
│   └── services/                # Business logic services
│
├── extraction/                  # PDF Extraction Pipeline
│   ├── pdf_type_detector.py     # Auto-detect scanned vs digital PDF
│   ├── docling_extractor.py     # Core Docling extraction engine + hybrid fallback
│   ├── block_cleaner.py         # 7-pass post-extraction cleanup
│   ├── extraction_validator.py  # Runtime QA validation report
│   ├── boilerplate_detector.py  # Header/footer/page-number detection
│   ├── content_corrector.py     # Hyphen, ligature & abbreviation fixes
│   ├── ner_extractor.py         # UPSC Named Entity Recognition
│   ├── document_validator.py    # Input PDF validation
│   ├── json_builder.py          # Structured JSON output builder
│   ├── models.py                # Data models
│   └── config.py                # Pipeline configuration
│
├── preprocessing/               # Chunking & Preprocessing Pipeline
│   ├── text_cleaner.py          # Text normalization
│   ├── chunker.py               # Semantic chunking
│   └── ...
│
├── run_pipeline.py              # CLI: Run full extraction + preprocessing
├── requirements_api.txt         # API dependencies
├── requirements_extraction.txt  # Extraction pipeline dependencies
└── requirements_preprocessing.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ramsivaneshbs11-sys/upsc-pdf-extraction-pipeline.git
cd upsc-pdf-extraction-pipeline
```

### 2. Install dependencies
```bash
# For the extraction pipeline
pip install -r requirements_extraction.txt

# For the API server
pip install -r requirements_api.txt
```

### 3. Run the extraction pipeline
```bash
# Place PDF files in the inputs/ directory, then:
python run_pipeline.py inputs/
```

### 4. Start the API server
```bash
uvicorn app.main:app --reload
```

---

## 🔄 Extraction Pipeline Flow

```
PDF Upload
    │
    ▼
[Step 0] Auto PDF Type Detection (PyMuPDF)
    │   → SCANNED?  OCR = ON
    │   → DIGITAL?  OCR = OFF (fast)
    ▼
[Step 1] Docling Extraction (Layout parsing, headings, paragraphs, tables)
    ▼
[Step 2] Hybrid PyMuPDF Fallback (fills pages Docling missed)
    ▼
[Step 3] Zero-Block OCR Retry (if 0 blocks extracted)
    ▼
[Step 4] Post-Processing
    │   → Boilerplate detection
    │   → Text correction (hyphens, ligatures)
    │   → UPSC NER enrichment
    ▼
[Step 5] Block Cleaner (7 passes)
    │   → Strip watermarks
    │   → Remove glyph garbage (headright, boxshadowdwn)
    │   → Fix cursive heading OCR (Let's recal l → Let's recall)
    │   → Deduplicate headings & TOC
    │   → Rejoin split captions
    │   → Sort by reading order (PDF bbox coordinates)
    ▼
[Step 6] QA Validation Report
    │   → Page coverage check
    │   → Cross-page contamination
    │   → Structural quality audit
    │   → NER factual spot-check
    ▼
[Step 7] Structured JSON Output → Ready for RAG / Vector DB
```

---

## 📊 Extraction Quality Results

| Metric | Before | After |
|---|---|---|
| Page Coverage (NCERT 152 pages) | 27.6% (42 pages) | **~100%** (151-152 pages) |
| Watermark noise blocks | High | **0** (fully filtered) |
| Glyph garbage tokens | Present | **0** (filtered/stripped) |
| OCR heading corruption | Present | **Fixed** (lookup table) |
| QA Audit Score | N/A | **9/9 rules** |
| Server crashes (std::bad_alloc) | Frequent | **None** (RAM-safe config) |

---

## 🛠️ Configuration

Edit `extraction/config.py` to tune pipeline options:

```python
DOCLING_PIPELINE_OPTIONS = {
    "do_ocr": False,              # Overridden dynamically by pdf_type_detector
    "do_table_structure": False,  # Prevents std::bad_alloc on large PDFs
    "generate_page_images": False,
    "generate_table_images": False,
}
```

---

## 📋 Supported Document Types

- NCERT Textbooks (Class 6–12, all subjects)
- ePathshala Modules (NIOS)
- UPSC Reference Books
- Any digital or scanned PDF

---

## 🤝 Contributing

Pull requests welcome. For major changes, please open an issue first.

---

## 📄 License

MIT License
