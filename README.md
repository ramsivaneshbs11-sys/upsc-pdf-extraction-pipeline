# UPSC RAG — Full Knowledge Ingestion & Vector Pipeline (`upsc-2`)

A production-grade, single-endpoint FastAPI application designed to register, validate, store, extract, preprocess, chunk, embed, and index PDF documents into Qdrant for a UPSC RAG (Retrieval-Augmented Generation) system.

---

## 🚀 Workflow Architecture

The single endpoint `POST /api/v1/documents` executes the complete pipeline sequentially:

```
PDF File + Classification (History / Anthropology)
                     │
                     ▼
             1. Validate Inputs
                     │
                     ▼
             2. Generate UUID
                     │
                     ▼
         3. Save PDF to uploads/<class>/
                     │
                     ▼
      4. Register in PostgreSQL (status=registered)
                     │
                     ▼
      5. Run Docling PDF Data Extraction (status=extracting -> extracted)
                     │
                     ▼
     6. Text Preprocessing & Layout Chunking (status=preprocessing -> preprocessed)
                     │
                     ▼
    7. BAAI/bge-base-en-v1.5 Embedding Generation (status=embedding)
                     │
                     ▼
   8. Qdrant Vector DB Ingestion per Classification (status=ingested)
```

---

## 📁 Repository Structure

```
upsc-2/
├── app/
│   ├── api/routes/documents.py  # Single endpoint POST /api/v1/documents
│   ├── core/config.py           # Configuration loader (.env & defaults)
│   ├── database/
│   │   ├── session.py           # SQLAlchemy connection setup
│   │   ├── models.py            # PostgreSQL Document DB model
│   │   └── repository.py        # Database CRUD helper functions
│   ├── services/
│   │   ├── storage_service.py      # PDF disk storage handler
│   │   ├── extraction_service.py   # Docling extraction & QA audit bridge
│   │   ├── preprocessing_service.py# Layout-aware chunking service
│   │   ├── embedding_service.py    # SentenceTransformer (BGE) vector generator
│   │   └── qdrant_service.py       # Qdrant collection manager & vector upsert
│   └── main.py                  # FastAPI application entry point
├── extraction/                  # Docling parsing modules & postprocessors
├── preprocessing/               # Cleaning & layout-aware chunking engine
├── docker-compose.yml           # Docker services for PostgreSQL and Qdrant
├── .env.example                 # Environment configuration template
└── requirements_api.txt         # API dependencies
```

---

## 🛠️ Quick Start Guide

### 1. Start Infrastructure via Docker Compose
Launch PostgreSQL and Qdrant containers in detached mode:

```bash
docker-compose up -d
```

- **PostgreSQL**: `localhost:5432` (`upsc_rag` database)
- **Qdrant Vector DB**: `localhost:6333` (Web UI available at `http://localhost:6333/dashboard`)

### 2. Set Up Environment Variables
Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Default `.env` configuration:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/upsc_rag
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
```

### 3. Install Python Dependencies

```bash
pip install -r requirements_api.txt python-dotenv
```

---

## 🏃 Running the Application

### Start the FastAPI Server

```bash
python -m uvicorn app.main:app --reload
```

- **Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Testing Document Ingestion

### Via Swagger UI (`/docs`)
1. Expand the `POST /api/v1/documents` endpoint.
2. Click **Try it out**.
3. Select a `.pdf` file in the `file` parameter.
4. Set `classification` to `History` or `Anthropology`.
5. Click **Execute**.

### Pipeline Status Transitions
The API tracks status transitions in PostgreSQL:
- `registered` → `extracting` → `extracted` → `preprocessing` → `preprocessed` → `embedding` → `ingested`

If any step fails, status is automatically set to `failed` with detailed log information.
