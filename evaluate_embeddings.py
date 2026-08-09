"""
evaluate_embeddings.py
───────────────────────
Evaluates candidate embedding models on preprocessed UPSC RAG chunks based on
the metrics defined in embedding_evaluation_metrics.md.

Target Embedding Models:
  1. BAAI/bge-base-en-v1.5 (BGE Base - 768 dim)
  2. BAAI/bge-small-en-v1.5 (BGE Small - 384 dim)
  3. intfloat/e5-base-v2 (E5 Base / KiwiMate - 768 dim)
  4. Snowflake/snowflake-arctic-embed-m (Snowflake Arctic Embed - 768 dim)

Usage:
  python evaluate_embeddings.py                 # Evaluates all candidate models
  python evaluate_embeddings.py --model bge_base # Evaluates ONLY a specific model
"""

import time
import math
import json
import logging
import argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import BASE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_embeddings")

# ── Candidate Embedding Models ────────────────────────────────────────────────
ALL_EVAL_MODELS = [
    {
        "id_key": "bge_base",
        "name": "BGE-Base (BAAI/bge-base-en-v1.5)",
        "model_id": "BAAI/bge-base-en-v1.5",
        "collection": "history_bge_base",
    },
    {
        "id_key": "bge_small",
        "name": "BGE-Small (BAAI/bge-small-en-v1.5)",
        "model_id": "BAAI/bge-small-en-v1.5",
        "collection": "history_bge_small",
    },
    {
        "id_key": "uae_large",
        "name": "UAE-Large-V1 (whereisAI/UAE-Large-V1)",
        "model_id": "whereisAI/UAE-Large-V1",
        "collection": "history_uae_large",
    },
    {
        "id_key": "gemini",
        "name": "Google Gemini (text-embedding-004)",
        "model_id": "text-embedding-004",
        "collection": "history_gemini",
    },
    {
        "id_key": "e5",
        "name": "E5-Base / KiwiMate (intfloat/e5-base-v2)",
        "model_id": "intfloat/e5-base-v2",
        "collection": "history_kiwimate_e5",
    },
    {
        "id_key": "arctic",
        "name": "Snowflake-Arctic (Snowflake/snowflake-arctic-embed-m)",
        "model_id": "Snowflake/snowflake-arctic-embed-m",
        "collection": "history_arctic",
    },
]

class GeminiEmbedder:
    """Wrapper class for Google Gemini text-embedding-004 model API."""
    def __init__(self, api_key: str | None = None):
        import os
        from google import genai
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required to evaluate Google Gemini text-embedding-004. "
                "Please set GEMINI_API_KEY in your environment or .env file."
            )
        self.client = genai.Client(api_key=key)

    def encode(self, texts, batch_size=16, show_progress_bar=False):
        import time
        import numpy as np

        def _get_vec(res):
            if hasattr(res, "embeddings") and res.embeddings:
                return res.embeddings[0].values
            elif hasattr(res, "embedding") and res.embedding:
                return res.embedding.values
            raise ValueError(f"Unknown Gemini response structure: {res}")

        if isinstance(texts, str):
            res = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts
            )
            return np.array(_get_vec(res))

        all_vecs = []
        for idx, text in enumerate(texts):
            if idx > 0 and idx % 90 == 0:
                logger.info(f"Pacing Gemini API requests ({idx}/{len(texts)})... sleeping 35s to respect 100 RPM free tier quota")
                time.sleep(35)

            for attempt in range(5):
                try:
                    res = self.client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=text
                    )
                    all_vecs.append(_get_vec(res))
                    time.sleep(0.65)  # Pace call speed to avoid bursts
                    break
                except Exception as exc:
                    if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                        wait_sec = 40
                        logger.warning(f"Gemini API 429 Rate Limit (Free Tier 100 RPM). Pausing {wait_sec}s (Attempt {attempt+1}/5)...")
                        time.sleep(wait_sec)
                    else:
                        logger.error(f"Error calling Gemini API: {exc}")
                        time.sleep(2)
                        if attempt == 4:
                            raise exc
        return np.array(all_vecs)

# ── Metrics Helper Functions ──────────────────────────────────────────────────

def compute_mrr(rankings: list[int]) -> float:
    """Calculate Mean Reciprocal Rank from list of 1-based ranks (0 if not in top K)."""
    rr_list = [1.0 / r if r > 0 else 0.0 for r in rankings]
    return float(np.mean(rr_list)) if rr_list else 0.0


def compute_ndcg_at_k(rankings: list[int], k: int = 5) -> float:
    """Calculate nDCG@K for binary relevance where ground truth has 1 relevant chunk."""
    dcg_list = []
    for r in rankings:
        if 0 < r <= k:
            dcg_list.append(1.0 / math.log2(r + 1))
        else:
            dcg_list.append(0.0)
    return float(np.mean(dcg_list)) if dcg_list else 0.0


def compute_recall_at_k(rankings: list[int], k: int) -> float:
    """Calculate Recall@K (percentage of queries where correct chunk is in top K)."""
    hits = [1.0 if 0 < r <= k else 0.0 for r in rankings]
    return float(np.mean(hits)) if hits else 0.0


# ── Main Benchmark Pipeline ───────────────────────────────────────────────────

def load_chunks_and_queries():
    """Load preprocessed chunks and build evaluation queries across all preprocessed data."""
    preprocessed_dir = BASE_DIR / "data" / "preprocessed"
    json_files = list(preprocessed_dir.glob("*.json"))

    chunks = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks.extend(data.get("chunks", []))

    if not chunks:
        # Synthetic fallback chunk for testing
        chunks = [
            {
                "chunk_id": "c1",
                "text": "The Indus Valley Civilization was a Bronze Age civilization in South Asia, extending from western Pakistan to northeastern India.",
                "metadata": {"section_name": "History"}
            },
            {
                "chunk_id": "c2",
                "text": "Anthropology is the scientific study of humanity, concerned with human behavior, human biology, cultures, societies, and linguistics.",
                "metadata": {"section_name": "Anthropology"}
            }
        ]

    # Dynamically select evaluation queries matched to actual chunk content
    test_queries = []
    seen_queries = set()

    for c in chunks:
        text = c.get("text", "")
        c_id = c.get("chunk_id")
        
        q = None
        if "Polity" in text and "Source" in text:
            q = "What are the primary literary and epigraphic sources for reconstructing Ancient Indian Polity?"
        elif "Kingship" in text or "king" in text.lower():
            q = "How is the theory of the origin of kingship described in ancient Indian political thought?"
        elif "Taxation" in text or "tax" in text.lower():
            q = "What was the structure of the taxation system in Ancient and Early Medieval India?"
        elif "Epic" in text or "Mahabharata" in text:
            q = "How is statecraft and political philosophy depicted in the Indian Epics?"
        elif "Svara or accent" in text or "Sanskrit language" in text:
            q = "What are the rules regarding Svara and accents in Vaidika Sanskrit?"
        elif "Dichotomy of Veda" in text or "divided into" in text:
            q = "How are the Vedas divided into Samhita, Brahmana, Aranyaka and Upanishad?"
        elif "Vedavy" in text or "heap" in text:
            q = "Who organized the initial heap of Vedic texts into structured Vedas?"
        elif "Vedic, Epic and Puranic culture" in text:
            q = "Overview of Vedic, Epic and Puranic culture of India"

        if q and q not in seen_queries:
            seen_queries.add(q)
            test_queries.append({"query": q, "expected_chunk_id": c_id})

    # Default fallback queries if none matched
    if not test_queries:
        test_queries = [
            {
                "query": "Where was the Indus Valley Civilization located?",
                "expected_chunk_id": chunks[0]["chunk_id"] if chunks else "c1"
            },
            {
                "query": "What is the scientific study of human cultures and biology?",
                "expected_chunk_id": chunks[1]["chunk_id"] if len(chunks) > 1 else "c1"
            }
        ]

    return chunks, test_queries


def evaluate_embedding_models(selected_model_key: str | None = None):
    chunks, eval_queries = load_chunks_and_queries()
    logger.info(f"Loaded {len(chunks)} chunks and {len(eval_queries)} evaluation queries.")

    # Filter target models if specific model requested
    if selected_model_key:
        target_models = [m for m in ALL_EVAL_MODELS if m["id_key"] == selected_model_key.lower()]
        if not target_models:
            logger.error(f"Unknown model key '{selected_model_key}'. Available keys: {[m['id_key'] for m in ALL_EVAL_MODELS]}")
            return
    else:
        target_models = ALL_EVAL_MODELS

    # Initialize fast in-memory Qdrant client to eliminate file lock conflicts
    client = QdrantClient(":memory:")

    results_table = []

    for model_info in target_models:
        name = model_info["name"]
        model_id = model_info["model_id"]
        collection_name = model_info["collection"]

        logger.info(f"\n==================================================")
        logger.info(f"Evaluating Model: {name} ({model_id})")
        logger.info(f"==================================================")

        try:
            # 1. Load Model
            start_load = time.time()
            if model_info.get("id_key") == "gemini":
                model = GeminiEmbedder()
            else:
                model = SentenceTransformer(model_id, trust_remote_code=True)
                if hasattr(model, "tokenizer") and model.tokenizer and getattr(model.tokenizer, "pad_token", None) is None:
                    model.tokenizer.pad_token = model.tokenizer.eos_token
            load_time = time.time() - start_load
            logger.info(f"Model loaded in {load_time:.2f} seconds.")

            # 2. Generate Embeddings & Measure Indexing Time
            start_index = time.time()
            id_k = model_info.get("id_key")
            if id_k == "e5":
                texts = ["passage: " + c["text"] for c in chunks]
            else:
                texts = [c["text"] for c in chunks]

            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)
            index_time = time.time() - start_index
            dimension = embeddings.shape[1]
            logger.info(f"Embedded {len(chunks)} chunks in {index_time:.2f}s (Dimension: {dimension}).")

            # 3. Create Qdrant Collection & Upsert
            existing_cols = {c.name for c in client.get_collections().collections}
            if collection_name in existing_cols:
                client.delete_collection(collection_name)

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )

            points = [
                PointStruct(
                    id=i + 1,
                    vector=embeddings[i].tolist(),
                    payload={"text": c["text"], "chunk_id": c["chunk_id"]}
                )
                for i, c in enumerate(chunks)
            ]
            client.upsert(collection_name=collection_name, points=points, wait=True)

            # 4. Evaluate Retrieval Metrics
            rankings = []
            latencies_ms = []

            for eq in eval_queries:
                raw_q = eq["query"]
                target_id = eq["expected_chunk_id"]

                if id_k == "e5":
                    q_text = "query: " + raw_q
                elif id_k in ["arctic", "uae_large"]:
                    q_text = "Represent this sentence for searching relevant passages: " + raw_q
                else:
                    q_text = raw_q

                t0 = time.time()
                q_vec = model.encode(q_text).tolist()

                search_res = client.query_points(
                    collection_name=collection_name,
                    query=q_vec,
                    limit=10,
                ).points
                latency = (time.time() - t0) * 1000.0  # in ms
                latencies_ms.append(latency)

                # Determine rank of ground truth chunk
                rank = 0
                for r_idx, pt in enumerate(search_res, start=1):
                    if pt.payload.get("chunk_id") == target_id or pt.id == target_id:
                        rank = r_idx
                        break
                rankings.append(rank)

            # Compute Summary Metrics
            rec5 = compute_recall_at_k(rankings, k=5)
            rec10 = compute_recall_at_k(rankings, k=10)
            mrr = compute_mrr(rankings)
            ndcg5 = compute_ndcg_at_k(rankings, k=5)
            avg_latency = float(np.mean(latencies_ms))

            metrics_row = {
                "Model": name,
                "Dimension": dimension,
                "Recall@5": f"{rec5 * 100:.1f}%",
                "Recall@10": f"{rec10 * 100:.1f}%",
                "MRR": f"{mrr:.3f}",
                "nDCG@5": f"{ndcg5:.3f}",
                "Avg Latency": f"{avg_latency:.2f} ms",
                "Index Time": f"{index_time:.2f} s",
            }
            results_table.append(metrics_row)

        except Exception as exc:
            logger.exception(f"Failed evaluation for model {name}: {exc}")
            results_table.append({
                "Model": name,
                "Dimension": "N/A",
                "Recall@5": "ERR",
                "Recall@10": "ERR",
                "MRR": "ERR",
                "nDCG@5": "ERR",
                "Avg Latency": "ERR",
                "Index Time": "ERR",
            })

    # ── Print Final Markdown Report ───────────────────────────────────────────
    print("\n\n" + "=" * 95)
    print("                 EMBEDDING EVALUATION METRICS BENCHMARK REPORT                 ")
    print("=" * 95 + "\n")

    header = f"| {'Model':<42} | {'Dim':<5} | {'Recall@5':<9} | {'Recall@10':<10} | {'MRR':<6} | {'nDCG@5':<7} | {'Avg Latency':<12} | {'Index Time':<10} |"
    divider = "|:" + "-"*42 + "|:" + "-"*5 + ":|:" + "-"*9 + ":|:" + "-"*10 + ":|:" + "-"*6 + ":|:" + "-"*7 + ":|:" + "-"*12 + ":|:" + "-"*10 + ":|"

    print(header)
    print(divider)
    for row in results_table:
        print(f"| {row['Model']:<42} | {str(row['Dimension']):<5} | {row['Recall@5']:<9} | {row['Recall@10']:<10} | {row['MRR']:<6} | {row['nDCG@5']:<7} | {row['Avg Latency']:<12} | {row['Index Time']:<10} |")

    print("\n" + "=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Embedding Models")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target model key to evaluate (e.g. bge_base, bge_small, e5, arctic)."
    )
    args = parser.parse_args()

    evaluate_embedding_models(selected_model_key=args.model)
