import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

key = os.environ.get("GEMINI_API_KEY")
from google import genai

client = genai.Client(api_key=key)

test_batch = ["Text chunk 1", "Text chunk 2", "Text chunk 3"]

try:
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=test_batch
    )
    print("Batch call response type:", type(res))
    if hasattr(res, "embeddings"):
        vecs = [e.values for e in res.embeddings]
        print(f"SUCCESS! Got {len(vecs)} vectors, dimension = {len(vecs[0])}")
except Exception as exc:
    print(f"Batch call failed: {exc}")
