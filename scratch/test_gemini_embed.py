import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

key = os.environ.get("GEMINI_API_KEY")
from google import genai

client = genai.Client(api_key=key)

for m_name in ["gemini-embedding-001", "gemini-embedding-2", "text-embedding-004"]:
    try:
        res = client.models.embed_content(
            model=m_name,
            contents="Test Indus Valley Civilization embedding"
        )
        vec = res.embedding.values
        print(f"SUCCESS with model '{m_name}': vector len = {len(vec)}")
        break
    except Exception as exc:
        print(f"FAILED with model '{m_name}': {exc}")
