import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

key = os.environ.get("GEMINI_API_KEY")
from google import genai

client = genai.Client(api_key=key)
res = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Test Indus Valley Civilization embedding"
)

print("Attributes of EmbedContentResponse:")
print(dir(res))
if hasattr(res, "embeddings") and res.embeddings:
    vec = res.embeddings[0].values
    print(f"\nSUCCESS! Vector dimension = {len(vec)}")
