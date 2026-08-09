import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

key = os.environ.get("GEMINI_API_KEY")
print("Key loaded:", key[:10] + "..." if key else "None")

try:
    from google import genai
    client = genai.Client(api_key=key)
    print("Listing available models for this API key...")
    models = list(client.models.list())
    for m in models:
        if "embed" in m.name.lower():
            print(f" - {m.name} ({getattr(m, 'supported_generation_methods', '')})")
except Exception as exc:
    print(f"Error listing models: {exc}")
