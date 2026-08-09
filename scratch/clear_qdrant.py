"""
scratch/clear_qdrant.py
───────────────────────
Script to wipe all collections and floating-point vector embeddings from Qdrant.
"""
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient
from app.core.config import QDRANT_HOST, QDRANT_PORT

def clear_all_qdrant_embeddings():
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections_response = client.get_collections()
        collections = [col.name for col in collections_response.collections]

        if not collections:
            print("No Qdrant collections found. Database is already clean.")
            return

        print(f"Found {len(collections)} collections: {collections}")
        for col_name in collections:
            print(f"Deleting collection '{col_name}'...")
            client.delete_collection(collection_name=col_name)
            print(f"Deleted '{col_name}' ✓")

        print("\nAll Qdrant collections and vector embeddings have been successfully removed.")

    except Exception as exc:
        print(f"Error connecting or deleting from Qdrant: {exc}")

if __name__ == "__main__":
    clear_all_qdrant_embeddings()
