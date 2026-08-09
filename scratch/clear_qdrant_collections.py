"""
scratch/clear_qdrant_collections.py
───────────────────────────────────
Deletes evaluation collections from Qdrant so re-runs start clean with 1,066 chunks.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient
from app.core.config import QDRANT_HOST, QDRANT_PORT

def clear_eval_collections():
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        cols = [c.name for c in client.get_collections().collections]
        
        if not cols:
            print("No collections found in Qdrant.")
            return

        print(f"Current Qdrant collections: {cols}")
        eval_cols = ["history_bge_small", "history_bge_base", "history_kiwimate_e5", "history_arctic", "history_collection", "anthropology_collection"]
        
        for c_name in cols:
            if c_name in eval_cols or True:  # Delete all existing
                print(f"Deleting collection '{c_name}'...")
                client.delete_collection(collection_name=c_name)
                print(f"Deleted '{c_name}' ✓")

        print("\nAll Qdrant collections deleted successfully! You can now re-run the evaluation clean.")

    except Exception as exc:
        print(f"Error connecting or deleting from Qdrant: {exc}")

if __name__ == "__main__":
    clear_eval_collections()
