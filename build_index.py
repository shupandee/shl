"""
Run this once to pre-build the FAISS index before starting the server.
Usage: python build_index.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in .env first")

# Import build logic from main to keep a single source of truth
from main import build_faiss_index, _build_catalog_indexes

if __name__ == "__main__":
    _build_catalog_indexes()
    vs = build_faiss_index()
    print(f"Done! Index contains {vs.index.ntotal} vectors.")