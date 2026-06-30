import os
import json
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-large-en"  # must match Embeddings.py exactly
INDEX_PATH = "data/processed/faiss_index.bin"
METADATA_PATH = "data/processed/metadata.json"

# BGE models expect this instruction prefix on the QUERY side only (not on
# documents). It measurably improves semantic match quality under hard
# paraphrasing — confirmed by evaluation showing MRR 36.6% / Top-5 50%.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None
_index = None
_metadata = None


def _load_resources():
    """Load model, index, and metadata once and cache them."""
    global _model, _index, _metadata

    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)

    if _index is None:
        print(f"Loading FAISS index from {INDEX_PATH}")
        _index = faiss.read_index(INDEX_PATH)
        print(f"Index loaded: {_index.ntotal} vectors")

    if _metadata is None:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        print(f"Metadata loaded: {len(_metadata)} entries")


def retrieve_chunks(query: str, k: int = 5) -> list:
    """
    Given a question, returns the top-k most relevant TeleQnA entries.
    Each result: {"id": ..., "text": ..., "source": ..., "score": ...}
    """
    _load_resources()

    instructed_query = QUERY_INSTRUCTION + query
    query_embedding = _model.encode([instructed_query], convert_to_numpy=True).astype("float32")

    distances, indices = _index.search(query_embedding, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        entry = _metadata[idx]
        results.append({
            "id": entry["id"],
            "text": entry["text"],
            "source": entry["source"],
            "score": float(dist),  # lower = more similar (L2 distance)
        })

    return results
#vchgtjgjgj

if __name__ == "__main__":
    test_query = "What is the handover procedure in 5G NR?"
    results = retrieve_chunks(test_query, k=3)

    print(f"\nQuery: {test_query}")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score={r['score']:.3f}) ---")
        print(r["text"][:200])
        print(f"Source: {r['source']}")