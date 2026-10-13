from . import config
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


FAISS_INDEX_PATH = str(config.path('vectorstore/faiss_index.index'))
METADATA_PATH = str(config.path('vectorstore/chunk_metadata.json'))

MODEL_NAME = config.EMBED_MODEL
TOP_K = 5


def load_index():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def embed_query(query, model):
    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding.astype("float32")


def search(query):
    print(f"\nQuery: {query}")

    index, metadata = load_index()

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = embed_query(query, model)

    scores, indices = index.search(query_embedding, TOP_K)

    print("\nTop Retrieved Chunks:\n")

    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx]

        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Score: {scores[0][rank-1]:.4f}")
        print(f"PMCID: {chunk['pmcid']}")
        print(f"Title: {chunk['title']}")
        print(f"Section: {chunk['section']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("\nChunk Text Preview:")
        print(chunk["chunk_text"][:1000])
        print("=" * 80)


if __name__ == "__main__":
    query = input("Enter your medical research query: ")
    search(query)