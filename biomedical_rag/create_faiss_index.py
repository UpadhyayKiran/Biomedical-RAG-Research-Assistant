from . import config
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================
CHUNKS_JSONL = str(config.path('chunks/chunks.jsonl'))

OUTPUT_FOLDER = str(config.path('vectorstore'))
FAISS_INDEX_PATH = str(config.path('vectorstore/faiss_index.index'))
METADATA_PATH = str(config.path('vectorstore/chunk_metadata.json'))

EMBEDDING_MODEL_NAME = config.EMBED_MODEL
BATCH_SIZE = 32

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================
# LOAD CHUNKS
# =========================
def load_chunks(jsonl_path):
    chunks = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded chunks: {len(chunks)}")
    return chunks


# =========================
# CREATE EMBEDDINGS
# =========================
def create_embeddings(chunks, model_name):
    model = SentenceTransformer(model_name)

    texts = [
    chunk["chunk_text"]
    for chunk in chunks
    if chunk.get("chunk_text", "").strip()
]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


# =========================
# CREATE FAISS INDEX
# =========================
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"FAISS index created with {index.ntotal} vectors.")
    return index


# =========================
# SAVE METADATA
# =========================
def save_metadata(chunks, metadata_path):
    metadata = []

    for idx, chunk in enumerate(chunks):
        metadata.append(
            {
                "faiss_id": idx,
                "chunk_id": chunk.get("chunk_id", ""),
                "pmcid": chunk.get("pmcid", ""),
                "title": chunk.get("title", ""),
                "journal": chunk.get("journal", ""),
                "year": chunk.get("year", ""),
                "authors": chunk.get("authors", []),
                "keywords": chunk.get("keywords", ""),
                "section": chunk.get("section", ""),
                "chunk_index": chunk.get("chunk_index", ""),
                "chunk_text": chunk.get("chunk_text", ""),
                "source_pdf": chunk.get("source_pdf", ""),
                "pdf_path": chunk.get("pdf_path", ""),
            }
        )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved to: {metadata_path}")


# =========================
# MAIN
# =========================
def main():
    chunks = load_chunks(CHUNKS_JSONL)

    chunks = [c for c in chunks if c.get("chunk_text", "").strip()]
    if not chunks:
        raise ValueError("No non-empty chunks to index.")

    embeddings = create_embeddings(
        chunks=chunks,
        model_name=EMBEDDING_MODEL_NAME
    )

    index = create_faiss_index(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")

    save_metadata(chunks, METADATA_PATH)

    print("\nVector store creation complete.")


if __name__ == "__main__":
    main()