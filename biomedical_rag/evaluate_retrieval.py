from . import config
import json
import csv
import faiss
from sentence_transformers import SentenceTransformer


FAISS_INDEX_PATH = str(config.path('vectorstore/faiss_index.index'))

METADATA_PATH = str(config.path('vectorstore/chunk_metadata.json'))

OUTPUT_CSV = str(config.path('results/evaluate_retrieval_results.csv'))
EMBED_MODEL = config.EMBED_MODEL
TOP_K = 3


TEST_QUERIES = [
    {
        "query": "atrial fibrillation mobile health patient outcomes",
        "expected_keywords": [
            "atrial fibrillation", "mobile health", "mhealth",
            "patient-reported", "outcomes", "quality of life",
            "telewear", "impact-af"
        ],
    },
    {
        "query": "oncology trial enrollment barriers",
        "expected_keywords": [
            "oncology", "cancer", "clinical trial",
            "enrollment", "barriers", "patients"
        ],
    },
    {
        "query": "remote patient monitoring clinical benefits",
        "expected_keywords": [
            "remote monitoring",
            "patient monitoring",
            "clinical benefits",
            "telemedicine",
            "digital health",
            "patient outcomes"
        ],
    },
    {
        "query": "clinical trial patient retention strategies",
        "expected_keywords": [
            "clinical trial", "patient retention", "recruitment",
            "enrollment", "participation", "withdrawal"
        ],
    },
    {
        "query": "wearable devices in cardiovascular monitoring",
        "expected_keywords": [
            "wearable", "cardiovascular", "ecg",
            "monitoring", "mobile health", "heart"
        ],
    },
]


def load_components():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SentenceTransformer(EMBED_MODEL)

    return index, metadata, model


def retrieve(query, index, metadata, model, top_k=3):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def is_relevant(chunk, expected_keywords):
    combined_text = (
        str(chunk.get("title", "")) + " " +
        str(chunk.get("section", "")) + " " +
        str(chunk.get("keywords", "")) + " " +
        str(chunk.get("chunk_text", ""))
    ).lower()

    matched_keywords = []

    for keyword in expected_keywords:
        if keyword.lower() in combined_text:
            matched_keywords.append(keyword)

    relevant = len(matched_keywords) >= 1

    return relevant, matched_keywords


def evaluate():
    index, metadata, model = load_components()

    rows = []
    summary_rows = []

    for test_item in TEST_QUERIES:
        query = test_item["query"]
        expected_keywords = test_item["expected_keywords"]

        retrieved_chunks = retrieve(
            query=query,
            index=index,
            metadata=metadata,
            model=model,
            top_k=TOP_K
        )

        relevant_count = 0

        print("\n" + "=" * 100)
        print(f"Query: {query}")
        print("=" * 100)

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            relevant, matched_keywords = is_relevant(chunk, expected_keywords)

            if relevant:
                relevant_count += 1

            print(f"\nRank: {rank}")
            print(f"Relevant: {relevant}")
            print(f"Score: {chunk.get('score', 0):.4f}")
            print(f"PMCID: {chunk.get('pmcid', '')}")
            print(f"Title: {chunk.get('title', '')}")
            print(f"Section: {chunk.get('section', '')}")
            print(f"Matched Keywords: {matched_keywords}")

            rows.append({
                "query": query,
                "rank": rank,
                "similarity_score": chunk.get("score", ""),
                "pmcid": chunk.get("pmcid", ""),
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "matched_keywords": ", ".join(matched_keywords),
                "is_relevant": relevant,
            })

        precision_at_k = relevant_count / TOP_K

        print(f"\nPrecision@{TOP_K}: {precision_at_k:.2f}")

        summary_rows.append({
            "query": query,
            f"relevant_chunks_in_top_{TOP_K}": relevant_count,
            f"precision@{TOP_K}": precision_at_k,
        })

    config.path("results").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "query",
            "rank",
            "similarity_score",
            "pmcid",
            "title",
            "section",
            "matched_keywords",
            "is_relevant",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total_precision = 0

    for row in summary_rows:
        print(row)
        total_precision += row[f"precision@{TOP_K}"]

    avg_precision = total_precision / len(summary_rows)

    print(f"\nAverage Precision@{TOP_K}: {avg_precision:.2f}")
    print(f"Detailed results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    evaluate()