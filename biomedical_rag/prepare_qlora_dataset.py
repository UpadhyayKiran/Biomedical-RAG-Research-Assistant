from . import config
import os
import json
import faiss
import random
import re
import torch
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================
FAISS_INDEX_PATH = str(config.path('vectorstore/faiss_index.index'))
METADATA_PATH = str(config.path('vectorstore/chunk_metadata.json'))

OUTPUT_FOLDER = str(config.path('qlora_dataset'))
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "biomistral_qlora_train.jsonl")

EMBED_MODEL = config.EMBED_MODEL

NUM_EXAMPLES = 175
TOP_K = 3
MAX_CHUNK_CHARS = 700
MAX_ATTEMPTS = 2000

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================
# QUESTIONS
# =========================
BASE_QUESTIONS = [
    "What does the evidence say about atrial fibrillation mobile health patient outcomes?",
    "How do mobile health tools support atrial fibrillation management?",
    "What are the patient-reported outcomes in atrial fibrillation clinical studies?",
    "What evidence is available about wearable ECG monitoring in cardiovascular care?",
    "What does the evidence say about patient-reported outcome measures in clinical trials?",
    "How are patient-reported outcomes used in cardiovascular clinical research?",
    "What are the benefits and limitations of digital health tools in patient care?",
    "What does the evidence say about clinical trial outcomes in oncology?",
    "What are the reported costs and resource needs in cancer clinical trials?",
    "What evidence is available about treatment outcomes in cancer clinical studies?",
    "What are the key findings about prognosis and recovery in patient studies?",
    "What risk factors are associated with functional recovery in clinical research?",
    "How are adverse events and safety outcomes reported in clinical studies?",
    "What does the evidence say about quality of life measurement in patients?",
    "How do clinical studies evaluate treatment effectiveness using patient outcomes?",
    "What evidence exists regarding telemedicine effectiveness in chronic disease management?",
    "What are the reported benefits of wearable health monitoring systems?",
    "What barriers affect patient participation in biomedical clinical trials?",
    "How do remote monitoring systems improve patient outcomes?",
    "What evidence exists regarding cardiovascular digital health interventions?",
    "What are the limitations of mobile health systems in clinical care?",
    "What patient-reported outcome measures are commonly used in clinical research?",
    "What are the major findings regarding decentralized clinical trials?",
    "What evidence supports remote patient monitoring technologies?",
    "What clinical evidence exists for digital healthcare interventions?",
]

QUESTION_TEMPLATES = [
    "{}",
    "Summarize the evidence for: {}",
    "Provide an evidence-based answer about: {}",
    "What are the key findings and limitations regarding: {}",
    "Using retrieved evidence, answer: {}",
    "Based only on the provided biomedical evidence, explain: {}",
    "Give a concise clinical research summary for: {}",
]


# =========================
# LOAD
# =========================
def load_resources():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embed_model = SentenceTransformer(
        EMBED_MODEL,
        device=device
    )

    print(f"Loaded FAISS vectors: {index.ntotal}")
    print(f"Loaded metadata records: {len(metadata)}")
    print(f"Embedding device: {device}")

    return index, metadata, embed_model


# =========================
# TEXT HELPERS
# =========================
def clean_text(text):
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text):
    text = clean_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 40]


def get_terms(text):
    return set(
        w.lower()
        for w in re.findall(r"[a-zA-Z]+", str(text))
        if len(w) > 3
    )


def relevance_score(chunk, query):
    query_terms = get_terms(query)

    chunk_text = (
        str(chunk.get("title", "")) + " " +
        str(chunk.get("keywords", "")) + " " +
        str(chunk.get("section", "")) + " " +
        str(chunk.get("chunk_text", ""))
    )

    chunk_terms = get_terms(chunk_text)

    return len(query_terms.intersection(chunk_terms))


def best_sentence_for_query(text, query):
    sentences = split_sentences(text)

    if not sentences:
        return clean_text(text)[:250]

    query_terms = get_terms(query)

    best = sentences[0]
    best_score = -1

    for sentence in sentences:
        sentence_terms = get_terms(sentence)
        score = len(query_terms.intersection(sentence_terms))

        if score > best_score:
            best_score = score
            best = sentence

    return clean_text(best)


def truncate_text(text, max_chars):
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_end = max(
        truncated.rfind("."),
        truncated.rfind("?"),
        truncated.rfind("!")
    )

    if last_end > 250:
        return truncated[:last_end + 1].strip()

    return truncated.rsplit(" ", 1)[0].strip() + "..."


# =========================
# RETRIEVE
# =========================
def retrieve_chunks(query, index, metadata, embed_model, top_k=3):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    search_k = min(max(top_k * 15, 30), index.ntotal)
    scores, indices = index.search(query_embedding, search_k)

    selected = []
    seen_pmcids = set()
    seen_chunk_ids = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx].copy()
        chunk["score"] = float(score)

        section = str(chunk.get("section", "")).lower()
        text = clean_text(chunk.get("chunk_text", ""))

        if section in {
            "references",
            "supplementary",
            "acknowledgments",
            "author_contributions",
            "title",
        }:
            continue

        if len(text.split()) < 60:
            continue

        rel_score = relevance_score(chunk, query)

        if rel_score < 2:
            continue

        chunk_id = chunk.get("chunk_id", "")

        if chunk_id in seen_chunk_ids:
            continue

        pmcid = chunk.get("pmcid", "")

        if pmcid not in seen_pmcids:
            selected.append(chunk)
            seen_pmcids.add(pmcid)
            seen_chunk_ids.add(chunk_id)

        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue

            chunk = metadata[idx].copy()
            chunk["score"] = float(score)

            text = clean_text(chunk.get("chunk_text", ""))
            chunk_id = chunk.get("chunk_id", "")

            if len(text.split()) < 60:
                continue

            if chunk_id in seen_chunk_ids:
                continue

            if relevance_score(chunk, query) < 2:
                continue

            selected.append(chunk)
            seen_chunk_ids.add(chunk_id)

            if len(selected) >= top_k:
                break

    return selected[:top_k]


# =========================
# BUILD EXAMPLE
# =========================
def build_input(question, chunks):
    evidence_text = ""

    for i, chunk in enumerate(chunks, start=1):
        evidence_text += f"""
Evidence {i}:
Title: {chunk.get("title", "")}
PMCID: {chunk.get("pmcid", "")}
Section: {chunk.get("section", "")}
Text: {truncate_text(chunk.get("chunk_text", ""), MAX_CHUNK_CHARS)}
"""

    return f"""Question:
{question}

Evidence:
{evidence_text}
"""


def build_output(question, chunks):
    summary_sentences = []

    for chunk in chunks[:2]:
        sentence = best_sentence_for_query(chunk.get("chunk_text", ""), question)
        summary_sentences.append(sentence)

    summary = " ".join(summary_sentences)

    key_findings = []

    for i, chunk in enumerate(chunks, start=1):
        finding = best_sentence_for_query(chunk.get("chunk_text", ""), question)

        key_findings.append(
            f"{i}. {finding}\n"
            f"   Source: {chunk.get('title', '')} | PMCID: {chunk.get('pmcid', '')} | Section: {chunk.get('section', '')}"
        )

    return f"""Summary:
{summary}

Key Findings:
{chr(10).join(key_findings)}

Limitations:
The answer is based only on the retrieved evidence chunks. The retrieved excerpts may not represent the full article or the complete biomedical literature. Clinical interpretation should be validated using the full papers and expert review.
"""


def build_training_example(question, chunks):
    return {
        "instruction": (
            "You are an evidence-based biomedical research assistant. "
            "Answer ONLY using the provided evidence. "
            "Do not hallucinate or use external knowledge. "
            "Generate a professional response with these exact sections:\n"
            "Summary:\n"
            "Key Findings:\n"
            "Limitations:\n\n"
            "For every finding include:\n"
            "- source title\n"
            "- PMCID\n"
            "- section name\n\n"
            "If evidence is insufficient, explicitly state that."
        ),
        "input": build_input(question, chunks),
        "output": build_output(question, chunks),
    }


# =========================
# MAIN
# =========================
def main():
    random.seed(42)

    index, metadata, embed_model = load_resources()

    examples = []
    used_questions = set()
    skipped_insufficient_chunks = 0

    attempts = 0

    while len(examples) < NUM_EXAMPLES and attempts < MAX_ATTEMPTS:
        attempts += 1

        base_question = random.choice(BASE_QUESTIONS)
        template = random.choice(QUESTION_TEMPLATES)
        question = template.format(base_question)

        if question in used_questions:
            continue

        chunks = retrieve_chunks(
            query=question,
            index=index,
            metadata=metadata,
            embed_model=embed_model,
            top_k=TOP_K,
        )

        if len(chunks) < 2:
            skipped_insufficient_chunks += 1
            continue

        example = build_training_example(question, chunks)

        examples.append(example)
        used_questions.add(question)

        if len(examples) % 25 == 0:
            print(f"Created {len(examples)} examples...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("\nRetrieval-based QLoRA dataset created successfully.")
    print(f"Requested examples: {NUM_EXAMPLES}")
    print(f"Examples created: {len(examples)}")
    print(f"Total attempts: {attempts}")
    print(f"Skipped due to insufficient chunks: {skipped_insufficient_chunks}")
    print(f"Saved to: {OUTPUT_FILE}")

    if len(examples) < NUM_EXAMPLES:
        print(
            f"Warning: Requested {NUM_EXAMPLES} examples, "
            f"but only created {len(examples)} within {MAX_ATTEMPTS} attempts."
        )


if __name__ == "__main__":
    main()