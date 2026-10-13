from . import config
import json
import csv
import re
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


FAISS_INDEX_PATH = str(config.path('vectorstore/faiss_index.index'))

METADATA_PATH = str(config.path('vectorstore/chunk_metadata.json'))

OUTPUT_CSV = str(config.path('results/evaluate_generation_auto_results.csv'))

ADAPTER_PATH = str(config.path('qlora_biomistral_adapter'))

EMBED_MODEL = config.EMBED_MODEL
BASE_MODEL = config.BASE_MODEL
ADAPTER_PATH = str(config.path('qlora_biomistral_adapter'))

TOP_K = 3


TEST_QUERIES = [
    "atrial fibrillation mobile health patient outcomes",
    "oncology trial enrollment barriers",
    "remote patient monitoring clinical benefits",
    "clinical trial patient retention strategies",
    "wearable devices in cardiovascular monitoring",
]


def clean_text(text):
    if not text:
        return ""

    text = str(text)
    text = text.replace("�", "")
    text = text.replace("\ufffd", "")
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_retrieval():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)

    return index, metadata, embed_model


def load_model():
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_safetensors=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        use_safetensors=False,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()

    return tokenizer, model


def retrieve_chunks(query, index, metadata, embed_model, top_k=3):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    chunks = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx].copy()
        chunk["score"] = float(score)
        chunks.append(chunk)

    return chunks


def truncate_to_sentence(text, max_chars=900):
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

    return truncated.rsplit(" ", 1)[0] + "..."


def build_prompt(query, chunks):
    evidence_text = ""

    for i, chunk in enumerate(chunks, start=1):
        evidence_text += f"""
Evidence {i}:
Title: {chunk.get("title", "")}
PMCID: {chunk.get("pmcid", "")}
Section: {chunk.get("section", "")}
Text: {truncate_to_sentence(chunk.get("chunk_text", ""), 900)}
"""

    prompt = f"""
You are an evidence-based clinical research assistant.

Using ONLY the provided research evidence, generate:
1. Summary
2. Key Findings
3. Limitations

Keep response concise and professional.
Do not repeat sentences.
Do not copy raw text.

Clinical Query:
{query}

Evidence:
{evidence_text}

Answer:
"""

    return prompt


def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_text(full_response.replace(prompt, "").strip())


def evaluate_answer(answer, chunks):
    lower_answer = answer.lower()

    has_summary = "summary" in lower_answer
    has_key_findings = "key findings" in lower_answer or "finding" in lower_answer
    has_limitations = "limitations" in lower_answer or "limitation" in lower_answer

    has_pmcid = bool(re.search(r"pmc\d+", lower_answer))

    answer_words = set(
        w for w in re.findall(r"[a-zA-Z]+", lower_answer)
        if len(w) > 4
    )

    evidence_text = " ".join(
        clean_text(chunk.get("chunk_text", ""))[:1000]
        for chunk in chunks
    ).lower()

    evidence_words = set(
        w for w in re.findall(r"[a-zA-Z]+", evidence_text)
        if len(w) > 4
    )

    if len(answer_words) == 0:
        grounding_overlap = 0
    else:
        grounding_overlap = len(answer_words.intersection(evidence_words)) / len(answer_words)

    structure_score = sum([
        has_summary,
        has_key_findings,
        has_limitations
    ]) / 3

    traceability_score = 1.0 if has_pmcid else 0.0

    length_score = 1.0 if 80 <= len(answer.split()) <= 350 else 0.5

    overall_score = (
        0.35 * structure_score +
        0.35 * grounding_overlap +
        0.20 * traceability_score +
        0.10 * length_score
    )

    return {
        "has_summary": has_summary,
        "has_key_findings": has_key_findings,
        "has_limitations": has_limitations,
        "has_pmcid": has_pmcid,
        "structure_score": round(structure_score, 3),
        "grounding_overlap": round(grounding_overlap, 3),
        "traceability_score": traceability_score,
        "length_score": length_score,
        "overall_generation_score": round(overall_score, 3),
        "answer_word_count": len(answer.split()),
    }


def main():
    index, metadata, embed_model = load_retrieval()
    tokenizer, model = load_model()

    rows = []

    for query in TEST_QUERIES:
        print("\n" + "=" * 100)
        print("Query:", query)

        chunks = retrieve_chunks(
            query=query,
            index=index,
            metadata=metadata,
            embed_model=embed_model,
            top_k=TOP_K
        )

        prompt = build_prompt(query, chunks)
        answer = generate_answer(prompt, tokenizer, model)

        eval_result = evaluate_answer(answer, chunks)

        print("Generated Answer:")
        print(answer[:800])
        print("\nEvaluation:")
        print(eval_result)

        rows.append({
            "query": query,
            "answer": answer,
            "retrieved_pmcids": "; ".join([c.get("pmcid", "") for c in chunks]),
            "retrieved_titles": " || ".join([c.get("title", "") for c in chunks]),
            **eval_result
        })

    config.path("results").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "query",
            "answer",
            "retrieved_pmcids",
            "retrieved_titles",
            "has_summary",
            "has_key_findings",
            "has_limitations",
            "has_pmcid",
            "structure_score",
            "grounding_overlap",
            "traceability_score",
            "length_score",
            "overall_generation_score",
            "answer_word_count",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    avg_score = sum(row["overall_generation_score"] for row in rows) / len(rows)

    print("\n" + "=" * 100)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 100)
    print(f"Average Generation Score: {avg_score:.3f}")
    print(f"Detailed results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()