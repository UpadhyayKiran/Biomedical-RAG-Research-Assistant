from . import config
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# =========================
# CONFIG
# =========================
FAISS_INDEX_PATH = str(config.path('vectorstore/faiss_index.index'))
METADATA_PATH = str(config.path('vectorstore/chunk_metadata.json'))

EMBED_MODEL = config.EMBED_MODEL
LLM_MODEL = config.BASE_MODEL

TOP_K = 3


# =========================
# LOAD RETRIEVAL COMPONENTS
# =========================
def load_retrieval():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embed_model = SentenceTransformer(EMBED_MODEL)

    return index, metadata, embed_model


# =========================
# LOAD BIOMISTRAL
# =========================
def load_llm():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: No GPU detected")

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        quantization_config=quant_config,
        device_map="auto"
    )

    return tokenizer, model


# =========================
# RETRIEVE CHUNKS
# =========================
def retrieve_chunks(query, index, metadata, embed_model):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, TOP_K)

    retrieved_chunks = []

    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = metadata[idx]
        retrieved_chunks.append(chunk)

    return retrieved_chunks


# =========================
# BUILD PROMPT
# =========================
def build_prompt(query, chunks):
    evidence_text = ""

    for i, chunk in enumerate(chunks, start=1):
        evidence_text += f"""
Evidence {i}:
Title: {chunk['title']}
PMCID: {chunk['pmcid']}
Section: {chunk['section']}
Text: {chunk['chunk_text'][:1000]}

"""

    prompt = f"""
### Instruction:
You are an evidence-based biomedical research assistant.

Answer ONLY using the provided evidence.
Do NOT use external knowledge.
Do NOT hallucinate information.

For every important finding:
- mention the exact source title
- mention PMCID
- mention section name

If evidence is insufficient, clearly say so.

### Question:
{query}

### Retrieved Evidence:
{evidence_text}

### Required Response Format:

Summary:
Provide a concise evidence-based summary.

Key Findings:
1. Finding:
   Source: Title | PMCID | Section

2. Finding:
   Source: Title | PMCID | Section

3. Finding:
   Source: Title | PMCID | Section

Limitations:
Mention limitations of the available evidence.

### Response:
"""

    return prompt


# =========================
# GENERATE ANSWER
# =========================
def generate_answer(prompt, tokenizer, model):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1
    )

    full_response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove prompt echo
    final_response = full_response.replace(prompt, "").strip()

    return final_response


# =========================
# MAIN
# =========================
def main():
    index, metadata, embed_model = load_retrieval()
    tokenizer, model = load_llm()

    query = input("Enter clinical query: ")

    chunks = retrieve_chunks(
        query,
        index,
        metadata,
        embed_model
    )

    prompt = build_prompt(query, chunks)

    answer = generate_answer(
        prompt,
        tokenizer,
        model
    )

    print("\nGenerated Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()