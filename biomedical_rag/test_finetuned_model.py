from . import config
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


FAISS_INDEX_PATH = str(config.path('vectorstore/faiss_index.index'))

METADATA_PATH = str(config.path('vectorstore/chunk_metadata.json'))

EMBED_MODEL = config.EMBED_MODEL
BASE_MODEL = config.BASE_MODEL
ADAPTER_PATH = str(config.path('qlora_biomistral_adapter'))

TOP_K = 3


def load_retrieval():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")

    return index, metadata, embed_model


def retrieve_chunks(query, index, metadata, embed_model):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, TOP_K)

    chunks = []

    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        chunks.append(metadata[idx])

    return chunks


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
Answer the biomedical research question using only the provided evidence.
Provide a structured evidence-based response with Summary, Key Findings, exact source title, PMCID, section name, and Limitations.

### Input:
Question:
{query}

Evidence:
{evidence_text}

### Response:
"""

    return prompt


def load_finetuned_model():
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
        use_safetensors=False
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    model.eval()

    return tokenizer, model


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
            max_new_tokens=250,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    final_response = full_response.replace(prompt, "").strip()

    return final_response


def main():
    index, metadata, embed_model = load_retrieval()
    tokenizer, model = load_finetuned_model()

    query = input("Enter clinical query: ")

    chunks = retrieve_chunks(
        query=query,
        index=index,
        metadata=metadata,
        embed_model=embed_model
    )

    print("\nRetrieved Sources:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk['title']} | {chunk['pmcid']} | {chunk['section']}")

    prompt = build_prompt(query, chunks)

    answer = generate_answer(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model
    )

    print("\nFine-tuned BioMistral Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()