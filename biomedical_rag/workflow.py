"""Workflow helpers extracted from the academic application; heuristics preserved."""

import re
import math
import unicodedata
import pandas as pd
import fitz

TEST_QUERIES = [
    {
        "query": "atrial fibrillation mobile health patient outcomes",
        "expected_keywords": [
            "atrial fibrillation", "mobile health", "mhealth",
            "patient-reported", "outcomes", "quality of life", "telewear"
        ],
    },
    {
        "query": "oncology trial enrollment barriers",
        "expected_keywords": [
            "oncology", "cancer", "clinical trial", "enrollment", "barriers", "patients"
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
            "clinical trial", "patient retention", "recruitment", "enrollment", "participation"
        ],
    },
    {
        "query": "wearable devices in cardiovascular monitoring",
        "expected_keywords": [
            "wearable", "cardiovascular", "ecg", "monitoring", "mobile health", "heart"
        ],
    },
]

def clean_display_text(text):
    if not text:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", text)
    text = text.replace("�", "").replace("\ufffd", "").replace("\x00", "")
    text = text.replace("­", "")
    text = re.sub(r"[□■]", "", text)

    noisy_patterns = [
        r"XSL\s*•?\s*FO\s*RenderX",
        r"page number not for citation purposes",
        r"DOI:\s*\S+",
        r"https?://\S+",
        r"\bRenderX\b",
    ]

    for pattern in noisy_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)

    return text.strip()

def truncate_to_sentence(text, max_chars=1000):
    text = clean_display_text(text)

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

def generate_dynamic_summary(chunks):
    if not chunks:
        return "No relevant biomedical evidence was retrieved."

    summary_parts = []

    for chunk in chunks[:2]:
        text = clean_display_text(chunk.get("chunk_text", ""))

        text = truncate_to_sentence(text, 350)

        sentences = re.split(r'(?<=[.!?])\s+', text)

        if sentences:
            first_sentence = sentences[0].strip()

            if len(first_sentence.split()) > 6:
                summary_parts.append(first_sentence)

    if not summary_parts:
        return "Relevant biomedical evidence was retrieved and synthesized from the literature."

    summary = " ".join(summary_parts)

    summary = re.sub(r"\s+", " ", summary).strip()

    return summary

def get_terms(text):
    return set(
        w.lower()
        for w in re.findall(r"[a-zA-Z]+", str(text))
        if len(w) > 3
    )

def extract_biomedical_entities(query):
    biomedical_keywords = [
        "atrial fibrillation", "diabetes", "oncology", "cancer",
        "cardiovascular", "wearable", "remote monitoring", "mobile health",
        "mhealth", "clinical trial", "patient outcomes", "quality of life",
        "patient-reported outcomes", "telemedicine", "digital health",
        "enrollment", "retention", "adverse events", "treatment",
        "monitoring", "ecg", "heart failure"
    ]

    found = []
    q = query.lower()

    for term in biomedical_keywords:
        if term in q:
            found.append(term)

    return found

def extract_pico(query):
    q = query.lower()

    pico = {
        "Population": "Not explicitly specified",
        "Intervention": "Not explicitly specified",
        "Comparison": "Not explicitly specified",
        "Outcome": "Not explicitly specified",
    }

    if any(x in q for x in ["patient", "patients", "adult", "clinical trial"]):
        pico["Population"] = "Patients / clinical study participants"

    if any(x in q for x in ["mobile health", "mhealth", "wearable", "remote monitoring", "telemedicine", "digital health"]):
        pico["Intervention"] = "Digital health / remote monitoring intervention"

    if any(x in q for x in ["usual care", "control", "compared", "comparison"]):
        pico["Comparison"] = "Usual care / comparator group"

    if any(x in q for x in ["outcome", "quality of life", "effectiveness", "retention", "enrollment", "barriers"]):
        pico["Outcome"] = "Patient outcomes / clinical effectiveness / study participation outcomes"

    return pico

def query_understanding_agent(query):
    entities = extract_biomedical_entities(query)
    pico = extract_pico(query)

    return {
        "original_query": query,
        "entities": entities,
        "pico": pico,
    }

def query_reformulation_agent(query_analysis):
    query = query_analysis["original_query"]
    entities = query_analysis["entities"]

    combined_text = query + " " + " ".join(entities)

    words = combined_text.split()
    unique_words = list(dict.fromkeys(words))

    reformulated = " ".join(unique_words)
    reformulated = re.sub(r"\s+", " ", reformulated).strip()

    return reformulated

def retrieval_planning_agent(query_analysis, user_top_k):
    return {
        "top_k": user_top_k,
        "strategy": "FAISS semantic retrieval with metadata-aware filtering",
    }

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

def retrieve_chunks(query, index, metadata, embed_model, top_k=3):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    search_k = min(max(top_k * 12, 30), index.ntotal)
    scores, indices = index.search(query_embedding, search_k)

    selected = []
    seen_chunk_ids = set()
    seen_pmcids = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        chunk = metadata[idx].copy()
        chunk["score"] = float(score)

        section = str(chunk.get("section", "")).lower()
        text = clean_display_text(chunk.get("chunk_text", ""))

        if section in {"references", "supplementary", "acknowledgments", "author_contributions", "title"}:
            continue

        if len(text.split()) < 60:
            continue

        score = relevance_score(chunk, query)

        query_words = set(get_terms(query))
        chunk_words = set(get_terms(chunk["chunk_text"]))

        overlap = len(query_words.intersection(chunk_words))

        important_terms = [
            word for word in query_words
            if len(word) > 4
        ]

        important_overlap = len(
            set(important_terms).intersection(chunk_words)
        )

        if score < 2:
            continue

        if overlap < 2:
            continue

        if important_overlap < 1:
            continue

        chunk_id = chunk.get("chunk_id", "")
        pmcid = chunk.get("pmcid", "")

        if chunk_id in seen_chunk_ids:
            continue

        if pmcid not in seen_pmcids:
            selected.append(chunk)
            seen_chunk_ids.add(chunk_id)
            seen_pmcids.add(pmcid)

        if len(selected) >= top_k:
            break

        # If filtering becomes too strict,
        # allow slightly relaxed retrieval
        # but still preserve query overlap

        if len(selected) < top_k:

            for score, idx in zip(scores[0], indices[0]):

                if idx < 0 or idx >= len(metadata):
                    continue

                chunk = metadata[idx].copy()
                chunk["score"] = float(score)

                chunk_id = chunk.get("chunk_id", "")
                pmcid = chunk.get("pmcid", "")

                if chunk_id in seen_chunk_ids:
                    continue

                if pmcid in seen_pmcids:
                    continue

                text = clean_display_text(
                    chunk.get("chunk_text", "")
                )

                if len(text.split()) < 60:
                    continue

                query_words = set(get_terms(query))
                chunk_words = set(get_terms(text))

                overlap = len(
                    query_words.intersection(chunk_words)
                )

                # relaxed fallback, but still requires query overlap
                if overlap < 1:
                    continue

                selected.append(chunk)
                seen_chunk_ids.add(chunk_id)
                seen_pmcids.add(pmcid)

                if len(selected) >= top_k:
                    break

    return selected[:top_k]

def build_prompt(query, chunks, query_analysis):
    evidence_text = ""

    for i, chunk in enumerate(chunks, start=1):
        evidence_text += f"""
Evidence {i}:
Title: {chunk.get('title', '')}
PMCID: {chunk.get('pmcid', '')}
Section: {chunk.get('section', '')}
Text: {truncate_to_sentence(chunk.get('chunk_text', ''), 900)}
"""

    prompt = f"""
### Instruction:
You are an evidence-based biomedical research assistant.

Answer ONLY using the provided evidence.
Do not use external knowledge.
Do not hallucinate.
Generate the answer with exactly these sections:

Summary:
Key Findings:
Limitations:

For every key finding, include:
Source title | PMCID | Section

### Query Understanding:
Biomedical Entities: {query_analysis.get("entities", [])}
PICO: {query_analysis.get("pico", {})}

### Clinical Query:
{query}

### Retrieved Evidence:
{evidence_text}

### Response:
"""

    return prompt

def generate_answer(prompt, tokenizer, model, max_new_tokens=350):
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.25,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_response.replace(prompt, "").strip()

def validate_answer(answer):
    lower = answer.lower()

    has_summary = "summary" in lower
    has_key_findings = "key findings" in lower or "finding" in lower
    has_limitations = "limitations" in lower or "limitation" in lower
    has_pmcid = bool(re.search(r"pmc\d+", lower))

    bad_patterns = [
        "fetched:",
        "json document",
        "full response:",
        "clinicaldecision",
        "reportedoutcome",
        "thepatient",
        "```json",
    ]

    has_bad_pattern = any(p in lower for p in bad_patterns)

    score = 0
    score += 0.25 if has_summary else 0
    score += 0.25 if has_key_findings else 0
    score += 0.25 if has_limitations else 0
    score += 0.25 if has_pmcid else 0

    is_valid = score >= 0.75 and not has_bad_pattern

    return {
        "is_valid": is_valid,
        "structure_score": score,
        "has_summary": has_summary,
        "has_key_findings": has_key_findings,
        "has_limitations": has_limitations,
        "has_pmcid": has_pmcid,
        "has_bad_pattern": has_bad_pattern,
    }

def build_safe_evidence_answer(query, chunks):

    dynamic_summary = generate_dynamic_summary(chunks)

    answer = f"""Summary:
{dynamic_summary}

Key Findings:
"""

    for i, chunk in enumerate(chunks, start=1):

        evidence_preview = truncate_to_sentence(
            chunk.get("chunk_text", ""),
            320
        )

        answer += f"""
{i}. {evidence_preview}

   Source: {chunk.get('title', '')}
   PMCID: {chunk.get('pmcid', '')}
   Section: {chunk.get('section', '')}
"""

    answer += """

Limitations:
The answer is synthesized only from the retrieved biomedical evidence chunks. The retrieved excerpts may not represent the complete article or the entire biomedical literature. Clinical interpretation should be validated through full-paper review and expert assessment.
"""

    return answer

def calculate_confidence(chunks, validation_result):
    if not chunks:
        return 0.0

    avg_similarity = sum(chunk.get("score", 0) for chunk in chunks) / len(chunks)
    structure_score = validation_result.get("structure_score", 0)

    confidence = (0.7 * avg_similarity) + (0.3 * structure_score)
    return round(min(confidence, 1.0), 3)

def build_study_comparison_table(chunks):
    rows = []

    for i, chunk in enumerate(chunks, start=1):
        rows.append(
            {
                "Rank": i,
                "PMCID": chunk.get("pmcid", ""),
                "Title": chunk.get("title", ""),
                "Journal": chunk.get("journal", ""),
                "Year": chunk.get("year", ""),
                "Section": chunk.get("section", ""),
                "Similarity Score": round(chunk.get("score", 0), 4),
            }
        )

    return pd.DataFrame(rows)

def is_relevant(chunk, expected_keywords):
    combined_text = (
        str(chunk.get("title", "")) + " " +
        str(chunk.get("section", "")) + " " +
        str(chunk.get("keywords", "")) + " " +
        str(chunk.get("chunk_text", ""))
    ).lower()

    matched = [
        kw for kw in expected_keywords
        if kw.lower() in combined_text
    ]

    return len(matched) >= 1, matched

def dcg(relevances):
    score = 0.0

    for i, rel in enumerate(relevances, start=1):
        score += rel / math.log2(i + 1)

    return score

def ndcg_at_k(relevances):
    actual_dcg = dcg(relevances)
    ideal_dcg = dcg(sorted(relevances, reverse=True))

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

def run_retrieval_evaluation(index, metadata, embed_model, k):
    rows = []

    for test in TEST_QUERIES:
        query = test["query"]
        expected_keywords = test["expected_keywords"]

        chunks = retrieve_chunks(
            query=query,
            index=index,
            metadata=metadata,
            embed_model=embed_model,
            top_k=k,
        )

        relevances = []
        relevant_count = 0

        for rank, chunk in enumerate(chunks, start=1):
            relevant, matched = is_relevant(chunk, expected_keywords)
            rel_value = 1 if relevant else 0
            relevances.append(rel_value)

            if relevant:
                relevant_count += 1

            rows.append(
                {
                    "Query": query,
                    "Rank": rank,
                    "PMCID": chunk.get("pmcid", ""),
                    "Title": chunk.get("title", ""),
                    "Section": chunk.get("section", ""),
                    "Score": round(chunk.get("score", 0), 4),
                    "Relevant": relevant,
                    "Matched Keywords": ", ".join(matched),
                }
            )

        precision = relevant_count / max(k, 1)
        ndcg = ndcg_at_k(relevances)

        rows.append(
            {
                "Query": query,
                "Rank": "SUMMARY",
                "PMCID": "",
                "Title": "",
                "Section": "",
                "Score": "",
                "Relevant": f"Precision@{k}: {precision:.2f}",
                "Matched Keywords": f"NDCG@{k}: {ndcg:.2f}",
            }
        )

    return pd.DataFrame(rows)

def extract_text_from_uploaded_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text_parts = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if page_text:
            text_parts.append(f"\n[Page {page_num}]\n{page_text}")

    text = "\n".join(text_parts)

    # remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return clean_display_text(text)

def summarize_uploaded_pdf(pdf_text, tokenizer=None, model=None, max_new_tokens=250):
    if not pdf_text.strip():
        return "No readable text could be extracted from the uploaded PDF."

    text = clean_display_text(pdf_text)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    useful_sentences = []
    keywords = [
        "objective", "background", "method", "methods", "result", "results",
        "conclusion", "case", "patient", "study", "finding", "treatment",
        "clinical", "outcome", "limitation"
    ]

    skip_patterns = [
    "received:",
    "accepted:",
    "revised:",
    "author(s)",
    "copyright",
]

    for sentence in sentences:
        sentence_clean = sentence.strip()

        if len(sentence_clean.split()) < 8:
            continue

        if len(sentence_clean) > 350:
            continue

        # Skip publication metadata
        if any(p in sentence_clean.lower() for p in skip_patterns):
            continue

        if any(k in sentence_clean.lower() for k in keywords):
            useful_sentences.append(sentence_clean)

    if len(useful_sentences) < 5:
        useful_sentences = [
            s.strip() for s in sentences
            if 8 <= len(s.strip().split()) <= 45
        ]

    selected = useful_sentences[1:7]

    summary = selected[0] if len(selected) > 0 else "The uploaded PDF text was extracted successfully, but a meaningful summary could not be generated from the available text."

    summary = re.sub(r"\s+", " ", summary).strip()
    summary = summary.replace("[Page 1]", "").strip()

    key_points = []

    for sentence in selected[2:6]:
        sub_sentences = re.split(r'(?<=[.!?])\s+', sentence)

        for sub in sub_sentences:
            sub = sub.strip()

            if len(sub.split()) >= 8:
                key_points.append(sub)

    key_points = key_points[:3]

    limitations = (
        "This summary is generated only from the extracted PDF text preview and may not cover the complete document. "
        "Full-paper review is recommended for clinical or research interpretation."
    )

    summary = re.sub(r"CASE BASED REVIEW", "", summary, flags=re.IGNORECASE)
    summary = re.sub(r"Received:.*?2020", "", summary)
    summary = re.sub(r"Abstract", "", summary, flags=re.IGNORECASE)
    summary = re.sub(r"\s+", " ", summary).strip()

    output = "Summary:\n"
    output += summary + "\n\n"

    output += "Key Points:\n"
    for point in key_points:
        output += f"\n• {point.strip()}\n"

    output += "\nLimitations:\n"
    output += f"- {limitations}"

    return output
