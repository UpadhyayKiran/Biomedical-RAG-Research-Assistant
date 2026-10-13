from pathlib import Path
from biomedical_rag import config
from biomedical_rag.workflow import (clean_display_text, truncate_to_sentence, generate_dynamic_summary, get_terms, extract_biomedical_entities, extract_pico, query_understanding_agent, query_reformulation_agent, retrieval_planning_agent, relevance_score, retrieve_chunks, build_prompt, generate_answer, validate_answer, build_safe_evidence_answer, calculate_confidence, build_study_comparison_table, is_relevant, dcg, ndcg_at_k, run_retrieval_evaluation, extract_text_from_uploaded_pdf, summarize_uploaded_pdf)
import json
import faiss
import fitz
import torch
import re
import math
import unicodedata
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from biomedical_rag.logger_utils import log_query_result


# =========================
# PATHS
# =========================
FAISS_INDEX_PATH = str(config.path("vectorstore/faiss_index.index"))
METADATA_PATH = str(config.path("vectorstore/chunk_metadata.json"))

EMBED_MODEL = config.EMBED_MODEL
BASE_MODEL = config.BASE_MODEL
ADAPTER_PATH = str(config.path("qlora_biomistral_adapter"))

DEFAULT_TOP_K = 3


st.set_page_config(
    page_title="Agentic Evidence-Based Clinical Research Assistant",
    page_icon="🧬",
    layout="wide",
)


# =========================
# LOADERS
# =========================
@st.cache_resource
def load_retrieval_components():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)

    return index, metadata, embed_model


@st.cache_resource
def load_finetuned_model():
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

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    return tokenizer, model

# =========================
# TEXT CLEANING
# =========================






# =========================
# AGENT STEP 1: QUERY UNDERSTANDING
# =========================






# =========================
# AGENT STEP 2: QUERY REFORMULATION
# =========================


# =========================
# AGENT STEP 3: RETRIEVAL PLANNING
# =========================


# =========================
# AGENT STEP 4: EVIDENCE RETRIEVAL
# =========================




# =========================
# AGENT STEP 5: EVIDENCE SYNTHESIS
# =========================




# =========================
# AGENT STEP 6: VALIDATION
# =========================






# =========================
# STUDY COMPARISON TABLE
# =========================


# =========================
# RETRIEVAL EVALUATION
# =========================












# =========================
# STREAMLIT APP
# =========================
def main():
    st.title("🧬 Agentic Evidence-Based Clinical Research Assistant")
    st.caption(
        "Agentic RAG pipeline using PubMedBERT embeddings, FAISS retrieval, "
        "QLoRA-adapted BioMistral, response validation, and citation-grounded evidence display."
    )

    with st.sidebar:
        st.header("Settings")

        top_k = st.slider(
            "Number of retrieved evidence chunks",
            1,
            5,
            DEFAULT_TOP_K
        )

        max_new_tokens = st.slider(
            "Max answer tokens",
            100,
            600,
            350,
            step=50
        )

        st.markdown("---")

        st.markdown("""
        ### Agentic Workflow
        1. Query Understanding  
        2. PICO + Entity Extraction  
        3. Query Reformulation  
        4. Retrieval Planning  
        5. Evidence Retrieval  
        6. Evidence Synthesis  
        7. Response Validation  
        8. Final Evidence Output  
        """)

        st.markdown("---")
        st.write("### System Status")
        st.write(f"CUDA available: `{torch.cuda.is_available()}`")

        if torch.cuda.is_available():
            st.write(f"GPU: `{torch.cuda.get_device_name(0)}`")

    tab1, tab2, tab3 = st.tabs([
    "🧬 Research Agent",
    "📊 Retrieval Evaluation",
    "📄 PDF Summary"
    ])

    st.warning("Research prototype only. Not for medical diagnosis or treatment. Do not enter patient-identifiable information.")

    with tab1:
        query = st.text_area(
            "Enter your clinical research question:",
            value="atrial fibrillation mobile health patient outcomes",
            height=100,
        )

        st.markdown("""
        ### Example Queries
        - atrial fibrillation mobile health patient outcomes  
        - oncology trial enrollment barriers  
        - remote patient monitoring clinical benefits  
        - clinical trial patient retention strategies  
        - wearable devices in cardiovascular monitoring  
        """)

        run_button = st.button("Run Clinical Research Agent", type="primary")

        if run_button:
            if not query.strip():
                st.warning("Please enter a query.")
                return

            missing = [p for p in [FAISS_INDEX_PATH, METADATA_PATH, ADAPTER_PATH] if not Path(p).exists()]
            if missing:
                st.error("Local artifacts are missing. Follow docs/DATA.md to rebuild the index and adapter.")
                return

            with st.spinner("Loading retrieval components and fine-tuned model..."):
                index, metadata, embed_model = load_retrieval_components()
                tokenizer, model = load_finetuned_model()

            with st.spinner("Agent Step 1: Understanding query..."):
                query_analysis = query_understanding_agent(query)

            with st.spinner("Agent Step 2: Reformulating query..."):
                reformulated_query = query_reformulation_agent(query_analysis)

            with st.spinner("Agent Step 3: Planning retrieval..."):
                retrieval_plan = retrieval_planning_agent(query_analysis, top_k)

            with st.spinner("Agent Step 4: Retrieving evidence..."):
                chunks = retrieve_chunks(
                    query=reformulated_query,
                    index=index,
                    metadata=metadata,
                    embed_model=embed_model,
                    top_k=retrieval_plan["top_k"],
                )

            if not chunks:
                st.error("No relevant evidence chunks found.")
                return

            with st.spinner("Agent Step 5: Synthesizing answer using QLoRA BioMistral..."):
                prompt = build_prompt(query, chunks, query_analysis)
                answer = generate_answer(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=max_new_tokens,
                )

            cleaned_answer = clean_display_text(answer)

            with st.spinner("Agent Step 6: Validating response..."):
                validation_result = validate_answer(cleaned_answer)

                if not validation_result["is_valid"]:
                    cleaned_answer = build_safe_evidence_answer(query, chunks)
                    validation_result = validate_answer(cleaned_answer)

                confidence = calculate_confidence(chunks, validation_result)

                try:
                    log_query_result(
                        query=query,
                        chunks=chunks,
                        confidence=confidence,
                        validation_score=validation_result["structure_score"],
                    )
                except Exception as e:
                    st.warning(f"Logging failed: {e}")

            st.success("Agentic evidence-based answer generated successfully.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Retrieval / format heuristic", confidence)
            col2.metric("Retrieved Evidence", len(chunks))
            col3.metric("Validation Score", validation_result["structure_score"])

            with st.expander("Agent Reasoning Trace", expanded=False):
                st.write("### Query Understanding")
                st.json(query_analysis)

                st.write("### Reformulated Query")
                st.write(reformulated_query)

                st.write("### Retrieval Plan")
                st.json(retrieval_plan)

                st.write("### Response Validation")
                st.json(validation_result)

            st.subheader("Generated Evidence-Based Answer")

            formatted_answer = cleaned_answer.replace(
                "Summary:", "### Summary\n"
            ).replace(
                "Key Findings:", "\n### Key Findings\n"
            ).replace(
                "Limitations:", "\n### Limitations\n"
            )

            st.markdown(formatted_answer)

            st.subheader("Study Comparison Table")
            comparison_df = build_study_comparison_table(chunks)
            st.dataframe(comparison_df, use_container_width=True)

            st.subheader("Retrieved Evidence Sources")

            for i, chunk in enumerate(chunks, start=1):
                with st.expander(
                    f"Evidence {i}: {chunk.get('title', '')} | {chunk.get('pmcid', '')} | {chunk.get('section', '')}"
                ):
                    st.write(f"**Similarity Score:** {chunk.get('score', 0):.4f}")
                    st.write(f"**Title:** {chunk.get('title', '')}")
                    st.write(f"**PMCID:** {chunk.get('pmcid', '')}")
                    st.write(f"**Journal:** {chunk.get('journal', '')}")
                    st.write(f"**Year:** {chunk.get('year', '')}")
                    st.write(f"**Section:** {chunk.get('section', '')}")
                    st.write("**Evidence Preview:**")
                    st.write(truncate_to_sentence(chunk.get("chunk_text", ""), 900))

    with tab2:
        st.subheader("Retrieval Evaluation")

        st.write(
            "This evaluates the retrieval layer using representative biomedical queries "
            "with Precision@K and NDCG@K."
        )
        
        eval_k = st.selectbox("Select K", [3, 5], index=0)

        if st.button("Run Retrieval Evaluation"):
            with st.spinner("Running retrieval evaluation..."):
                index, metadata, embed_model = load_retrieval_components()
                eval_df = run_retrieval_evaluation(
                    index=index,
                    metadata=metadata,
                    embed_model=embed_model,
                    k=eval_k,
                )

            st.dataframe(eval_df, use_container_width=True)

            csv_data = eval_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download Retrieval Evaluation CSV",
                data=csv_data,
                file_name=f"retrieval_evaluation_k{eval_k}.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader("Uploaded PDF Summarization")

        st.write(
            "Upload a biomedical PDF to generate a lightweight summary from the uploaded document. "
            "This feature summarizes the uploaded PDF text directly and does not modify the existing FAISS vector database."
        )

        uploaded_pdf = st.file_uploader(
            "Upload biomedical PDF",
            type=["pdf"]
        )

        if uploaded_pdf is not None:
            st.success(f"Uploaded file: {uploaded_pdf.name}")

            with st.spinner("Extracting text from uploaded PDF..."):
                uploaded_text = extract_text_from_uploaded_pdf(uploaded_pdf)

            st.write("Extracted text length:", len(uploaded_text), "characters")

            with st.expander("Preview extracted text"):
                st.write(uploaded_text[:1500])

            if st.button("Generate PDF Summary"):
                with st.spinner("Generating extractive PDF summary..."):
                    pdf_summary = summarize_uploaded_pdf(
                        pdf_text=uploaded_text
                    )

                pdf_summary = pdf_summary.strip()

                formatted_summary = pdf_summary.replace(
                    "Summary:", "### Summary\n"
                ).replace(
                    "Key Points:", "\n### Key Points\n"
                ).replace(
                    "Limitations:", "\n### Limitations\n"
                )

                st.subheader("Generated PDF Summary")
                st.markdown(formatted_summary)

if __name__ == "__main__":
    main()