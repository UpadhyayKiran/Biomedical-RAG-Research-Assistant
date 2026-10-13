# Publication review

The maintainer expressly authorized public publication of the reviewed independent-source copy under UpadhyayKiran, without a repository-wide license. Original academic files and repositories are preserved.

## Proposed GitHub details

- Name: `Biomedical-RAG-Research-Assistant`
- Description: `Biomedical RAG research prototype integrating BioMistral-7B, PubMedBERT, FAISS, QLoRA, a staged agent workflow, and retrieval evaluation.`
- Topics: `agentic-ai`, `rag`, `generative-ai`, `biomedical-nlp`, `python`, `faiss`, `biomistral`, `qlora`, `llm-evaluation`, `streamlit`

LangChain is removed from the proposed description/topics because it is not part of the final pipeline. The original proposed description's unsupported orchestration claim is not carried forward.

## Included

The original final Python implementation as package modules; extracted workflow helpers and UI entry point; shared configuration and examples; dependency specifications; acquisition, processing, indexing, training and evaluation modules; focused synthetic tests; historical retrieval CSVs and generation scores; a corpus identifier/license manifest; newly written documentation and Mermaid architecture; source provenance and an exact publication-file hash inventory.

## Excluded

All existing `.git` histories, earlier experiments and notebooks, teammate-named code folders, private course material, group reports/presentations, personal notes/contact details, query logs, PDFs, article-derived JSON/chunks, raw generation answers, training text, vector indexes, embedding metadata containing article passages, all checkpoints/model/adapter weights, tokenizer assets, caches, archives, original screenshots and original document figures. `.gitignore` reinforces exclusion but is not a substitute for reviewing the file inventory.

## Changes

Original algorithms are preserved in modules rather than reimplemented. Publication preparation adds shared environment-based paths, separates UI helpers, adds a source-aligned architecture, accurate evaluation and attribution notes, optional query logging, a research disclaimer, missing-artifact feedback, dependency cleanup and synthetic tests. The baseline FAISS CLI now rejects invalid result IDs; the index builder drops empty texts before both embedding and metadata construction to avoid positional mismatch. Historical measurements are retained, including duplicate exports, with raw generated passages removed from publishable copies.

The authenticated GitHub account was verified as UpadhyayKiran. Publication uses a fresh repository and an initial publication-copy commit, without inherited or backdated academic history.

## Review evidence

See [VALIDATION.md](VALIDATION.md), [SECURITY.md](SECURITY.md), [ATTRIBUTION.md](ATTRIBUTION.md), [LICENSING.md](LICENSING.md), [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) and [PUBLICATION_FILES.csv](PUBLICATION_FILES.csv). Local-only scan findings and original inventories are stored alongside this repository in `publication_review/`; they must not be pushed.

Publication is authorized on the maintainer's ownership declaration. No concrete applicable contractual restriction on the selected original source was identified. Unprovided agreements remain a documented uncertainty. Third-party assets stay excluded and no repository-wide license is added.
