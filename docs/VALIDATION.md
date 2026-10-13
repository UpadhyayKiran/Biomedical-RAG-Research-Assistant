# Local validation — 13 October 2026

## Passed

- Six automated tests: rule-based query hints, format-only validation/fallback citations, historical NDCG behavior, actual FAISS retrieval diversity/padding, synthetic section chunking/index alignment, and workspace overrides.
- Tests passed both in the existing Python environment and a separate environment using Transformers 4.57.6. Warnings were FAISS/SWIG deprecations and a deprecated cache environment setting.
- All direct runtime packages imported in the existing environment.
- The cached biomedical embedding model loaded offline. The original index contained 2,267 vectors of dimension 768, aligned with 2,267 metadata records.
- The publication workflow's filtered retrieval evaluation ran on the original local artifacts and reproduced all five saved K=3 and K=5 summaries. Newly produced validation exports remain in the local review folder, separate from the historical CSVs.
- Streamlit's server started on loopback and returned HTTP 200 `ok` at its health endpoint. Streamlit AppTest executed the app with zero exceptions, displayed all three tabs and the research disclaimer, and verified the missing-artifact message on an attempted query.
- Python source parsing and relative documentation links are checked by `python -m tools.review_publication`. It also generates the candidate publication-file inventory and scans candidate text for selected credential patterns.
- Editable package installation (`pip install -e . --no-deps --no-build-isolation`) succeeded in the separate validation environment. A dependency-install dry run found all publication requirements satisfied there.

## Dependency findings

Existing Python was 3.12.6 with torch 2.6.0+cu124, Sentence Transformers 5.1.0, FAISS CPU 1.13.2, PEFT 0.19.1, bitsandbytes 0.49.2, Streamlit 1.49.1, pandas 2.3.3 and NumPy 1.26.4. Existing Transformers 5.6.2 violates Sentence Transformers 5.1.0's declared `<5` dependency despite the successful retrieval run.

The publication requirements therefore constrain Transformers to `>=4.41,<5` and Sentence Transformers to `>=5.1,<5.2`. Transformers 4.57.6 and Hugging Face Hub 0.36.2 were installed in a **separate local virtual environment**, with existing system packages visible; the global environment was not changed. Tests passed there. This is not a fully isolated clean-room install or a cross-platform lockfile.

Global `pip check` also found unrelated pre-existing conflicts involving LangChain, pubget/joblib and sklearn-compat/scikit-learn. Those packages are not required by this publication implementation. Their conflicts were not repaired in the user's environment. A fresh environment remains the recommended installation path.

## Not established

- **End-to-end model generation did not pass.** The offline attempt to load the 4-bit BioMistral base and original PEFT adapter on an NVIDIA RTX 4060 Laptop GPU (8 GiB) exited with a bitsandbytes CUDA out-of-memory error. A retry with the compatible Transformers environment also exited unsuccessfully before a generated response was confirmed. No generated-answer quality claim follows from these attempts.
- No training rerun, adapter reproduction, five-query generation rerun, latency benchmark or clinical validation was completed.
- No complete network download of the corpus was performed. Downloader CLI syntax and source were inspected; full article retrieval and metadata reconstruction were not validated end to end.
- A fresh dependency installation on another platform, Intel XPU/OpenVINO execution, and production deployment were not tested.
- No public GitHub rendering or accessibility checks can be performed before an approved publication. Markdown local links and code fences were reviewed locally; Mermaid rendering on GitHub remains a post-publication check.

Historical results are preserved with provenance, not treated as fresh successful generation validation. Full source-artifact checks and excluded-file review are described in SECURITY.md and PUBLICATION_REVIEW.md.
