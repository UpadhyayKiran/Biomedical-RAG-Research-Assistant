# Implementation map

The Mermaid diagram in the README follows the final application, rather than proposed architectures in academic notes.

| Module | Responsibility |
| --- | --- |
| `pmc_search_and_collect` | E-Utilities search across keywords/years and metadata collection |
| `data_collection_pdf`, `data_collection_dir` | PMC Cloud PDF or article-prefix acquisition |
| `generate_metadata_new` | Merge available JSON/XML/text metadata |
| `metadata_balanced_pdf_filter`, `metadata_cleaning` | Balanced selection and text/date normalization |
| `preprocessing_final` | PDF text extraction, section parsing and cleaning |
| `create_chunks` | Overlapping section chunks, metadata preservation, noise/duplicate filtering |
| `create_faiss_index` | Sentence Transformer embeddings, normalized inner-product index, positional metadata |
| `prepare_qlora_dataset`, `qlora_train` | Synthetic evidence examples and PEFT training |
| `workflow` | Original UI's query rules, retrieval, prompts, generation, validation, fallback and evaluation |
| `ui` | Cached model/index loaders and Streamlit presentation |
| `config`, `logger_utils` | Shared local paths, environment settings and opt-in query logs |

The strict evidence filter considers section name, minimum length, lexical overlap and article diversity. The original relaxed fallback is nested within the strict candidate loop and does not repeat every strict filter. This behavior is preserved, not presented as a learned ranking algorithm.

The application loads the adapter when answering a question. Retrieval evaluation does not need the generation model. Uploaded-PDF summarization extracts selected sentences directly and does not use BioMistral or alter the index.

Helper extraction preserves the original algorithms. Publication changes include configurable paths, a missing-artifact message, opt-in logging, an invalid FAISS-ID guard in the baseline CLI, and filtering blank chunks before embedding/index/metadata construction. Historical results were not regenerated or replaced by these changes.
