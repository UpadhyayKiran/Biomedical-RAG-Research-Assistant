# Local data and artifact reconstruction

No article bodies or weights are distributed. `data/corpus_manifest.csv` identifies the original 152 articles and their recorded license labels/URLs. Exact historical bytes and article versions were not pinned in the original project; current downloads may differ or be unavailable. Rebuilt outputs need not exactly reproduce historical results.

Use [PMC's supported acquisition services](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) and inspect each article version's license before reuse. Do not scrape the main PMC article website in bulk. The academic downloader scripts use the PMC Cloud S3 endpoint, try versions 1–3, and do not enforce license conditions or exhaustively enumerate versions. The complete-prefix downloader can retrieve supplements as well. Treat these as preserved acquisition utilities, not a rights-clearance system.

For the original identifier set, after checking the intended use against article terms:

```powershell
python -m biomedical_rag.data_collection_dir data/corpus_manifest.csv workspace/pmc_dirs
python -m biomedical_rag.data_collection_pdf data/corpus_manifest.csv workspace/pdfs
python -m biomedical_rag.generate_metadata_new --strict-pdf-name
```

The downloader output paths above are CLI arguments. If `BIOMEDICAL_RAG_WORKSPACE` is overridden, replace `workspace/...` with that directory. The metadata builder uses the configured workspace and writes `pmc_dirs/metadata.csv`. Copy that CSV to `workspace/metadata.csv`; optionally run `python -m biomedical_rag.metadata_cleaning` and use its `metadata_cleaned.csv` output as `metadata.csv` after inspection. The PDF preprocessing module expects filenames such as `PMC10029859.pdf`, which the PDF-only downloader produces, and a metadata CSV with at least `pmcid`; `title`, `year`, `journal_title`/`journal`, `authors`, `abstract` and `keywords`/`subjects` improve quality.

On PowerShell, the metadata copy is:

```powershell
Copy-Item workspace/pmc_dirs/metadata.csv workspace/metadata.csv
```

The preserved balanced-selection module reads `workspace/pmc_dirs/metadata.csv` and writes `workspace/metadata.csv`, selecting up to 19 articles per year. It scores open-access metadata but does not enforce an open-access or permissive-license filter. Use it only after excluding ineligible articles. The historical search collector is optional and does not reproduce the fixed manifest; it also lacks an explicit OA search filter and originally maps fetched articles to requested IDs positionally. Recheck identifiers and rights before using its output.

Then run preprocessing, chunking, indexing, dataset preparation and training as listed in the README. Output locations are:

| Artifact | Default path |
| --- | --- |
| Processed sections | `workspace/json/*.json` |
| Chunks and report | `workspace/chunks/chunks.jsonl`, `chunking_report.csv` |
| Index and aligned metadata | `workspace/vectorstore/faiss_index.index`, `chunk_metadata.json` |
| Synthetic training examples | `workspace/qlora_dataset/biomistral_qlora_train.jsonl` |
| Trained PEFT adapter | `workspace/qlora_biomistral_adapter/` |
| New evaluation outputs | `workspace/results/` |

The base model and embedding model are obtained by their libraries from Hugging Face, subject to upstream terms and access. Model revisions were not fixed in the academic code. The app requires a local trained adapter; it does not silently fetch an old externally hosted adapter. Training can be rerun locally when hardware and data rights permit, but exact adapter reproduction is not guaranteed.

Original academic artifacts remain available only in the original local submission. Copying them into a private workspace for local validation does not clear them for publication. Do not upload the generated workspace, article text, queries or adapter without a separate rights and privacy review.
