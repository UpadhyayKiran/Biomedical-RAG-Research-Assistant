# Evaluation provenance and interpretation

`results/historical/` contains saved academic exports, not newly benchmarked model performance. The final source and result files were inspected directly. The duplicate `_1` exports are retained because they existed in the final folder; they are not independent experiments.

| Query | P@3 | P@5 | Implementation NDCG@3 | Implementation NDCG@5 |
| --- | --- | --- | --- | --- |
| Atrial fibrillation / mHealth | 1.00 | 1.00 | 1.00 | 1.00 |
| Oncology enrollment barriers | 1.00 | 1.00 | 1.00 | 1.00 |
| Remote patient monitoring | 1.00 | 0.80 | 1.00 | 1.00 |
| Clinical trial retention | 1.00 | 1.00 | 1.00 | 1.00 |
| Cardiovascular wearables | 1.00 | 1.00 | 1.00 | 1.00 |
| Mean | 1.00 | 0.96 | 1.00 | 1.00 |

These are five convenience queries, with no expert-adjudicated relevance judgments or independent benchmark split. A chunk passes the binary relevance rule if any expected keyword is found in its text or metadata. The remote-monitoring K=5 export includes one negative label. Since that negative is last, sorting the returned labels gives the same DCG and NDCG remains 1.00. This does not demonstrate optimal recall or global ranking.

`evaluate_retrieval` uses unfiltered FAISS top-3 results. The UI and `tools.evaluate_workflow` use filtered retrieval. Their keyword sets differ slightly. Compare like with like.

The saved generation metrics are **0.377, 0.403, 0.259, 0.371, 0.537**, giving a mean of **0.3894**. A 0.411 mean mentioned in other academic notes does not match these CSVs and is not used here. The original answer strings and concatenated titles are omitted from the publication metrics files; all numeric scores, boolean checks, query strings and retrieved PMCIDs are retained. Raw answer exports remain in the unchanged academic source.

The automated generation score is `0.35*structure + 0.35*word_overlap + 0.20*pmcid_presence + 0.10*length`. It checks superficial properties, not whether a finding is supported, whether the cited paper is correct, or whether the response is medically safe. The UI validator and fallback are a separate execution path and cannot be assumed to improve these saved backend scores.

For a defensible future benchmark, freeze article/model revisions, separate training and test questions, obtain expert graded judgments over the corpus, calculate IDCG from those judgments, and report citation correctness, abstention, latency and resource use separately.
