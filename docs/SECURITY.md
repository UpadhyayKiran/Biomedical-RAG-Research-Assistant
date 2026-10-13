# Publication security review

The publication copy uses an explicit file selection. Dataset bodies, model files, archives, private academic documents, original screenshots, local environments, Git history and query logs are excluded. A newly written corpus manifest includes only identifiers, years, source links and recorded license labels/URLs. Original email literals were replaced with empty environment settings. Query logging is opt-in and writes only to the ignored local workspace.

`python -m tools.review_publication` scans every candidate UTF-8 publication file for selected Hugging Face, GitHub, OpenAI/AWS credential and private-key patterns, rejects disallowed file types and oversized files, parses Python, checks local Markdown links, and records paths/bytes/SHA-256 in `PUBLICATION_FILES.csv`. The inventory excludes its own hash to avoid circular hashing. This is a heuristic check, not a guarantee that every possible credential or personal datum is detectable.

An earlier broader workspace scan was stopped at the maintainer's request and is not claimed complete. Final publication checks are limited to the prepared publication directory and exact file inventory. Local review artifacts are outside this publication directory and must not be pushed.

Corpus PDFs, binary weights/checkpoints, media, archives, Git object history and oversized text are inventoried but are not cleared by content scanning. They are not in the proposed publication. Automated patient-marker matching does not establish the presence or absence of actual patient-identifiable information; the full original biomedical corpus is not represented as de-identified or privacy-cleared.

Do not upload patient documents or identifiers into the prototype. The UI processes uploaded PDF text locally in the Python process and does not add it to FAISS, but no clinical privacy, sandboxing or adversarial-document security assessment is claimed. The preserved model-loading code and dependencies require their own upstream trust and vulnerability review before deployment.

No credentials or personal-contact literals are intentionally included in the reviewed publication files. Security findings and the publication file list must be refreshed immediately before any approved commit/push. Any genuine credential found in excluded original material should be revoked at its provider; exclusion from this copy cannot undo earlier exposure.
