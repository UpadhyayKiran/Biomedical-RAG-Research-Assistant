from . import config
import csv
import os
from datetime import datetime


LOG_FILE = str(config.path('logs/query_logs.csv'))


def log_query_result(query, chunks, confidence, validation_score):
    if not config.LOG_QUERIES:
        return
    config.path("logs").mkdir(parents=True, exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)

    retrieved_pmcids = "; ".join(
        [str(chunk.get("pmcid", "")) for chunk in chunks]
    )

    row = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_pmcids": retrieved_pmcids,
        "confidence_score": confidence,
        "validation_score": validation_score,
        "retrieved_count": len(chunks),
    }

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)