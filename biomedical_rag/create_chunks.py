from . import config
import os
import json
import csv
import hashlib
from pathlib import Path


# =========================
# CONFIG
# =========================
JSON_FOLDER = str(config.path('json'))
OUTPUT_FOLDER = str(config.path('chunks'))

CHUNKS_JSONL = str(config.path('chunks/chunks.jsonl'))
CHUNKING_REPORT = os.path.join(OUTPUT_FOLDER, "chunking_report.csv")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

MIN_CHUNK_WORDS = 60
MIN_ALPHA_WORDS = 40

SKIP_SECTIONS = {
    "title",
    "references",
    "supplementary",
    "acknowledgments",
    "author_contributions",
}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_chunk_text(text):
    if not text:
        return ""

    text = str(text)
    text = " ".join(text.split())
    return text.strip()


def split_text_into_chunks(text, chunk_size=500, overlap=100):
    words = text.split()

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def is_table_heavy(text):
    words = text.split()

    if len(words) < 40:
        return True

    numeric_tokens = sum(1 for w in words if any(ch.isdigit() for ch in w))
    numeric_ratio = numeric_tokens / max(len(words), 1)

    return numeric_ratio > 0.35


def is_low_information_chunk(text):
    words = text.split()

    if len(words) < MIN_CHUNK_WORDS:
        return True

    alphabetic_words = [
        w for w in words
        if any(ch.isalpha() for ch in w)
    ]

    if len(alphabetic_words) < MIN_ALPHA_WORDS:
        return True

    return False


def is_noisy_unknown_chunk(text):
    lower = text.lower()

    noise_patterns = [
        "academic editor",
        "licensee mdpi",
        "creative commons",
        "creativecommons",
        "correspondence:",
        "received:",
        "accepted:",
        "published:",
        "copyright",
        "doi.org",
        "www.mdpi.com",
        "journal of clinical medicine article",
        "open access article",
        "distributed under the terms",
        "department of",
        "university",
        "institute",
        "faculty of",
        "correspondence",
        "conflicts of interest",
        "data availability statement",
        "funding",
        "publisher's note",
    ]

    return any(pattern in lower for pattern in noise_patterns)


def normalize_for_duplicate_check(text):
    normalized = text.lower().strip()
    normalized = " ".join(normalized.split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def build_chunk_id(pmcid, section_name, chunk_number):
    safe_section = section_name.lower().replace(" ", "_")
    return f"{pmcid}_{safe_section}_{chunk_number:04d}"


def write_chunk(
    out_f,
    pmcid,
    title,
    journal,
    year,
    authors,
    keywords,
    section_name,
    chunk_index,
    chunk_text,
    source_pdf,
    pdf_path,
):
    chunk = {
        "chunk_id": build_chunk_id(pmcid, section_name, chunk_index),
        "pmcid": pmcid,
        "title": title,
        "journal": journal,
        "year": year,
        "authors": authors,
        "keywords": keywords,
        "section": section_name,
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "source_pdf": source_pdf,
        "pdf_path": pdf_path,
    }

    out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def create_chunks():
    json_files = sorted(Path(JSON_FOLDER).glob("*.json"))

    if not json_files:
        print("No JSON files found.")
        return

    total_chunks = 0
    skipped_noisy_unknown = 0
    skipped_table_heavy = 0
    skipped_low_information = 0
    skipped_duplicates = 0
    skipped_missing_metadata = 0

    report_rows = []
    seen_chunk_texts = set()

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as out_f:
        for file_index, json_file in enumerate(json_files, start=1):
            print(f"[{file_index}/{len(json_files)}] Processing {json_file.name}")

            record = load_json_file(json_file)

            pmcid = str(record.get("pmcid", "")).strip()
            title = str(record.get("title", "")).strip()
            journal = str(record.get("journal", "")).strip()
            year = str(record.get("year", "")).strip()
            authors = record.get("authors", [])
            keywords = str(record.get("keywords", "")).strip()
            abstract = str(record.get("abstract", "")).strip()
            source_pdf = str(record.get("source", "")).strip()
            pdf_path = str(record.get("pdf_path", "")).strip()

            if not pmcid or not title:
                skipped_missing_metadata += 1
                continue

            doc_chunk_count = 0
            section_chunk_tracker = {}

            # Add top-level abstract as one chunk
            if abstract:
                abstract_text = clean_chunk_text(abstract)

                if is_table_heavy(abstract_text):
                    skipped_table_heavy += 1
                elif is_low_information_chunk(abstract_text):
                    skipped_low_information += 1
                else:
                    duplicate_key = normalize_for_duplicate_check(abstract_text)

                    if duplicate_key in seen_chunk_texts:
                        skipped_duplicates += 1
                    else:
                        seen_chunk_texts.add(duplicate_key)

                        write_chunk(
                            out_f=out_f,
                            pmcid=pmcid,
                            title=title,
                            journal=journal,
                            year=year,
                            authors=authors,
                            keywords=keywords,
                            section_name="abstract",
                            chunk_index=1,
                            chunk_text=abstract_text,
                            source_pdf=source_pdf,
                            pdf_path=pdf_path,
                        )

                        total_chunks += 1
                        doc_chunk_count += 1

            for section in record.get("sections", []):
                section_name = str(section.get("section", "")).strip().lower()
                section_text = clean_chunk_text(section.get("text", ""))

                if not section_name or not section_text:
                    continue

                if section_name in SKIP_SECTIONS:
                    continue

                # Avoid duplicate abstract because top-level abstract is already added
                if section_name == "abstract":
                    continue

                # Remove noisy front-matter chunks that appear as unknown
                if section_name == "unknown" and is_noisy_unknown_chunk(section_text):
                    skipped_noisy_unknown += 1
                    continue

                if is_table_heavy(section_text):
                    skipped_table_heavy += 1
                    continue

                text_chunks = split_text_into_chunks(
                    section_text,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )

                for chunk_text in text_chunks:
                    chunk_text = clean_chunk_text(chunk_text)

                    if not chunk_text:
                        continue

                    if section_name == "unknown" and is_noisy_unknown_chunk(chunk_text):
                        skipped_noisy_unknown += 1
                        continue

                    if is_table_heavy(chunk_text):
                        skipped_table_heavy += 1
                        continue

                    if is_low_information_chunk(chunk_text):
                        skipped_low_information += 1
                        continue

                    duplicate_key = normalize_for_duplicate_check(chunk_text)

                    if duplicate_key in seen_chunk_texts:
                        skipped_duplicates += 1
                        continue

                    seen_chunk_texts.add(duplicate_key)

                    section_chunk_tracker[section_name] = (
                        section_chunk_tracker.get(section_name, 0) + 1
                    )

                    chunk_number = section_chunk_tracker[section_name]

                    write_chunk(
                        out_f=out_f,
                        pmcid=pmcid,
                        title=title,
                        journal=journal,
                        year=year,
                        authors=authors,
                        keywords=keywords,
                        section_name=section_name,
                        chunk_index=chunk_number,
                        chunk_text=chunk_text,
                        source_pdf=source_pdf,
                        pdf_path=pdf_path,
                    )

                    total_chunks += 1
                    doc_chunk_count += 1

            report_rows.append(
                {
                    "pmcid": pmcid,
                    "title": title,
                    "year": year,
                    "journal": journal,
                    "chunks_created": doc_chunk_count,
                    "source_file": json_file.name,
                }
            )

    with open(CHUNKING_REPORT, "w", newline="", encoding="utf-8-sig") as csv_f:
        fieldnames = [
            "pmcid",
            "title",
            "year",
            "journal",
            "chunks_created",
            "source_file",
        ]

        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    avg_chunks_per_doc = round(total_chunks / max(len(json_files), 1), 2)

    print("\nChunking complete.")
    print(f"JSON files processed: {len(json_files)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Average chunks per document: {avg_chunks_per_doc}")
    print(f"Noisy unknown chunks skipped: {skipped_noisy_unknown}")
    print(f"Table-heavy chunks skipped: {skipped_table_heavy}")
    print(f"Low-information chunks skipped: {skipped_low_information}")
    print(f"Duplicate chunks skipped: {skipped_duplicates}")
    print(f"Records skipped due to missing PMCID/title: {skipped_missing_metadata}")
    print(f"Chunks saved to: {CHUNKS_JSONL}")
    print(f"Report saved to: {CHUNKING_REPORT}")


if __name__ == "__main__":
    create_chunks()