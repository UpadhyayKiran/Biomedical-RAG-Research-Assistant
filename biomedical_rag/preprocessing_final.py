from . import config
import os
import re
import json
import pandas as pd
import pymupdf as fitz
from pathlib import Path


PDF_FOLDER = str(config.path('pdfs'))
METADATA_CSV = str(config.path('metadata.csv'))
OUTPUT_FOLDER = str(config.path('json'))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_metadata(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [col.strip().lower() for col in df.columns]

    if "pmcid" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'pmcid' column.")

    df["pmcid"] = df["pmcid"].astype(str).str.strip()
    df = df.fillna("")

    metadata_map = {}
    for _, row in df.iterrows():
        metadata_map[row["pmcid"]] = row.to_dict()

    print(f"Loaded metadata for {len(metadata_map)} records.")
    return metadata_map


def extract_text_from_pdf(pdf_path):
    text_blocks = []

    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_blocks.append(page_text)
        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

    return "\n".join(text_blocks)


def normalize_text(text):
    if not text:
        return ""

    text = text.replace("\r", "\n")

    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\xa0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\b[aA]1{5,}\b", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def remove_repeated_noise(text):
    cleaned_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            continue

        # Generic PDF running headers / footers
        if re.search(r"doi\.org|/\s*\d+\s*$", lower):
            continue

        # Page numbers like 1 / 12
        if re.match(r"^\d+\s*/\s*\d+$", stripped):
            continue

        # Dates like January 8, 2019
        if re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b.*\d{4}",
            lower,
        ):
            continue

        # Specific repeated running header seen in PMC6324822
        if re.search(r"costs of ct with anticancer biologic agents", lower):
            continue

        # More general repeated title/header pattern
        if "using the abc methodology" in lower and "cancer center" in lower:
            continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def infer_title(text, fallback_title=""):
    if fallback_title and str(fallback_title).strip():
        return str(fallback_title).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title_candidates = []

    for line in lines[:25]:
        lower = line.lower()

        if lower in {"research article", "open access", "abstract"}:
            continue

        if re.search(r"@|doi\.org|received:|accepted:|published:", lower):
            continue

        if len(line.split()) >= 4:
            title_candidates.append(line)

    if title_candidates:
        return max(title_candidates, key=len).strip()

    return ""


def clean_author_name(name):
    name = re.sub(r"\b\d+\b", "", name)
    name = re.sub(r"\bID\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[*†‡§¶#]+", "", name)
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip(" ,;-")


def extract_authors_from_text(text):
    first_chunk = "\n".join(text.splitlines()[:80])

    first_chunk = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        " ",
        first_chunk,
    )

    stop_words = ["abstract", "introduction", "aim", "background", "methods"]
    lines = [ln.strip() for ln in first_chunk.splitlines() if ln.strip()]

    candidate_lines = []
    for line in lines:
        if line.lower() in stop_words:
            break
        candidate_lines.append(line)

    block = " ".join(candidate_lines)
    raw_parts = re.split(r",|;", block)

    authors = []
    for part in raw_parts:
        part = clean_author_name(part)

        if 2 <= len(part.split()) <= 5 and re.match(
            r"^[A-Za-zÀ-ÖØ-öø-ÿ'`\-.\s]+$",
            part,
        ):
            if not re.search(
                r"\b(university|department|institute|hospital|napoli|italy|unit|college|school)\b",
                part,
                re.I,
            ):
                authors.append(part)

    unique_authors = []
    seen = set()

    for author in authors:
        key = author.lower()
        if author and key not in seen:
            seen.add(key)
            unique_authors.append(author)

    return unique_authors


def parse_authors(metadata):
    authors = []

    if "authors" in metadata and str(metadata["authors"]).strip():
        possible = str(metadata["authors"]).strip()

        if possible.startswith("[") and possible.endswith("]"):
            try:
                parsed = json.loads(possible)
                if isinstance(parsed, list):
                    authors = [str(a).strip() for a in parsed if str(a).strip()]
            except Exception:
                authors = []
        else:
            authors = [
                a.strip()
                for a in re.split(r";|,", possible)
                if a.strip()
            ]

    return authors


def cleanup_section_text(text):
    if not text:
        return ""

    junk_prefixes = [
        "open access",
        "citation:",
        "editor:",
        "received:",
        "accepted:",
        "published:",
        "copyright:",
        "data availability statement:",
        "funding:",
        "competing interests:",
    ]

    cleaned = []

    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()

        if not stripped:
            continue

        if any(lower.startswith(p) for p in junk_prefixes):
            continue

        if re.match(r"^https?://doi\.org/", lower):
            continue

        if re.match(r"^(fig|figure|table)\s*\d+", lower):
            continue

        # Remove repeated running headers inside sections
        if re.search(r"costs of ct with anticancer biologic agents", lower):
            continue

        if "using the abc methodology" in lower and "cancer center" in lower:
            continue

        # Remove table-like noisy rows
        if len(re.findall(r"\d+", stripped)) > 12 and len(stripped.split()) > 12:
            continue

        # Remove noisy all-caps table headers
        if stripped.isupper() and len(stripped.split()) > 10:
            continue

        cleaned.append(stripped)

    text = "\n".join(cleaned)

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def extract_abstract(text):
    match = re.search(
        r"\babstract\b(.*?)(\bintroduction\b|\bbackground\b|\bmethods\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return cleanup_section_text(match.group(1))

    pattern = re.search(
        r"\b(aim|objective|objectives)\b(.*?)(\bintroduction\b|\bbackground\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if pattern:
        return cleanup_section_text(pattern.group(0))

    return ""


SECTION_PATTERNS = {
    "abstract": [
        r"^\s*(\d+\.?\s*)?abstract\s*$",
    ],
    "introduction": [
        r"^\s*(\d+\.?\s*)?introduction\s*$",
        r"^\s*(\d+\.?\s*)?background\s*$",
    ],
    "methods": [
        r"^\s*(\d+\.?\s*)?methods\s*$",
        r"^\s*(\d+\.?\s*)?materials and methods\s*$",
        r"^\s*(\d+\.?\s*)?methodology\s*$",
        r"^\s*(\d+\.?\s*)?patients and methods\s*$",
        r"^\s*(\d+\.?\s*)?study design\s*$",
    ],
    "results": [
        r"^\s*(\d+\.?\s*)?results\s*$",
        r"^\s*(\d+\.?\s*)?findings\s*$",
    ],
    "discussion": [
        r"^\s*(\d+\.?\s*)?discussion\s*$",
    ],
    "conclusion": [
        r"^\s*(\d+\.?\s*)?conclusion[s]?\s*$",
        r"^\s*(\d+\.?\s*)?concluding remarks\s*$",
    ],
    "acknowledgments": [
        r"^\s*acknowledg?ments?\s*$",
    ],
    "references": [
        r"^\s*references\s*$",
        r"^\s*bibliography\s*$",
    ],
    "supplementary": [
        r"^\s*supporting information\s*$",
        r"^\s*supplementary information\s*$",
        r"^\s*supplementary material[s]?\s*$",
    ],
    "author_contributions": [
        r"^\s*author contributions\s*$",
        r"^\s*contributions\s*$",
    ],
}


def identify_heading(line):
    stripped = line.strip()

    if len(stripped.split()) > 8:
        return None

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, stripped, flags=re.IGNORECASE):
                return section

    return None


def split_into_sections(text):
    sections = []
    current_section = "unknown"
    buffer = []

    for line in text.splitlines():
        detected = identify_heading(line)

        if detected:
            section_text = "\n".join(buffer).strip()

            if section_text:
                sections.append(
                    {
                        "section": current_section,
                        "text": cleanup_section_text(section_text),
                    }
                )

            current_section = detected
            buffer = []
        else:
            buffer.append(line)

    section_text = "\n".join(buffer).strip()

    if section_text:
        sections.append(
            {
                "section": current_section,
                "text": cleanup_section_text(section_text),
            }
        )

    return [s for s in sections if s["text"].strip()]


def extract_front_matter(text):
    lower = text.lower()

    stops = []
    for marker in [
        "abstract",
        "introduction",
        "background",
        "aim",
        "methods",
        "materials and methods",
    ]:
        idx = lower.find(marker)
        if idx != -1:
            stops.append(idx)

    if not stops:
        return text[:3000]

    return text[: min(stops)].strip()


def refine_sections(sections, full_text, title):
    refined = []

    if title:
        refined.append(
            {
                "section": "title",
                "text": title,
            }
        )

    found_abstract = False

    for sec in sections:
        sec_name = sec["section"]
        sec_text = sec["text"].strip()

        if not sec_text:
            continue

        lower = sec_text.lower()

        # Remove duplicate front matter/title leakage
        if sec_name == "unknown":
            if "research article" in lower and title.lower()[:30] in lower:
                continue

            if title and sec_text.startswith(title[:40]):
                continue

            if len(sec_text.split()) < 25 and title.lower()[:20] in lower:
                continue

        # Remove author/affiliation pollution
        if sec_name in {"unknown", "methods"}:
            if re.search(r"@|university|department|institute|hospital", sec_text, re.I):
                if len(sec_text.split()) < 300:
                    continue

        # Skip sections that should not enter retrieval/training
        if sec_name in {
            "references",
            "supplementary",
            "acknowledgments",
            "author_contributions",
        }:
            continue

        # Remove very short noisy sections
        if len(sec_text.split()) < 40 and sec_name not in {"title", "abstract"}:
            continue

        # Remove license/copyright/front-matter pollution
        if re.search(
            r"creative commons|licensee|copyright|correspondence|received:|accepted:|published:|data availability|conflicts of interest|funding",
            sec_text,
            re.I,
        ):
            if len(sec_text.split()) < 250:
                continue

        # Remove table-heavy sections
        numeric_tokens = sum(
            1 for w in sec_text.split() if any(ch.isdigit() for ch in w)
        )
        numeric_ratio = numeric_tokens / max(len(sec_text.split()), 1)

        if numeric_ratio > 0.35:
            continue

        if sec_name == "abstract":
            found_abstract = True

        refined.append(
            {
                "section": sec_name,
                "text": sec_text,
            }
        )

    if not found_abstract:
        abs_text = extract_abstract(full_text)

        if abs_text:
            refined.insert(
                1,
                {
                    "section": "abstract",
                    "text": abs_text,
                },
            )

    return refined


def build_output_record(pmcid, pdf_filename, pdf_path, raw_text, metadata):
    raw_text = normalize_text(raw_text)
    raw_text = remove_repeated_noise(raw_text)

    title = infer_title(raw_text, metadata.get("title", ""))
    front_matter = extract_front_matter(raw_text)

    authors = parse_authors(metadata)

    if not authors:
        authors = extract_authors_from_text(front_matter)

    abstract = str(metadata.get("abstract", "")).strip()

    if not abstract:
        abstract = extract_abstract(raw_text)

    journal = str(
        metadata.get("journal_title", "")
        or metadata.get("journal", "")
    ).strip()

    keywords = str(
        metadata.get("keywords", "")
        or metadata.get("subjects", "")
    ).strip()

    sections = split_into_sections(raw_text)
    sections = refine_sections(sections, raw_text, title)

    record = {
        "pmcid": pmcid,
        "source": pdf_filename,
        "pdf_path": pdf_path,
        "title": title,
        "year": str(metadata.get("year", "")).strip(),
        "journal": journal,
        "authors": authors,
        "abstract": abstract,
        "keywords": keywords,
        "sections": sections,
    }

    return record


def process_all_pdfs(pdf_folder, metadata_map, output_folder):
    pdf_files = sorted(
        [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    )

    if not pdf_files:
        print("No PDF files found.")
        return

    processed = 0
    skipped = 0
    missing_metadata = 0

    for idx, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pmcid = Path(pdf_file).stem.strip()

        print(f"[{idx}/{len(pdf_files)}] Processing {pdf_file} ...")

        raw_text = extract_text_from_pdf(pdf_path)

        if not raw_text.strip():
            print(f"Skipping {pdf_file}: no text extracted.")
            skipped += 1
            continue

        metadata = metadata_map.get(pmcid, {})

        if not metadata:
            print(f"Warning: metadata not found for {pmcid}")
            missing_metadata += 1

        record = build_output_record(
            pmcid=pmcid,
            pdf_filename=pdf_file,
            pdf_path=pdf_path,
            raw_text=raw_text,
            metadata=metadata,
        )

        output_path = os.path.join(output_folder, f"{pmcid}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        processed += 1

    print("\nProcessing complete.")
    print(f"Processed PDFs: {processed}")
    print(f"Skipped PDFs: {skipped}")
    print(f"Missing metadata: {missing_metadata}")
    print(f"Output folder: {output_folder}")


if __name__ == "__main__":
    metadata_map = load_metadata(METADATA_CSV)
    process_all_pdfs(PDF_FOLDER, metadata_map, OUTPUT_FOLDER)