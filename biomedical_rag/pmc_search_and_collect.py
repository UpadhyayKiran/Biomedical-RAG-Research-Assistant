from . import config
import csv
import os
import re
import time
import html
from collections import defaultdict
from typing import Dict, List, Optional

import requests
import xml.etree.ElementTree as ET

BASE_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

EMAIL = config.get('NCBI_EMAIL', '')
API_KEY = config.get('NCBI_API_KEY', '')

KEYWORDS = [
    "clinical trial",
    "systematic review",
    "evidence-based medicine",
    "treatment outcomes",
    "biomedical literature",
]

TARGET_COUNT = 500
YEAR_START = 2019
YEAR_END = 2026
LANGUAGE = "english"
HUMANS_ONLY = True

# Balanced collection
PER_YEAR_LIMIT = 63
PER_KEYWORD_YEAR_LIMIT = 100

OUTPUT_CSV = str(config.path('pmc_ids.csv'))
LOG_DIR = str(config.path('logs'))
SEEN_PMCIDS_FILE = os.path.join(LOG_DIR, "seen_pmcids.txt")

REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.34


# ------------------ Utilities ------------------

def ensure_dirs():
    os.makedirs("data/raw_pdfs", exist_ok=True)
    os.makedirs("data/raw_xml", exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def request_get(url: str, params: Optional[Dict] = None):
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return resp


def strip_namespace(elem):
    for el in elem.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def normalize(text):
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def text(elem):
    if elem is None:
        return ""
    return normalize("".join(elem.itertext()))


# ------------------ Query ------------------

def build_query(keyword: str, year: int) -> str:
    parts = [
        f'("{keyword}")',
        f'"{LANGUAGE}"[Language]',
        f'("{year}/01/01"[Date - Publication] : "{year}/12/31"[Date - Publication])',
    ]
    if HUMANS_ONLY:
        parts.append('(humans[MeSH Terms] OR clinical[Title/Abstract] OR patient[Title/Abstract])')
    return " AND ".join(parts)


# ------------------ API ------------------

def esearch(query: str, retmax: int = PER_KEYWORD_YEAR_LIMIT) -> List[str]:
    url = f"{BASE_EUTILS}/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": query,
        "retmax": str(retmax),
        "retmode": "json",
        "sort": "relevance",
    }
    if EMAIL:
        params["email"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY

    data = request_get(url, params).json()
    return data.get("esearchresult", {}).get("idlist", [])


def efetch(ids: List[str]):
    url = f"{BASE_EUTILS}/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": ",".join(ids),
        "retmode": "xml",
    }
    if EMAIL:
        params["email"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY

    root = ET.fromstring(request_get(url, params).text)
    strip_namespace(root)
    return root


def esummary_pubmed(pmids: List[str]):
    url = f"{BASE_EUTILS}/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    if EMAIL:
        params["email"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY

    return request_get(url, params).json()


def extract_year_from_pubmed_summary(record: Dict) -> str:
    for field in ["pubdate", "epubdate", "sortpubdate"]:
        value = record.get(field, "")
        if value:
            match = re.search(r"\b(19\d{2}|20\d{2})\b", str(value))
            if match:
                return match.group(1)
    return ""


# ------------------ Extractors ------------------

def get_title(article):
    return text(article.find(".//article-title"))


def get_journal(article):
    return text(article.find(".//journal-title"))


def get_pmid(article):
    for aid in article.findall(".//article-id"):
        if aid.attrib.get("pub-id-type") == "pmid":
            return text(aid)
    return ""


def get_abstract(article):
    return text(article.find(".//abstract"))


def get_license(article):
    lic = article.find(".//license")
    if lic is not None:
        href = lic.attrib.get("{http://www.w3.org/1999/xlink}href", "")
        txt = text(lic)
        return href or txt
    return ""


# ------------------ Seen tracking ------------------

def load_seen():
    if not os.path.exists(SEEN_PMCIDS_FILE):
        return set()
    with open(SEEN_PMCIDS_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def save_seen(seen):
    with open(SEEN_PMCIDS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(seen)))


def write_csv(rows: List[Dict]):
    if not rows:
        return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ------------------ Main ------------------

def main():
    ensure_dirs()
    seen = load_seen()
    rows = []
    doc_id = 1
    collected_per_year = defaultdict(int)

    years = list(range(YEAR_START, YEAR_END + 1))

    for year in years:
        if len(rows) >= TARGET_COUNT:
            break

        print(f"\n===== YEAR {year} =====")
        year_collected = 0

        for keyword in KEYWORDS:
            if len(rows) >= TARGET_COUNT:
                break
            if year_collected >= PER_YEAR_LIMIT:
                break

            query = build_query(keyword, year)
            ids = esearch(query, retmax=PER_KEYWORD_YEAR_LIMIT)

            print(f"Keyword: {keyword} | Year: {year} | Found IDs: {len(ids)}")

            if not ids:
                continue

            root = efetch(ids)
            articles = root.findall(".//article")
            print(f"Articles fetched: {len(articles)}")

            batch_pmids = []
            for article in articles:
                pmid = get_pmid(article)
                batch_pmids.append(pmid if pmid else "")

            valid_pmids = [p for p in batch_pmids if p]
            pubmed_summary = esummary_pubmed(valid_pmids) if valid_pmids else {"result": {}}
            pubmed_map = pubmed_summary.get("result", {})

            for idx, article in enumerate(articles):
                if len(rows) >= TARGET_COUNT:
                    break
                if year_collected >= PER_YEAR_LIMIT:
                    break

                pmcid = f"PMC{ids[idx]}"

                if pmcid in seen:
                    continue

                title = get_title(article)
                pmid = get_pmid(article)

                if not title or not pmid:
                    continue

                summary_record = pubmed_map.get(pmid, {})
                actual_year = extract_year_from_pubmed_summary(summary_record)

                # Hard enforce exact year match
                if actual_year != str(year):
                    continue

                row = {
                    "doc_id": f"doc_{doc_id:04d}",
                    "title": title,
                    "pmcid": pmcid,
                    "pmid": pmid,
                    "journal": get_journal(article),
                    "publication_year": actual_year,
                    "article_type": article.attrib.get("article-type", ""),
                    "search_keyword": keyword,
                    "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
                    "abstract": get_abstract(article),
                    "license": get_license(article),
                    "pdf_filename": f"doc_{doc_id:04d}_{pmcid}.pdf",
                    "pdf_download_status": "pending",
                    "xml_filename": f"doc_{doc_id:04d}_{pmcid}.tgz",
                    "xml_download_status": "pending",
                }

                rows.append(row)
                seen.add(pmcid)
                collected_per_year[actual_year] += 1
                year_collected += 1

                print(f"Added: {pmcid} | PMID: {pmid} | Year: {actual_year}")
                doc_id += 1

                if len(rows) >= TARGET_COUNT:
                    break

    write_csv(rows)
    save_seen(seen)

    print(f"\nCollected {len(rows)} articles")
    print("Year distribution:")
    for y in sorted(collected_per_year.keys()):
        print(f"{y}: {collected_per_year[y]}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()