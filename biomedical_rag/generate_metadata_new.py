from . import config
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse


# -----------------------------
# CONFIG
# -----------------------------
ROOT_DIR = config.path('pmc_dirs')
OUTPUT_CSV = config.path('pmc_dirs/metadata.csv')


# -----------------------------
# HELPERS
# -----------------------------
MONTH_MAP = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "sept": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def safe_read_text(path: Path, encoding="utf-8"):
    try:
        return path.read_text(encoding=encoding, errors="replace")
    except Exception:
        return ""


def normalize_whitespace(text):
    if text is None:
        return ""
    text = str(text).replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def yes_no(value: bool):
    return "yes" if value else "no"


def first_file_with_suffix(folder: Path, suffix: str):
    files = sorted(folder.glob(f"*{suffix}"))
    return files[0] if files else None


def extract_year_from_citation(citation: str):
    if not citation:
        return ""
    m = re.search(r"\b(19|20)\d{2}\b", citation)
    return m.group(0) if m else ""


def normalize_date_string(date_str: str):
    if not date_str:
        return ""

    s = normalize_whitespace(date_str)
    s = s.replace(",", "").replace(";", "").strip()

    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    m = re.match(r"^(\d{4})\s+([A-Za-z]+)\s+(\d{1,2})$", s)
    if m:
        y, mon, d = m.groups()
        mon_num = MONTH_MAP.get(mon.lower()[:4].rstrip("."), "") or MONTH_MAP.get(mon.lower()[:3], "")
        if mon_num:
            return f"{y}-{mon_num}-{int(d):02d}"

    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", s)
    if m:
        d, mon, y = m.groups()
        mon_num = MONTH_MAP.get(mon.lower()[:4].rstrip("."), "") or MONTH_MAP.get(mon.lower()[:3], "")
        if mon_num:
            return f"{y}-{mon_num}-{int(d):02d}"

    return ""


def parse_received_accepted_from_txt_line(line: str):
    received_date = ""
    accepted_date = ""

    if not line:
        return received_date, accepted_date

    rec_match = re.search(r"Received\s+([^;]+)", line, flags=re.IGNORECASE)
    acc_match = re.search(r"Accepted\s+([^;]+)", line, flags=re.IGNORECASE)

    if rec_match:
        received_date = normalize_date_string(rec_match.group(1))
    if acc_match:
        accepted_date = normalize_date_string(acc_match.group(1))

    return received_date, accepted_date


def strip_at_first_marker(text, markers):
    if not text:
        return text

    cut_positions = []
    lower_text = text.lower()

    for marker in markers:
        pos = lower_text.find(marker.lower())
        if pos != -1:
            cut_positions.append(pos)

    if cut_positions:
        text = text[:min(cut_positions)]

    return text


def clean_abstract(text: str):
    """
    Clean abstract text and aggressively remove section bleed.
    """
    if not text:
        return ""

    text = text.replace("\ufeff", " ")
    text = text.replace("\x0c", " ")
    text = text.replace("", " ")
    text = re.sub(r"[\x01-\x08\x0b-\x1f]", " ", text)

    text = re.sub(r"^\s*abstract\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*summary\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)

    markers = [
        " 1. introduction",
        "\n1. introduction",
        " introduction ",
        " keywords:",
        " funding",
        " acknowledg",
        " references",
        " supplementary",
        " conflict of interest",
        " declaration of",
        " ethics statement",
        " data availability",
        " author contributions",
        " copyright",
        " received ",
        " accepted ",
        " published ",
        " pmcid:",
        " doi:",
        " http://",
        " https://",
    ]
    text = strip_at_first_marker(text, markers)

    text = re.sub(r"\s+\[\d+(?:,\d+)*\]\s*$", "", text)
    text = re.sub(r"\s+\d+\.\s+[A-Z][^.]{10,}$", "", text)

    text = normalize_whitespace(text)

    if len(text) > 4000:
        aggressive_markers = [
            "in this study,",
            "materials and methods",
            "methods",
            "methodology",
            "results",
            "conclusions",
        ]
        lower_text = text.lower()
        cut_positions = []
        for marker in aggressive_markers:
            pos = lower_text.find(marker.lower(), 1200)
            if pos != -1:
                cut_positions.append(pos)
        if cut_positions:
            text = text[:min(cut_positions)]
            text = normalize_whitespace(text)

    if len(text) > 5000:
        text = text[:5000].rsplit(" ", 1)[0].strip()

    return text


def extract_txt_abstract(text: str):
    """
    Better txt fallback abstract extraction.
    """
    if not text:
        return ""

    patterns = [
        r"Abstract\s*[:\-]?\s*(.*?)\n\s*1\.\s*Introduction",
        r"Abstract\s*[:\-]?\s*(.*?)\n\s*Introduction",
        r"Keywords\s*:[^\n]*\n+(.*?)\n\s*1\.\s*Introduction",
        r"Keywords\s*:[^\n]*\n+(.*?)\n\s*Introduction",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            candidate = clean_abstract(m.group(1))
            if candidate:
                return candidate

    m = re.search(r"\bAbstract\b\s*[:\-]?\s*", text, re.IGNORECASE)
    if m:
        start = m.end()
        candidate = text[start:start + 8000]
        candidate = clean_abstract(candidate)
        return candidate

    return ""


def extract_version_from_folder_name(folder_name: str):
    m = re.match(r"^(PMC\d+)\.(\d+)$", folder_name)
    if m:
        return m.group(2)
    return ""


def expected_pdf_filename(pmcid: str, version: str):
    if pmcid and version:
        return f"{pmcid}.{version}.pdf"
    return ""


def pdf_name_matches_expected(pdf_file: Path, pmcid: str, version: str):
    if pdf_file is None:
        return False
    expected_name = expected_pdf_filename(pmcid, version)
    if not expected_name:
        return False
    return pdf_file.name == expected_name


# -----------------------------
# TXT PARSER
# -----------------------------
def parse_txt_metadata(txt_path: Path):
    data = {
        "journal_title": "",
        "journal_id": "",
        "publisher": "",
        "issn": "",
        "eissn": "",
        "article_id": "",
        "article_version": "",
        "subjects": "",
        "electronic_publication_date": "",
        "print_publication_date": "",
        "volume": "",
        "issue": "",
        "elocation_id": "",
        "received_date": "",
        "accepted_date": "",
        "license_text": "",
        "license_url": "",
        "keywords": "",
        "abstract": "",
        "authors": "",
    }

    if not txt_path or not txt_path.exists():
        return data

    text = safe_read_text(txt_path)
    lines = [line.strip() for line in text.splitlines()]
    nonempty = [line for line in lines if line.strip()]

    def get_first_value(prefix):
        for line in nonempty:
            if line.startswith(prefix):
                return normalize_whitespace(line.split(":", 1)[1] if ":" in line else "")
        return ""

    if "ARTICLE INFORMATION" in text and "International Journal of Environmental Research and Public Health" in text:
        data["journal_title"] = "International Journal of Environmental Research and Public Health"

    data["journal_id"] = get_first_value("Journal ID")
    data["publisher"] = get_first_value("Publisher")
    data["issn"] = get_first_value("ISSN")
    data["eissn"] = get_first_value("EISSN")
    data["article_id"] = get_first_value("Article ID")
    data["article_version"] = get_first_value("Article version")
    data["subjects"] = get_first_value("Subjects")
    data["electronic_publication_date"] = get_first_value("Electronic publication date")
    data["print_publication_date"] = get_first_value("Print publication date")
    data["volume"] = get_first_value("Volume")
    data["issue"] = get_first_value("Issue")
    data["elocation_id"] = get_first_value("Electronic Location ID")
    data["license_text"] = get_first_value("License")
    data["license_url"] = get_first_value("License URL")

    for line in nonempty:
        if line.startswith("Received "):
            rec, acc = parse_received_accepted_from_txt_line(line)
            data["received_date"] = rec
            data["accepted_date"] = acc
            break

    title = ""
    try:
        subj_idx = next(i for i, line in enumerate(nonempty) if line.startswith("Subjects:"))
        for j in range(subj_idx + 1, min(subj_idx + 8, len(nonempty))):
            candidate = nonempty[j]
            if ":" not in candidate and candidate != "ARTICLE INFORMATION":
                title = candidate
                break
    except StopIteration:
        pass

    authors = []
    if title:
        try:
            title_idx = nonempty.index(title)
            for j in range(title_idx + 1, len(nonempty)):
                line = nonempty[j]
                if re.match(r"^\d+\s", line) or line.startswith("* Correspondence:") or line.startswith("Electronic publication date:"):
                    break
                if line:
                    authors.append(line)
        except ValueError:
            pass

    data["authors"] = normalize_whitespace("; ".join(authors))

    for line in nonempty:
        if line.startswith("Keywords:"):
            data["keywords"] = normalize_whitespace(line.split(":", 1)[1] if ":" in line else "")
            break

    data["abstract"] = extract_txt_abstract(text)
    return data


# -----------------------------
# XML PARSER
# -----------------------------
def parse_xml_metadata(xml_path: Path):
    data = {
        "article_type": "",
        "journal_title": "",
        "publisher": "",
        "issn": "",
        "eissn": "",
        "pmcid": "",
        "pmcid_version": "",
        "pmid": "",
        "doi": "",
        "publisher_id": "",
        "title": "",
        "authors": "",
        "electronic_publication_date": "",
        "print_publication_date": "",
        "volume": "",
        "issue": "",
        "elocation_id": "",
        "received_date": "",
        "accepted_date": "",
        "license_code": "",
        "license_url": "",
        "keywords": "",
        "abstract": "",
        "xml_has_pdf_flag": "",
    }

    if not xml_path or not xml_path.exists():
        return data

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return data

    def find_text(elem, path):
        found = elem.find(path)
        return normalize_whitespace(found.text if found is not None and found.text else "")

    data["article_type"] = normalize_whitespace(root.attrib.get("article-type", ""))

    front = root.find("front")
    if front is None:
        return data

    journal_meta = front.find("journal-meta")
    article_meta = front.find("article-meta")

    if journal_meta is not None:
        data["journal_title"] = find_text(journal_meta, "journal-title-group/journal-title")
        data["publisher"] = find_text(journal_meta, "publisher/publisher-name")

        for issn in journal_meta.findall("issn"):
            pub_type = issn.attrib.get("pub-type", "").lower()
            val = normalize_whitespace(issn.text)
            if pub_type == "ppub":
                data["issn"] = val
            elif pub_type == "epub":
                data["eissn"] = val

    if article_meta is not None:
        for aid in article_meta.findall("article-id"):
            pub_id_type = aid.attrib.get("pub-id-type", "")
            val = normalize_whitespace(aid.text)
            if pub_id_type == "pmcid":
                data["pmcid"] = val
            elif pub_id_type == "pmcid-ver":
                data["pmcid_version"] = val
            elif pub_id_type == "pmid":
                data["pmid"] = val
            elif pub_id_type == "doi":
                data["doi"] = val
            elif pub_id_type == "publisher-id":
                data["publisher_id"] = val

        title_elem = article_meta.find("title-group/article-title")
        if title_elem is not None:
            data["title"] = normalize_whitespace("".join(title_elem.itertext()))

        author_names = []
        contrib_group = article_meta.find("contrib-group")
        if contrib_group is not None:
            for contrib in contrib_group.findall("contrib"):
                if contrib.attrib.get("contrib-type") == "author":
                    surname = find_text(contrib, "name/surname")
                    given = find_text(contrib, "name/given-names")
                    full_name = normalize_whitespace(f"{given} {surname}")
                    if full_name:
                        author_names.append(full_name)
        data["authors"] = "; ".join(author_names)

        epub = article_meta.find("pub-date[@pub-type='epub']")
        if epub is not None:
            y = find_text(epub, "year")
            m = find_text(epub, "month")
            d = find_text(epub, "day")
            if y and m and d:
                data["electronic_publication_date"] = f"{y}-{int(m):02d}-{int(d):02d}"
            elif y and m:
                data["electronic_publication_date"] = f"{y}-{int(m):02d}"
            elif y:
                data["electronic_publication_date"] = y

        ppub = article_meta.find("pub-date[@pub-type='ppub']")
        if ppub is not None:
            y = find_text(ppub, "year")
            m = find_text(ppub, "month")
            d = find_text(ppub, "day")
            if y and m and d:
                data["print_publication_date"] = f"{y}-{int(m):02d}-{int(d):02d}"
            elif y and m:
                data["print_publication_date"] = f"{y}-{int(m):02d}"
            elif y:
                data["print_publication_date"] = y

        data["volume"] = find_text(article_meta, "volume")
        data["issue"] = find_text(article_meta, "issue")
        data["elocation_id"] = find_text(article_meta, "elocation-id")

        history = article_meta.find("history")
        if history is not None:
            received = history.find("date[@date-type='received']")
            accepted = history.find("date[@date-type='accepted']")

            if received is not None:
                ry = find_text(received, "year")
                rm = find_text(received, "month")
                rd = find_text(received, "day")
                if ry and rm and rd:
                    data["received_date"] = f"{ry}-{int(rm):02d}-{int(rd):02d}"

            if accepted is not None:
                ay = find_text(accepted, "year")
                am = find_text(accepted, "month")
                ad = find_text(accepted, "day")
                if ay and am and ad:
                    data["accepted_date"] = f"{ay}-{int(am):02d}-{int(ad):02d}"

        abstract = article_meta.find("abstract")
        if abstract is not None:
            paragraphs = []
            for p in abstract.findall(".//p"):
                p_text = normalize_whitespace("".join(p.itertext()))
                if p_text:
                    paragraphs.append(p_text)
            if paragraphs:
                data["abstract"] = clean_abstract(" ".join(paragraphs))
            else:
                abstract_text = normalize_whitespace("".join(abstract.itertext()))
                data["abstract"] = clean_abstract(abstract_text)

        kwd_group = article_meta.find("kwd-group")
        if kwd_group is not None:
            keywords = []
            for kw in kwd_group.findall("kwd"):
                kw_text = normalize_whitespace("".join(kw.itertext()))
                if kw_text:
                    keywords.append(kw_text)
            data["keywords"] = "; ".join(keywords)

        custom_meta_group = article_meta.find("custom-meta-group")
        if custom_meta_group is not None:
            for cm in custom_meta_group.findall("custom-meta"):
                meta_name = find_text(cm, "meta-name")
                meta_value = find_text(cm, "meta-value")
                if meta_name == "pmc-license-ref":
                    data["license_code"] = meta_value
                elif meta_name == "pmc-prop-has-pdf":
                    data["xml_has_pdf_flag"] = meta_value

        permissions = article_meta.find("permissions")
        if permissions is not None:
            for elem in permissions.iter():
                tag = elem.tag.split("}")[-1]
                if tag == "license_ref" and elem.text:
                    data["license_url"] = normalize_whitespace(elem.text)
                    break

    return data


# -----------------------------
# JSON PARSER
# -----------------------------
def load_json_metadata(json_path: Path):
    if not json_path or not json_path.exists():
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# -----------------------------
# BUILD RECORD
# -----------------------------
def build_record(folder: Path):
    folder_name = folder.name

    json_file = first_file_with_suffix(folder, ".json")
    txt_file = first_file_with_suffix(folder, ".txt")
    xml_file = first_file_with_suffix(folder, ".xml")
    pdf_file = first_file_with_suffix(folder, ".pdf")

    jpg_count = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.jpeg"))) + len(list(folder.glob("*.png")))

    jmeta = load_json_metadata(json_file)
    tmeta = parse_txt_metadata(txt_file)
    xmeta = parse_xml_metadata(xml_file)

    pmcid = jmeta.get("pmcid") or xmeta.get("pmcid") or re.sub(r"\.\d+$", "", folder_name)

    version = ""
    if jmeta.get("version") not in [None, ""]:
        version = str(jmeta.get("version"))
    elif xmeta.get("pmcid_version"):
        version = xmeta.get("pmcid_version", "").split(".")[-1]
    else:
        version = extract_version_from_folder_name(folder_name)

    title = jmeta.get("title") or xmeta.get("title") or ""
    citation = normalize_whitespace(jmeta.get("citation", ""))
    year = extract_year_from_citation(citation)

    abstract = xmeta.get("abstract") or tmeta.get("abstract") or ""
    abstract = clean_abstract(abstract)

    received_date = xmeta.get("received_date") or tmeta.get("received_date") or ""
    accepted_date = xmeta.get("accepted_date") or tmeta.get("accepted_date") or ""

    expected_pdf_name = expected_pdf_filename(pmcid, version)
    pdf_name_matches = pdf_name_matches_expected(pdf_file, pmcid, version)

    record = {
        "folder_name": folder_name,
        "folder_path": str(folder),
        "pmcid": pmcid,
        "version": version,
        "pmid": jmeta.get("pmid") or xmeta.get("pmid", ""),
        "doi": jmeta.get("doi") or xmeta.get("doi", ""),
        "title": normalize_whitespace(title),
        "citation": citation,
        "year": year,
        "journal_title": tmeta.get("journal_title") or xmeta.get("journal_title", ""),
        "journal_id": tmeta.get("journal_id", ""),
        "publisher": tmeta.get("publisher") or xmeta.get("publisher", ""),
        "issn": tmeta.get("issn") or xmeta.get("issn", ""),
        "eissn": tmeta.get("eissn") or xmeta.get("eissn", ""),
        "article_type": xmeta.get("article_type") or jmeta.get("article_type", ""),
        "subjects": tmeta.get("subjects", ""),
        "article_id": tmeta.get("article_id") or xmeta.get("publisher_id", ""),
        "authors": xmeta.get("authors") or tmeta.get("authors", ""),
        "electronic_publication_date": tmeta.get("electronic_publication_date") or xmeta.get("electronic_publication_date", ""),
        "print_publication_date": tmeta.get("print_publication_date") or xmeta.get("print_publication_date", ""),
        "volume": tmeta.get("volume") or xmeta.get("volume", ""),
        "issue": tmeta.get("issue") or xmeta.get("issue", ""),
        "elocation_id": tmeta.get("elocation_id") or xmeta.get("elocation_id", ""),
        "received_date": received_date,
        "accepted_date": accepted_date,
        "license_code": jmeta.get("license_code") or xmeta.get("license_code", ""),
        "license_text": tmeta.get("license_text", ""),
        "license_url": tmeta.get("license_url") or xmeta.get("license_url", ""),
        "is_pmc_openaccess": jmeta.get("is_pmc_openaccess", ""),
        "is_manuscript": jmeta.get("is_manuscript", ""),
        "is_historical_ocr": jmeta.get("is_historical_ocr", ""),
        "is_retracted": jmeta.get("is_retracted", ""),
        "keywords": tmeta.get("keywords") or xmeta.get("keywords", ""),
        "abstract": abstract,
        "pdf_present": yes_no(pdf_file is not None),
        "json_present": yes_no(json_file is not None),
        "txt_present": yes_no(txt_file is not None),
        "xml_present": yes_no(xml_file is not None),
        "image_count": jpg_count,
        "pdf_file": pdf_file.name if pdf_file else "",
        "json_file": json_file.name if json_file else "",
        "txt_file": txt_file.name if txt_file else "",
        "xml_file": xml_file.name if xml_file else "",
        "pdf_path": str(pdf_file) if pdf_file else "",
        "json_path": str(json_file) if json_file else "",
        "txt_path": str(txt_file) if txt_file else "",
        "xml_path": str(xml_file) if xml_file else "",
        "xml_url": jmeta.get("xml_url", ""),
        "pdf_url": jmeta.get("pdf_url", ""),
        "text_url": jmeta.get("text_url", ""),
        "media_urls_count": len(jmeta.get("media_urls", [])) if isinstance(jmeta.get("media_urls", []), list) else 0,
        "expected_pdf_name": expected_pdf_name,
        "pdf_name_matches_pmcid_version": yes_no(pdf_name_matches),
    }

    return record


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Build PMC metadata CSV.")
    parser.add_argument(
        "--strict-pdf-name",
        action="store_true",
        help="Include row only if PDF filename exactly matches PMCID.version.pdf"
    )
    args = parser.parse_args()

    if not ROOT_DIR.exists():
        print(f"Root directory does not exist: {ROOT_DIR}")
        return

    folders = [p for p in ROOT_DIR.iterdir() if p.is_dir()]
    if not folders:
        print(f"No PMC directories found in: {ROOT_DIR}")
        return

    records = []
    skipped_due_to_pdf_name = 0

    for folder in sorted(folders):
        try:
            record = build_record(folder)

            if args.strict_pdf_name and record["pdf_name_matches_pmcid_version"] != "yes":
                skipped_due_to_pdf_name += 1
                print(
                    f"Skipped: {folder.name} | PDF name mismatch | "
                    f"found='{record['pdf_file']}' | expected='{record['expected_pdf_name']}'"
                )
                continue

            records.append(record)
            print(f"Processed: {folder.name}")
        except Exception as e:
            print(f"Failed: {folder.name} -> {e}")

    if not records:
        print("No metadata records created.")
        print(f"Skipped due to strict PDF name rule: {skipped_due_to_pdf_name}")
        return

    fieldnames = list(records[0].keys())

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nDone. Metadata CSV saved at:\n{OUTPUT_CSV}")
    print(f"Total folders written: {len(records)}")
    print(f"Skipped due to strict PDF name rule: {skipped_due_to_pdf_name}")


if __name__ == "__main__":
    main()