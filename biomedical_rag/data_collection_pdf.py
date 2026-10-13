from . import config
import os
import sys
import time
import csv
from pathlib import Path

import requests

# S3 HTTPS endpoint for PMC Open Access data on AWS
# We will construct URLs like:
# https://pmc-oa-opendata.s3.amazonaws.com/PMC6824378.1/PMC6824378.1.pdf
S3_BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com/{prefix}/{filename}"


def check_s3_connectivity(session: requests.Session, timeout: float = 5.0) -> bool:
    """Quickly check whether the PMC S3 endpoint is reachable.

    This helps detect company firewall / proxy blocks upfront.
    """

    test_url = "https://pmc-oa-opendata.s3.amazonaws.com/README.txt"
    print(f"Checking connectivity to {test_url} ...")
    try:
        resp = session.get(test_url, timeout=timeout)
    except requests.RequestException as exc:
        print(" ERROR: Could not reach PMC S3 endpoint.")
        print(f" Details: {exc}")
        print(" This is likely a firewall or proxy issue on this network.")
        return False

    if resp.status_code != 200:
        print(f" WARNING: Got HTTP {resp.status_code} from PMC S3 endpoint.")
        print(" This may indicate that access is blocked or filtered.")
        return False

    print(" Connectivity to PMC S3 endpoint looks OK.")
    return True


def download_pmc_pdf(pmcid: str, out_dir: Path, session: requests.Session, delay: float = 0.2) -> None:
    pmcid = pmcid.strip()
    if not pmcid:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pmcid}.pdf"

    # Try a small number of possible article versions (most are .1)
    max_versions_to_try = 3

    try:
        for version in range(1, max_versions_to_try + 1):
            prefix = f"{pmcid}.{version}"
            filename = f"{prefix}.pdf"
            url = S3_BASE_URL.format(prefix=prefix, filename=filename)

            print(f"Downloading {pmcid} (version {version}) from {url}...")
            resp = session.get(url, allow_redirects=True, timeout=60)

            if resp.status_code != 200:
                print(f" Version {version}: HTTP {resp.status_code}, skipping")
                time.sleep(delay)
                continue

            # S3 may return application/pdf or binary/octet-stream; also sanity-check content starts with %PDF
            content_type = resp.headers.get("Content-Type", "").lower()
            is_pdf_type = "application/pdf" in content_type or "octet-stream" in content_type
            is_pdf_magic = resp.content.startswith(b"%PDF")

            if not (is_pdf_type or is_pdf_magic):
                print(f" Version {version}: response not recognized as PDF, skipping")
                time.sleep(delay)
                continue

            with open(out_path, "wb") as f:
                f.write(resp.content)
            print(f" Saved to {out_path}")
            break
        else:
            print(f" FAILED: no PDF found in first {max_versions_to_try} versions for {pmcid}")

    except Exception as e:
        print(f" ERROR for {pmcid}: {e}")
    finally:
        time.sleep(delay)  # be polite to the server


def load_pmcids(path: Path):
    """Load PMCIDs from a CSV file with a 'pmcid' column (e.g. pmc_metadata.csv)."""
    pmcids: list[str] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "pmcid" not in reader.fieldnames:
            print("Input CSV does not contain a 'pmcid' column.", file=sys.stderr)
            return []

        for row in reader:
            pmcid = (row.get("pmcid") or "").strip()
            if not pmcid:
                continue
            # Ensure PMC prefix and normalize case
            if not pmcid.upper().startswith("PMC"):
                pmcid = "PMC" + pmcid
            pmcids.append(pmcid.upper())

    return pmcids


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m biomedical_rag.data_collection_pdf <pmc_metadata.csv> <output_folder>")
        sys.exit(1)

    list_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    with requests.Session() as session:
        # Check that we can reach the PMC S3 bucket at all before starting
        if not check_s3_connectivity(session):
            print("Aborting due to connectivity problem to pmc-oa-opendata.s3.amazonaws.com.")
            sys.exit(1)

        pmcids = load_pmcids(list_path)
        if not pmcids:
            print("No valid PMCIDs found in input file.")
            sys.exit(1)

        for pmcid in pmcids:
            download_pmc_pdf(pmcid, out_dir, session)


if __name__ == "__main__":
    main()