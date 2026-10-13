from . import config
import os
import sys
import time
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import requests

# Base bucket URL
S3_BUCKET_URL = "https://pmc-oa-opendata.s3.amazonaws.com"


def check_s3_connectivity(session: requests.Session, timeout: float = 5.0) -> bool:
    """Quickly check whether the PMC S3 endpoint is reachable."""
    test_url = f"{S3_BUCKET_URL}/README.txt"
    print(f"Checking connectivity to {test_url} ...")
    try:
        resp = session.get(test_url, timeout=timeout)
    except requests.RequestException as exc:
        print("ERROR: Could not reach PMC S3 endpoint.")
        print(f"Details: {exc}")
        print("This is likely a firewall or proxy issue on this network.")
        return False

    if resp.status_code != 200:
        print(f"WARNING: Got HTTP {resp.status_code} from PMC S3 endpoint.")
        print("This may indicate that access is blocked or filtered.")
        return False

    print("Connectivity to PMC S3 endpoint looks OK.")
    return True


def load_pmcids(path: Path):
    """Load PMCIDs from a CSV file with a 'pmcid' column."""
    pmcids = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "pmcid" not in reader.fieldnames:
            print("Input CSV does not contain a 'pmcid' column.", file=sys.stderr)
            return []

        for row in reader:
            pmcid = (row.get("pmcid") or "").strip()
            if not pmcid:
                continue
            if not pmcid.upper().startswith("PMC"):
                pmcid = "PMC" + pmcid
            pmcids.append(pmcid.upper())

    return pmcids


def list_s3_objects(session: requests.Session, prefix: str):
    """
    List all objects under a given prefix using the public S3 bucket listing API.
    Example prefix: PMC6824378.1/
    """
    params = {
        "list-type": "2",
        "prefix": prefix,
    }

    url = S3_BUCKET_URL
    resp = session.get(url, params=params, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)

    # S3 XML namespace
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    keys = []
    for contents in root.findall("s3:Contents", ns):
        key_elem = contents.find("s3:Key", ns)
        if key_elem is not None and key_elem.text:
            keys.append(key_elem.text)

    return keys


def download_file(session: requests.Session, key: str, output_root: Path):
    """
    Download one object from the bucket and preserve its folder structure locally.
    """
    url = f"{S3_BUCKET_URL}/{quote(key, safe='/')}"
    local_path = output_root / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading: {key}")
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"  Saved to: {local_path}")


def download_pmc_folder(pmcid: str, out_dir: Path, session: requests.Session, delay: float = 0.2) -> None:
    """
    Download the full folder/prefix for a PMCID by trying a few possible versions.
    Example S3 prefix:
      PMC6824378.1/
    """
    pmcid = pmcid.strip().upper()
    if not pmcid:
        return

    max_versions_to_try = 3
    found_any = False

    try:
        for version in range(1, max_versions_to_try + 1):
            prefix = f"{pmcid}.{version}/"
            print(f"\nChecking folder for {pmcid} version {version} ...")

            try:
                keys = list_s3_objects(session, prefix)
            except Exception as e:
                print(f"  Version {version}: listing failed: {e}")
                time.sleep(delay)
                continue

            if not keys:
                print(f"  Version {version}: no files found")
                time.sleep(delay)
                continue

            print(f"  Version {version}: found {len(keys)} files")
            for key in keys:
                download_file(session, key, out_dir)
                time.sleep(delay)

            found_any = True
            break

        if not found_any:
            print(f"FAILED: no folder found in first {max_versions_to_try} versions for {pmcid}")

    except Exception as e:
        print(f"ERROR for {pmcid}: {e}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m biomedical_rag.data_collection_dir <pmc_metadata.csv> <output_folder>")
        sys.exit(1)

    list_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    with requests.Session() as session:
        if not check_s3_connectivity(session):
            print("Aborting due to connectivity problem to pmc-oa-opendata.s3.amazonaws.com.")
            sys.exit(1)

        pmcids = load_pmcids(list_path)
        if not pmcids:
            print("No valid PMCIDs found in input file.")
            sys.exit(1)

        print(f"Found {len(pmcids)} PMCIDs")

        for pmcid in pmcids:
            download_pmc_folder(pmcid, out_dir, session)


if __name__ == "__main__":
    main()