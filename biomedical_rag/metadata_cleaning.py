from . import config
import re
import pandas as pd
import unicodedata

# ---------- FILE PATHS ----------
input_file = str(config.path("metadata.csv"))
cleaned_file = str(config.path("metadata_cleaned.csv"))
comparison_file = str(config.path("metadata_cleaning_comparison.csv"))

# ---------- DATE COLUMNS ----------
date_cols = [
    "electronic_publication_date",
    "print_publication_date",
    "received_date",
    "accepted_date",
]

# ---------- CLEANER ----------
def clean_text(value):
    if pd.isna(value):
        return value

    text = str(value)

    # Remove BOM / invisible junk
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("ï»¿", "")

    # Try fixing common mojibake
    suspicious_patterns = ["Ã", "â€", "â€™", "â€œ", "â€\x9d", "â€“", "â€”", "ï»¿"]
    if any(p in text for p in suspicious_patterns):
        try:
            text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            pass

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Manual replacements for common broken chars
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â€": "-",
        "â€¦": "...",
        "Â ": " ",
        "Â": "",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def convert_date(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s == "":
        return ""

    # Exact DD-MM-YYYY -> YYYY-MM-DD
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", s)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"

    # Fallback: try pandas parsing with dayfirst (handles many variants)
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return s
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s


def main():
    # ---------- LOAD ----------
    df_before = pd.read_csv(input_file)
    df_after = df_before.copy()

    # Clean only text columns
    text_columns = df_after.select_dtypes(include=["object"]).columns.tolist()

    for col in text_columns:
        df_after[col] = df_after[col].apply(clean_text)

    # ---------- NORMALIZE DATE COLUMNS ----------
    existing = [c for c in date_cols if c in df_after.columns]
    print("Found date columns:", existing)
    if existing:
        for col in existing:
            print(f"Converting column: {col}")
            df_after[col] = df_after[col].apply(convert_date)
            print(df_after[col].head(10).to_string(index=False))
    else:
        print("No specified date columns found.")

    # Save cleaned full file
    df_after.to_csv(cleaned_file, index=False, encoding="utf-8-sig")

    # ---------- BUILD BEFORE/AFTER COMPARISON ----------
    comparison_rows = []

    for idx in range(len(df_before)):
        pmcid = df_before.at[idx, "pmcid"] if "pmcid" in df_before.columns else f"row_{idx}"

        for col in text_columns:
            before_val = df_before.at[idx, col]
            after_val = df_after.at[idx, col]

            before_str = "" if pd.isna(before_val) else str(before_val)
            after_str = "" if pd.isna(after_val) else str(after_val)

            if before_str != after_str:
                comparison_rows.append({
                    "row_index": idx,
                    "pmcid": pmcid,
                    "column_name": col,
                    "before": before_str,
                    "after": after_str
                })

    comparison_df = pd.DataFrame(comparison_rows)

    # Save comparison file
    comparison_df.to_csv(comparison_file, index=False, encoding="utf-8-sig")

    # ---------- SUMMARY ----------
    print(f"Cleaned file saved to: {cleaned_file}")
    print(f"Comparison file saved to: {comparison_file}")
    print(f"Total rows in original file: {len(df_before)}")
    print(f"Total changed cells: {len(comparison_df)}")

    if not comparison_df.empty:
        print("\nSample changes:")
        print(comparison_df.head(20).to_string(index=False))
    else:
        print("\nNo changes detected.")


if __name__ == "__main__":
    main()