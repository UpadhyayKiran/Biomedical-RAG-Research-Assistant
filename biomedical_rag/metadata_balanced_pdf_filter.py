from . import config
import pandas as pd

INPUT_FILE = str(config.path('pmc_dirs/metadata.csv'))
OUTPUT_FILE = str(config.path('metadata.csv'))
PDFS_PER_YEAR = 19

def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False
        })
    )

# Load file
df = pd.read_csv(INPUT_FILE)

# Normalize columns
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["article_type_norm"] = df["article_type"].astype(str).str.strip().str.lower()

bool_cols = [
    "pdf_present",
    "is_retracted",
    "is_historical_ocr",
    "is_pmc_openaccess",
    "xml_present",
    "txt_present",
    "json_present"
]

for col in bool_cols:
    if col in df.columns:
        df[col + "_bool"] = to_bool(df[col])

# Base filtering
filtered = df[
    (df["article_type_norm"] == "research-article") &
    (df["pdf_present_bool"] == True) &
    (df["is_retracted_bool"] != True) &
    (df["is_historical_ocr_bool"] != True)
].copy()

# Create a quality/completeness score
filtered["completeness_score"] = 0

text_cols = [
    "title",
    "abstract",
    "journal_title",
    "publisher",
    "authors",
    "doi",
    "pmid",
    "keywords",
    "received_date",
    "accepted_date"
]

for col in text_cols:
    if col in filtered.columns:
        filtered["completeness_score"] += (
            filtered[col].notna() &
            (filtered[col].astype(str).str.strip() != "") &
            (filtered[col].astype(str).str.lower() != "nan")
        ).astype(int)

extra_quality_cols = [
    "is_pmc_openaccess_bool",
    "xml_present_bool",
    "txt_present_bool",
    "json_present_bool"
]

for col in extra_quality_cols:
    if col in filtered.columns:
        filtered["completeness_score"] += filtered[col].fillna(False).astype(int)

# Select top 19 from each year
selected_parts = []
years = sorted(filtered["year"].dropna().unique())

for year in years:
    year_df = filtered[filtered["year"] == year].copy()

    if len(year_df) < PDFS_PER_YEAR:
        print(f"Warning: Year {year} has only {len(year_df)} matching records.")

    year_df = year_df.sort_values(
        by=[
            "completeness_score",
            "is_pmc_openaccess_bool",
            "xml_present_bool",
            "txt_present_bool",
            "json_present_bool",
            "pmcid"
        ],
        ascending=[False, False, False, False, False, True]
    )

    selected_parts.append(year_df.head(PDFS_PER_YEAR))

final_df = pd.concat(selected_parts, ignore_index=True)

# Keep original columns only if you want a clean output
helper_cols = [col for col in final_df.columns if col.endswith("_bool") or col in ["article_type_norm", "completeness_score"]]
clean_output = final_df.drop(columns=helper_cols, errors="ignore")

# Save
clean_output.to_csv(OUTPUT_FILE, index=False)

# Summary
print("\nSaved file:", OUTPUT_FILE)
print("Total selected records:", len(clean_output))
print("\nRecords per year:")
print(clean_output["year"].value_counts().sort_index())
print("\nArticle type check:")
print(clean_output["article_type"].value_counts(dropna=False))