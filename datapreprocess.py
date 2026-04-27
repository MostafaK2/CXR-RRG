import os
import re
import json
import numpy as np
import pandas as pd
from typing import Optional

# ----------------------------------------------------------------------------------- PREPROCESSING STEP 1  -----------------------------------------------------------------------

# ------------------------ Config ------------------------------
ROOT        = "/home/public/mkamal/dataset"
CSV_NAME    = "df_chexpert_plus_240401.csv"
LABELS_ROOT = "/home/public/mkamal/dataset/chesxbert_labels"
SAVE_DIR    = "/home/public/mkamal/dataset/filtered_data"

RANDOM_SEED  = 42

OUT_COLUMNS = [
    "report",
    "path_to_image",
    "path_to_png",
    "path_no_ext",
    "deid_patient_id",
    "patient_report_date_order",
    "split",
    "frontal_lateral",
    "ap_pa",
    "clinical_history",
    # Generation targets
    "section_impression_gen",    # lowercase — model trains on this
    "section_impression",
    "section_findings_gen",      # lowercase — encoder context (NaN if absent)
    "section_findings",
    "condensed_report",          # findings + impression, lowercase
    # Labelling targets
    "section_impression_label",  # cased   — RadGraph input
    "section_findings_label",    # cased   — RadGraph input
]


# -------------------------- Text Cleaning ----------------------------
# Compiled once — cheaper than recompiling per call
_RE_ANONYMISE   = re.compile(r'This report has been anonymized.*?(?:\n|$)', re.IGNORECASE)
_RE_ACCESSION   = re.compile(r'ACCESSION\s*NUMBER\s*:.*?(?:\n|$)', re.IGNORECASE)
_RE_SUMMARY     = re.compile(r'\d+-[A-Z][A-Z\s,/]+(?:REPORTED|ACTION|FINDING|REVIEWED)[^\n]*', re.IGNORECASE)
_RE_SIGNATURE   = re.compile(r'I have personally reviewed.*?(?:\n|$)', re.IGNORECASE)
_RE_TRANSCRIBED = re.compile(r'with the report transcribed above.*?(?:\n|$)', re.IGNORECASE)
_RE_EXAM_HEADER = re.compile(r'(?i)^\s*chest\s+\d+\s+view[s]?\s*[:\-,][^\n]*', re.MULTILINE)
_RE_COMPARISON  = re.compile(r'(?i)^[\s]*comparison\s*:[^\n]*', re.MULTILINE)
_RE_SEC_HEADERS = re.compile(
    r'(?i)^\s*(IMPRESSION|FINDINGS?|NARRATIVE|SUMMARY|HISTORY'
    r'|CLINICAL\s+HISTORY|COMPARISON|EXAM|INDICATION'
    r'|TECHNIQUE|PROCEDURE)\s*[:/]?\s*',
    re.MULTILINE
)
_RE_NUMBERED = re.compile(r'(?<!\d)\d+\.\s*(?=[A-Za-z])')
_RE_PAREN_NUM   = re.compile(r'\(\d+\)\s*')           # "(1) "
_RE_MERGE_SENT  = re.compile(r'\.(?=[A-Z])')          # "process.No" → "process. No"
_RE_WHITESPACE  = re.compile(r'[\n\r\t]+')
_RE_MULTI_SPACE = re.compile(r' {2,}')


def _remove_metadata(text: str) -> str:
    """Strip all non-clinical metadata lines."""
    text = _RE_ANONYMISE.sub('', text)
    text = _RE_ACCESSION.sub('', text)
    text = _RE_SUMMARY.sub('', text)
    text = _RE_SIGNATURE.sub('', text)
    text = _RE_TRANSCRIBED.sub('', text)
    text = _RE_EXAM_HEADER.sub('', text)
    text = _RE_COMPARISON.sub('', text)
    text = _RE_SEC_HEADERS.sub('', text)
    return text

def _normalise_whitespace(text: str) -> str:
    text = _RE_NUMBERED.sub('', text)       # remove "1. " markers
    text = _RE_PAREN_NUM.sub('', text)      # remove "(1) " markers
    text = _RE_WHITESPACE.sub(' ', text)    # collapse newlines
    text = _RE_MERGE_SENT.sub('. ', text)   # fix merged sentences
    text = _RE_MULTI_SPACE.sub(' ', text)   # collapse spaces
    return text.strip()

def clean_for_generation(text) -> Optional[str]:
    """
    Generation target and BLEU/ROUGE/CheXbert evaluation text.
    Lowercase. No list markers. No metadata.
 
    Use for:
      - Model training target
      - BLEU, ROUGE, METEOR evaluation
      - CheXbert CE-F1 (uses bert-base-UNCASED internally)
    """
    if not isinstance(text, str) or pd.isna(text):
        return None
    text = _remove_metadata(text)
    text = _normalise_whitespace(text)
    text = text.lower()
    return text if text else None

def extract_impression(text: str) -> Optional[str]:
    """
    Extract impression section from full report text.
    Handles merged Findings/Impression sections common in CheXpert Plus.
    """
    if not isinstance(text, str):
        return None
 
    # Standard IMPRESSION: ... section
    match = re.search(
        r'IMPRESSION\s*[:/]?\s*(.*?)\s*'
        r'(?:SUMMARY\s*:|END\s+OF\s+IMPRESSION|I\s+have\s+personally'
        r'|ACCESSION\s+NUMBER|This\s+report\s+has|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip() or None
 
    # Merged "Findings/Impression:" block — take everything after the header
    match = re.search(
        r'(?:FINDINGS?|OBSERVATIONS?)\s*/\s*IMPRESSION\s*[:/]?\s*(.*?)\s*'
        r'(?:SUMMARY\s*:|ACCESSION\s+NUMBER|This\s+report\s+has|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip() or None
 
    return None
  
def extract_findings(text: str) -> Optional[str]:
    """
    Extract findings section from full report text.
    Returns None if not present — never returns a placeholder.
    """
    if not isinstance(text, str):
        return None
 
    match = re.search(
        r'(?:FINDINGS?|OBSERVATIONS?)\s*[:/]?\s*(.*?)\s*'
        r'(?:IMPRESSION|SUMMARY\s*:|CONCLUSION|ACCESSION\s+NUMBER'
        r'|This\s+report\s+has|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        result = match.group(1).strip()
        # Guard against accidentally capturing impression content
        # (happens in merged Findings/Impression blocks)
        if result and len(result.split()) > 3:
            return result
    return None
 
# ------------------------------------ LOAD AND FILTER ---------------------------
def get_existing_image_paths(png_base_dir: str) -> set:
    """
    Walk PNG directory tree and return all paths without extension.
    Used to filter CSV rows to images that actually exist on disk.
    """
    existing = set()
    for split in ["train", "valid"]:
        split_dir = os.path.join(png_base_dir, split)
        if not os.path.exists(split_dir):
            print(f"  [WARN] Split directory not found: {split_dir}")
            continue
        for root, _, files in os.walk(split_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png")):
                    rel = os.path.relpath(os.path.join(root, f), png_base_dir)
                    existing.add(os.path.splitext(rel)[0])
    return existing

def load_and_filter(root: str, csv_name: str) -> tuple[pd.DataFrame, set]:
    """
    Load CSV, build path_no_ext, filter to existing images.
    Logs correct retained percentage (Fix #5).
    """
    csv_path = os.path.join(root, csv_name)
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    original_len = len(df)                          # store BEFORE filtering
    print(f"  Loaded: {original_len:,} rows")
 
    df["path_no_ext"] = df["path_to_image"].str.replace(
        r"\.(jpg|png|dcm)$", "", regex=True
    )
 
    # Filter to existing images
    existing = get_existing_image_paths(os.path.join(root, "PNG"))
    print(f"  Images on disk: {len(existing):,}")
 
    df = df[df["path_no_ext"].isin(existing)].copy()
    retained_pct = 100 * len(df) / original_len     # now correct (Fix #5)
    print(f"  After image filter: {len(df):,} rows ({retained_pct:.1f}% retained)")
 
    return df, existing
  
def filter_frontal_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep frontal views only (Fix #6).
    Lateral views share the same impression as their paired frontal —
    including them creates duplicate targets with different images.
    """
    before = len(df)
    df = df[df["frontal_lateral"].str.lower() == "frontal"].copy()
    print(f"  Frontal filter: {before:,} → {len(df):,} "
          f"(dropped {before - len(df):,} laterals)")
    return df

# -------------------------------------------  Build Sections ----------------------------
def build_sections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing impressions and findings from raw report.
    Clean both into generation and labelling variants.
    Never injects placeholders for missing findings (Fix #9).
    """
    print("\nBuilding sections...")
 
    # ── Impression ──────────────────────────────────────────────────────────
    missing_imp = df["section_impression"].isna()
    print(f"  Impression — missing before extraction: {missing_imp.sum():,}")
 
    df.loc[missing_imp, "section_impression"] = (
        df.loc[missing_imp, "report"].apply(extract_impression)
    )
    still_missing_imp = df["section_impression"].isna().sum()
    print(f"  Impression — filled: {missing_imp.sum() - still_missing_imp:,} "
          f"| still missing: {still_missing_imp:,}")
 
    # ── Findings ────────────────────────────────────────────────────────────
    if "section_findings" not in df.columns:
        df["section_findings"] = np.nan
 
    missing_find = df["section_findings"].isna()
    print(f"  Findings  — missing before extraction: {missing_find.sum():,}")
 
    df.loc[missing_find, "section_findings"] = (
        df.loc[missing_find, "report"].apply(extract_findings)
        # Returns None when not found — no placeholder injected
    )
    still_missing_find = df["section_findings"].isna().sum()
    print(f"  Findings  — filled: {missing_find.sum() - still_missing_find:,} "
          f"| still missing (impression-only studies): {still_missing_find:,}")
 
    # ── Clean both sections into two variants (Fix #3) ───────────────────────
    df["section_impression_gen"]   = df["section_impression"].apply(clean_for_generation)
    # df["section_impression_label"] = df["section_impression"].apply(clean_for_labelling)
    df["section_findings_gen"]     = df["section_findings"].apply(clean_for_generation)
    # df["section_findings_label"]   = df["section_findings"].apply(clean_for_labelling)
 
    # ── condensed_report: findings + impression, no placeholder (Fix #9) ────
    def _build_condensed(row):
        parts = []
        if pd.notna(row["section_findings_gen"]) and row["section_findings_gen"].strip():
            parts.append(row["section_findings_gen"].strip())
        if pd.notna(row["section_impression_gen"]) and row["section_impression_gen"].strip():
            parts.append(row["section_impression_gen"].strip())
        return " ".join(parts) if parts else None
 
    df["condensed_report"] = df.apply(_build_condensed, axis=1)
 
    # ── Drop rows with no usable impression ─────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["section_impression_gen"])
    print(f"\n  Dropped {before - len(df):,} rows with empty/unparseable impressions")
    print(f"  Usable rows: {len(df):,}")
 
    return df
 
# ─────────────────────────────────────────────────────────────────────────────
# 5.  Clinical history
# ─────────────────────────────────────────────────────────────────────────────
 
def build_clinical_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge section_clinical_history and section_history.
    Clean whitespace but keep as separate input — never mixed into
    the generation target.
    """
    has_clin  = "section_clinical_history" in df.columns
    has_hist  = "section_history" in df.columns
 
    if has_clin and has_hist:
        df["clinical_history"] = np.where(
            df["section_clinical_history"].notna(),
            df["section_clinical_history"],
            df["section_history"]
        )
    elif has_clin:
        df["clinical_history"] = df["section_clinical_history"]
    elif has_hist:
        df["clinical_history"] = df["section_history"]
    else:
        df["clinical_history"] = ""
 
    df["clinical_history"] = (
        df["clinical_history"]
        .fillna("")
        .apply(clean_for_generation)
        .fillna("")
    )
    return df


# 7.  Label alignment
# ─────────────────────────────────────────────────────────────────────────────
 
def load_and_align_labels(
    labels_root: str,
    filename: str,
    final_df: pd.DataFrame,
    existing: set,
) -> pd.DataFrame:
    """
    Load a CheXbert label file, filter to existing images,
    align to final_df order.
    Uses explicit key (path_to_image) — never assumes column[0] (Fix #4).
    Raises on misalignment rather than silently passing.
    """
    path = os.path.join(labels_root, filename)
    if not os.path.exists(path):
        print(f"  [WARN] Label file not found: {path}")
        return pd.DataFrame()
 
    ldf = pd.read_json(path, lines=True)
 
    # Explicit key — never df.columns[0]
    assert "path_to_image" in ldf.columns, \
        f"'path_to_image' column missing from {filename}. Columns: {ldf.columns.tolist()}"
 
    ldf["path_no_ext"] = ldf["path_to_image"].str.replace(
        r"\.(jpg|png|dcm)$", "", regex=True
    )
    ldf = ldf[ldf["path_no_ext"].isin(existing)]
 
    # Align to final_df order using explicit merge
    ldf = (
        final_df[["path_to_image"]]
        .merge(ldf, on="path_to_image", how="left")
    )
 
    # Hard assertion — values must match, not just index order (Fix #4)
    assert (ldf["path_to_image"].values == final_df["path_to_image"].values).all(), \
        f"Alignment failed for {filename} — path_to_image mismatch after merge"
 
    n_null = ldf.drop(columns=["path_to_image", "path_no_ext"]).isna().all(axis=1).sum()
    if n_null > 0:
        print(f"  [WARN] {filename}: {n_null:,} rows have all-null labels after alignment")
 
    print(f"  {filename}: {len(ldf):,} rows aligned ✓")
    return ldf
 
 
# Main Pipeline

def run_pipeline(
    root:        str = ROOT,
    csv_name:    str = CSV_NAME,
    labels_root: str = LABELS_ROOT,
    save_dir:    str = SAVE_DIR,
    seed:        int  = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    print("=" * 65)
    print("CheXpert Plus Preprocessing Pipeline")
    print("=" * 65)

    # loading and filtering 
    df, existing = load_and_filter(root, csv_name)

    # filtering for only frontal views
    print("\nFiltering to frontal views only...")
    df = filter_frontal_only(df)

    print("\nExtracting and cleaning sections...")
    df = build_sections(df)

    print("\nBuilding clinical history...")
    df = build_clinical_history(df)

    print("\n[6] Finalising output columns...")
    df["path_to_png"] = df["path_no_ext"] + ".png"
 
    # Keep only output columns that actually exist in df
    keep_cols = [c for c in OUT_COLUMNS if c in df.columns]
    missing_cols = [c for c in OUT_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"  [WARN] Columns not found in df (skipped): {missing_cols}")
    final = df[keep_cols].copy()

    return final, existing


results, existing = run_pipeline()

# ----------------------------------------------------------------------------------- PREPROCESSING STEP 2  -----------------------------------------------------------------------

def load_and_align_labels(
    root: str,
    filename: str,
    final_df: pd.DataFrame,
    existing: set,
) -> pd.DataFrame:
    path = os.path.join(root, filename)
    ldf  = pd.read_json(path, lines=True)

    # Bug 1 fix — explicit key, never assume column order
    assert "path_to_image" in ldf.columns, \
        f"'path_to_image' missing from {filename}. Columns: {ldf.columns.tolist()}"

    ldf["path_no_ext"] = ldf["path_to_image"].str.replace(
        r"\.(jpg|png|dcm)$", "", regex=True
    )
    ldf = ldf[ldf["path_no_ext"].isin(existing)]

    # Bug 2 fix — merge instead of reindex, NaN rows are visible
    ldf = final_df[["path_to_image"]].merge(ldf, on="path_to_image", how="left")

    # Bug 3 fix — assert values match, not just length
    assert (ldf["path_to_image"].values == final_df["path_to_image"].values).all(), \
        f"Alignment failed for {filename} — path_to_image mismatch after merge"

    n_null = ldf.drop(columns=["path_to_image", "path_no_ext"],
                      errors="ignore").isna().all(axis=1).sum()
    if n_null > 0:
        print(f"  [WARN] {filename}: {n_null:,} rows have all-null labels after alignment")

    print(f"  {filename}: {len(ldf):,} rows aligned ✓")
    return ldf

# ─────────────────────────── Main ─────────────────────────────
impression_df = load_and_align_labels(LABELS_ROOT, "impression_fixed.json", results, existing)
findings_df   = load_and_align_labels(LABELS_ROOT, "findings_fixed.json",   results, existing)
reports_lbl_df = load_and_align_labels(LABELS_ROOT, "report_fixed.json",   results, existing)
print(f"\nImpression aligned: {results['path_to_image'].tolist() == impression_df['path_to_image'].tolist()}")
print(f"Findings aligned:   {results['path_to_image'].tolist() == findings_df['path_to_image'].tolist()}")
print(f"reports lbl aligned:   {results['path_to_image'].tolist() == reports_lbl_df['path_to_image'].tolist()}")


# 7. Save
os.makedirs(SAVE_DIR, exist_ok=True)
results.to_csv(         os.path.join(SAVE_DIR, "reports_dataframe.csv"),  index=False)
impression_df.to_csv( os.path.join(SAVE_DIR, "impression_labels.csv"),  index=False)
findings_df.to_csv(   os.path.join(SAVE_DIR, "finding_labels.csv"),     index=False)
reports_lbl_df.to_csv(   os.path.join(SAVE_DIR, "reports_label.csv"),     index=False)
print("Saved all files.")

# ----------------------------------------------------------------------------------- PREPROCESSING STEP 3  -----------------------------------------------------------------------

# splitting final df into train and validation
path_train = "/home/public/mkamal/dataset/filtered_data/train" 
path_validation = "/home/public/mkamal/dataset/filtered_data/valid"

os.makedirs(path_train, exist_ok=True)
os.makedirs(path_validation, exist_ok=True)

final_data_path = "/home/public/mkamal/dataset/filtered_data/reports_dataframe.csv"
final = pd.read_csv(final_data_path)
print(final.isna().sum())

final_train = final[final["split"] == "train"].reset_index(drop=True)
final_val = final[final["split"] == "valid"].reset_index(drop=True)

print(len(final), len(final_train), len(final_val))
print(len(final_train) / len(final) * 100)
print(len(final_val) / len(final) * 100)

# save inside folders
train_csv_path = os.path.join(path_train, "train.csv")
val_csv_path = os.path.join(path_validation, "valid.csv")
final_train.to_csv(train_csv_path, index=False)
final_val.to_csv(val_csv_path, index=False)


from pathlib import Path
from PIL import Image
from torchvision import transforms
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def csv_to_h5(
    csv_path,
    base_dir,
    save_path,
    transform):
    # Load CSV

    df = pd.read_csv(csv_path)
    save_path = Path(save_path)
    
    failed = []

    with h5py.File(save_path, "w") as f:
        N = len(df)

        # image dataset
        images = f.create_dataset(
            "images",
            shape=(N, 1, 224, 224),
            dtype="float32"
        )

        # store paths (VERY useful for debugging)
        f.create_dataset(
            "paths",
            data=df["path_to_png"].astype("S")
        )

        # build dataset
        for i, path in enumerate(tqdm(df["path_to_png"])):
            try:
                img = Image.open(Path(base_dir) / path).convert("L")
                images[i] = transform(img).numpy()

            except Exception as e:
                print(f"Error at index {i}, {path}: {e}")
                failed.append({
                    "index": i,
                    "path": path,
                    "error": str(e)
                })

    print(f"Done. Failed: {len(failed)} images")
    return failed


# Transform -- IMAGENET Resize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


failed_train = csv_to_h5(
    csv_path="/home/public/mkamal/dataset/filtered_data/train/train_cleaned.csv",
    base_dir="/home/public/mkamal/dataset/PNG",
    save_path="/home/public/mkamal/dataset/filtered_data/train/train.h5",
    transform = transform
)

failed_train = csv_to_h5(
    csv_path="/home/public/mkamal/dataset/filtered_data/valid/valid_cleaned.csv",
    base_dir="/home/public/mkamal/dataset/PNG",
    save_path="/home/public/mkamal/dataset/filtered_data/valid/valid.h5",
    transform = transform
)



# ----------------------------------------------------------------------------------- PRERPROCESSING STEP 4 Further cleaning  -----------------------------------------------------------------------

