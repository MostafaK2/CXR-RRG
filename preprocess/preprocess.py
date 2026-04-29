import os
import re
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from PIL import Image
from torchvision import transforms
import h5py
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

with open("pipeline_config.yml") as f:
    cfg = yaml.safe_load(f)

ROOT        = cfg["paths"]["root"]
CSV_NAME    = cfg["paths"]["csv_name"]
LABELS_ROOT = cfg["paths"]["labels_root"]
SAVE_DIR    = cfg["paths"]["save_dir"]
PNG_BASE    = cfg["paths"]["png_base_dir"]
RANDOM_SEED = cfg["random_seed"]
OUT_COLUMNS = cfg["output_columns"]
IMG_SIZE    = cfg["image"]["size"]
LABEL_FILES = cfg["label_files"]

FRONTAL_ONLY       = cfg["filtering"]["frontal_only"]
REQUIRE_IMPRESSION = cfg["filtering"]["require_impression"]

PATH_TRAIN = os.path.join(SAVE_DIR, "train")
PATH_VALID = os.path.join(SAVE_DIR, "valid")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Text cleaning (section extraction)
# ─────────────────────────────────────────────────────────────────────────────

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
_RE_NUMBERED    = re.compile(r'(?<!\d)\d+\.\s*(?=[A-Za-z])')
_RE_PAREN_NUM   = re.compile(r'\(\d+\)\s*')
_RE_MERGE_SENT  = re.compile(r'\.(?=[A-Z])')
_RE_WHITESPACE  = re.compile(r'[\n\r\t]+')
_RE_MULTI_SPACE = re.compile(r' {2,}')


def _remove_metadata(text: str) -> str:
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
    text = _RE_NUMBERED.sub('', text)
    text = _RE_PAREN_NUM.sub('', text)
    text = _RE_WHITESPACE.sub(' ', text)
    text = _RE_MERGE_SENT.sub('. ', text)
    text = _RE_MULTI_SPACE.sub(' ', text)
    return text.strip()


def clean_for_generation(text) -> Optional[str]:
    """Lowercase, no list markers, no metadata. Used for training targets and BLEU/ROUGE/CheXbert eval."""
    if not isinstance(text, str) or pd.isna(text):
        return None
    text = _remove_metadata(text)
    text = _normalise_whitespace(text)
    text = text.lower()
    return text if text else None


def extract_impression(text: str) -> Optional[str]:
    """Extract impression section. Handles merged Findings/Impression blocks."""
    if not isinstance(text, str):
        return None
    match = re.search(
        r'IMPRESSION\s*[:/]?\s*(.*?)\s*'
        r'(?:SUMMARY\s*:|END\s+OF\s+IMPRESSION|I\s+have\s+personally'
        r'|ACCESSION\s+NUMBER|This\s+report\s+has|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip() or None
    match = re.search(
        r'(?:FINDINGS?|OBSERVATIONS?)\s*/\s*IMPRESSION\s*[:/]?\s*(.*?)\s*'
        r'(?:SUMMARY\s*:|ACCESSION\s+NUMBER|This\s+report\s+has|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip() or None
    return None


def extract_findings(text: str) -> Optional[str]:
    """Extract findings section. Returns None if absent — never injects a placeholder."""
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
        if result and len(result.split()) > 3:
            return result
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load & filter
# ─────────────────────────────────────────────────────────────────────────────

def get_existing_image_paths(png_base_dir: str) -> set:
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
    csv_path = os.path.join(root, csv_name)
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    original_len = len(df)
    print(f"  Loaded: {original_len:,} rows")

    df["path_no_ext"] = df["path_to_image"].str.replace(r"\.(jpg|png|dcm)$", "", regex=True)

    existing = get_existing_image_paths(PNG_BASE)
    print(f"  Images on disk: {len(existing):,}")

    df = df[df["path_no_ext"].isin(existing)].copy()
    print(f"  After image filter: {len(df):,} rows ({100 * len(df) / original_len:.1f}% retained)")
    return df, existing


def filter_frontal_only(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["frontal_lateral"].str.lower() == "frontal"].copy()
    print(f"  Frontal filter: {before:,} → {len(df):,} (dropped {before - len(df):,} laterals)")
    return df


def build_sections(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding sections...")

    missing_imp = df["section_impression"].isna()
    print(f"  Impression — missing before extraction: {missing_imp.sum():,}")
    df.loc[missing_imp, "section_impression"] = df.loc[missing_imp, "report"].apply(extract_impression)
    still_missing_imp = df["section_impression"].isna().sum()
    print(f"  Impression — filled: {missing_imp.sum() - still_missing_imp:,} | still missing: {still_missing_imp:,}")

    if "section_findings" not in df.columns:
        df["section_findings"] = np.nan
    missing_find = df["section_findings"].isna()
    print(f"  Findings  — missing before extraction: {missing_find.sum():,}")
    df.loc[missing_find, "section_findings"] = df.loc[missing_find, "report"].apply(extract_findings)
    still_missing_find = df["section_findings"].isna().sum()
    print(f"  Findings  — filled: {missing_find.sum() - still_missing_find:,} | still missing: {still_missing_find:,}")

    df["section_impression_gen"] = df["section_impression"].apply(clean_for_generation)
    df["section_findings_gen"]   = df["section_findings"].apply(clean_for_generation)

    def _build_condensed(row):
        parts = []
        if pd.notna(row["section_findings_gen"]) and row["section_findings_gen"].strip():
            parts.append(row["section_findings_gen"].strip())
        if pd.notna(row["section_impression_gen"]) and row["section_impression_gen"].strip():
            parts.append(row["section_impression_gen"].strip())
        return " ".join(parts) if parts else None

    df["condensed_report"] = df.apply(_build_condensed, axis=1)

    if REQUIRE_IMPRESSION:
        before = len(df)
        df = df.dropna(subset=["section_impression_gen"])
        print(f"\n  Dropped {before - len(df):,} rows with empty/unparseable impressions")

    print(f"  Usable rows: {len(df):,}")
    return df


def build_clinical_history(df: pd.DataFrame) -> pd.DataFrame:
    has_clin = "section_clinical_history" in df.columns
    has_hist = "section_history" in df.columns

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
        df["clinical_history"].fillna("").apply(clean_for_generation).fillna("")
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Label alignment
# ─────────────────────────────────────────────────────────────────────────────

def load_and_align_labels(
    labels_root: str,
    filename: str,
    final_df: pd.DataFrame,
    existing: set,
) -> pd.DataFrame:
    path = os.path.join(labels_root, filename)
    if not os.path.exists(path):
        print(f"  [WARN] Label file not found: {path}")
        return pd.DataFrame()

    ldf = pd.read_json(path, lines=True)

    assert "path_to_image" in ldf.columns, \
        f"'path_to_image' missing from {filename}. Columns: {ldf.columns.tolist()}"

    ldf["path_no_ext"] = ldf["path_to_image"].str.replace(r"\.(jpg|png|dcm)$", "", regex=True)
    ldf = ldf[ldf["path_no_ext"].isin(existing)]
    ldf = final_df[["path_to_image"]].merge(ldf, on="path_to_image", how="left")

    assert (ldf["path_to_image"].values == final_df["path_to_image"].values).all(), \
        f"Alignment failed for {filename} — path_to_image mismatch after merge"

    n_null = ldf.drop(columns=["path_to_image", "path_no_ext"], errors="ignore").isna().all(axis=1).sum()
    if n_null > 0:
        print(f"  [WARN] {filename}: {n_null:,} rows have all-null labels after alignment")

    print(f"  {filename}: {len(ldf):,} rows aligned ✓")
    return ldf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Noise removal
# ─────────────────────────────────────────────────────────────────────────────

_FIX_INSTITUTION  = re.compile(
    r'\b(?:usc\s+center\s+for\s+body\s+computing|kollabio|kratikal)'
    r'(?:\s+\d{1,2}:\d{2}\s*(?:hours?|hrs?)?|\s+\d{3,4}\s*(?:hours?|hrs?)?)?'
    r'\s*:?',
    re.IGNORECASE
)
_FIX_ON_DATETIME  = re.compile(
    r'\bon\s+(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}'
    r'|(?:january|february|march|april|may|june|july|august'
    r'|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{2,4})?'
    r'|\d{1,2}:\d{2}(?:\s*[ap]\.\s*m\.|\s*[ap]\.m\.|\s*[ap]\s*m)?)\b\s*:?',
    re.IGNORECASE
)
_FIX_CONJ_TIME        = re.compile(r'\b(?:and|or)\s+\d{1,2}:\d{2}\s*', re.IGNORECASE)
_FIX_AM_PM_ORPHAN     = re.compile(r'\b[ap]\.\s*m\.', re.IGNORECASE)
_FIX_DATE_AT_TIME     = re.compile(
    r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\s+at\s+'
    r'(?:\d{3,4}\s+(?:hours?|hrs?)|\d{1,2}:\d{2}(?:\s*[ap]\.?\s*m\.?)?)',
    re.IGNORECASE
)
_FIX_COMPARISON_DATE  = re.compile(r'\bcomparison\s+to\s+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b\s*[,\.]?', re.IGNORECASE)
_FIX_DATE_INLINE      = re.compile(
    r'\b(?:since|from|dated?|as\s+of|compared\s+to\s+(?:the\s+)?prior)\s+'
    r'(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}'
    r'|(?:january|february|march|april|may|june|july|august'
    r'|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{2,4})?)',
    re.IGNORECASE
)
_FIX_DATE_NUMERIC     = re.compile(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b\s*:?')
_FIX_DATE_WRITTEN     = re.compile(
    r'\b(?:january|february|march|april|may|june|july|august'
    r'|september|october|november|december)'
    r'\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{2,4})?\b\s*:?',
    re.IGNORECASE
)
_FIX_DATE_YEAR        = re.compile(
    r'(?<![x\d])(?<!\d\.)\b(?:19|20)\d{2}\b'
    r'(?!\s*(?:cm|mm|ml|mg|kg|lbs?|units?|cc))'
)
_FIX_TIME             = re.compile(
    r'\b(?:at\s+)?(?:\d{3,4}\s+(?:hours?|hrs?)|\d{1,2}:\d{2}'
    r'(?:\s*[ap]\.\s*m\.|\s*[ap]\.m\.|\s*[ap]\s*m)?\s*:?)',
    re.IGNORECASE
)
_FIX_COMPARISON_EMPTY = re.compile(r'\bcomparison\s+to\s*[,\.]?\s*(?=[,\.\s]|$)', re.IGNORECASE)
_FIX_NUM_START        = re.compile(r'^\s*(?<!\d)\d+\.\s*(?=[A-Za-z])')
_FIX_NUM_NEWLINE      = re.compile(r'\n\s*(?<!\d)\d+\.\s*(?=[A-Za-z])')
_FIX_NUM_INLINE       = re.compile(r'(?<=[\s])(?<!\d)\d+\.\s*(?=[A-Za-z])')
_FIX_PAREN_NUM        = re.compile(r'\(\d+\)\s*')
_FIX_WHITESPACE       = re.compile(r'[\n\r\t]+')
_FIX_MERGE_SENT       = re.compile(r'\.(?=[A-Z])')
_FIX_MULTI_SPACE      = re.compile(r' {2,}')
_FIX_DANGLING         = re.compile(r'\s*[,\.]\s*\.')
_FIX_LEAD_PUNCT       = re.compile(r'^\s*[,:\.]\s*')
_FIX_TRAIL_COMMA      = re.compile(r',\s*$')
_FIX_MAY_NEED_ACTION  = re.compile(
    r'^[,\s]*may\s+need\s+action.*$|[,\s]+may\s+need\s+action.*$',
    re.IGNORECASE | re.DOTALL
)


def fix_noise(text: str) -> Optional[str]:
    """Remove residual metadata noise from already-cleaned generation text."""
    if not isinstance(text, str) or pd.isna(text):
        return None

    text = _FIX_INSTITUTION.sub('', text)
    text = _FIX_ON_DATETIME.sub('', text)
    text = _FIX_CONJ_TIME.sub('', text)
    text = _FIX_AM_PM_ORPHAN.sub('', text)
    text = _FIX_DATE_AT_TIME.sub('', text)
    text = _FIX_COMPARISON_DATE.sub('', text)
    text = _FIX_DATE_INLINE.sub('', text)
    text = _FIX_DATE_NUMERIC.sub('', text)
    text = _FIX_DATE_WRITTEN.sub('', text)
    text = _FIX_DATE_YEAR.sub('', text)
    text = _FIX_TIME.sub('', text)
    text = _FIX_AM_PM_ORPHAN.sub('', text)       # second pass for orphans
    text = _FIX_COMPARISON_EMPTY.sub('', text)
    text = _FIX_NUM_START.sub('', text)
    text = _FIX_NUM_NEWLINE.sub(' ', text)
    text = _FIX_PAREN_NUM.sub('', text)
    text = _FIX_WHITESPACE.sub(' ', text)
    text = _FIX_NUM_INLINE.sub('', text)
    text = _FIX_MERGE_SENT.sub('. ', text)
    text = _FIX_MULTI_SPACE.sub(' ', text)
    text = _FIX_DANGLING.sub('.', text)
    text = _FIX_LEAD_PUNCT.sub('', text)
    text = _FIX_TRAIL_COMMA.sub('', text)
    text = _FIX_MULTI_SPACE.sub(' ', text)
    text = _FIX_MAY_NEED_ACTION.sub('', text)

    text = text.strip()
    return text if text else None


def apply_noise_fixes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    print("\nApplying noise fixes...")
    for col in cols:
        if col not in df.columns:
            print(f"  [WARN] '{col}' not found, skipping.")
            continue
        before_null = df[col].isna().sum()
        df[col] = df[col].apply(fix_noise).replace('', pd.NA)
        newly_empty = df[col].isna().sum() - before_null
        print(f"  {col}: done. Rows newly empty after fix: {newly_empty:,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train/valid split + H5 export
# ─────────────────────────────────────────────────────────────────────────────

def split_and_save(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(PATH_TRAIN, exist_ok=True)
    os.makedirs(PATH_VALID, exist_ok=True)

    final_train = results[results["split"] == "train"].reset_index(drop=True)
    final_val   = results[results["split"] == "valid"].reset_index(drop=True)

    print(f"\nSplit: {len(results):,} total → "
          f"{len(final_train):,} train ({100 * len(final_train) / len(results):.1f}%) | "
          f"{len(final_val):,} valid ({100 * len(final_val) / len(results):.1f}%)")

    final_train.to_csv(os.path.join(PATH_TRAIN, "train_cleaned.csv"), index=False)
    final_val.to_csv(  os.path.join(PATH_VALID, "valid_cleaned.csv"), index=False)
    print(f"  Saved CSVs → {PATH_TRAIN} / {PATH_VALID}")
    return final_train, final_val


def csv_to_h5(csv_path: str, base_dir: str, save_path: str, transform) -> list:
    df = pd.read_csv(csv_path)
    save_path = Path(save_path)
    failed = []

    with h5py.File(save_path, "w") as f:
        N = len(df)
        images = f.create_dataset("images", shape=(N, 1, IMG_SIZE, IMG_SIZE), dtype="float32")
        f.create_dataset("paths", data=df["path_to_png"].astype("S"))

        for i, path in enumerate(tqdm(df["path_to_png"], desc=str(save_path.name))):
            try:
                img = Image.open(Path(base_dir) / path).convert("L")
                images[i] = transform(img).numpy()
            except Exception as e:
                print(f"  Error at index {i}, {path}: {e}")
                failed.append({"index": i, "path": path, "error": str(e)})

    print(f"  Done. Failed: {len(failed)} images")
    return failed


def build_h5_datasets():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    for split, folder in [("train", PATH_TRAIN), ("valid", PATH_VALID)]:
        csv_path = os.path.join(folder, f"{split}_cleaned.csv")
        h5_path  = os.path.join(folder, f"{split}.h5")
        print(f"\nBuilding H5: {h5_path}")
        csv_to_h5(csv_path, PNG_BASE, h5_path, transform)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 65)
    print("CheXpert Plus Preprocessing Pipeline")
    print("=" * 65)

    # Step 1 — load, filter, extract sections
    df, existing = load_and_filter(ROOT, CSV_NAME)

    if FRONTAL_ONLY:
        print("\nFiltering to frontal views only...")
        df = filter_frontal_only(df)

    print("\nExtracting and cleaning sections...")
    df = build_sections(df)

    print("\nBuilding clinical history...")
    df = build_clinical_history(df)

    print("\nFinalising output columns...")
    df["path_to_png"] = df["path_no_ext"] + ".png"
    keep_cols    = [c for c in OUT_COLUMNS if c in df.columns]
    missing_cols = [c for c in OUT_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"  [WARN] Columns not found in df (skipped): {missing_cols}")
    results = df[keep_cols].copy()

    # Step 2 — align labels
    print("\nAligning labels...")
    impression_df  = load_and_align_labels(LABELS_ROOT, LABEL_FILES["impression"], results, existing)
    findings_df    = load_and_align_labels(LABELS_ROOT, LABEL_FILES["findings"],   results, existing)
    reports_lbl_df = load_and_align_labels(LABELS_ROOT, LABEL_FILES["report"],     results, existing)

    print(f"\nImpression aligned: {results['path_to_image'].tolist() == impression_df['path_to_image'].tolist()}")
    print(f"Findings aligned:   {results['path_to_image'].tolist() == findings_df['path_to_image'].tolist()}")
    print(f"Report lbl aligned: {results['path_to_image'].tolist() == reports_lbl_df['path_to_image'].tolist()}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    results.to_csv(        os.path.join(SAVE_DIR, "reports_dataframe.csv"), index=False)
    impression_df.to_csv(  os.path.join(SAVE_DIR, "impression_labels.csv"), index=False)
    findings_df.to_csv(    os.path.join(SAVE_DIR, "finding_labels.csv"),    index=False)
    reports_lbl_df.to_csv( os.path.join(SAVE_DIR, "reports_label.csv"),     index=False)
    print(f"\nSaved label CSVs → {SAVE_DIR}")

    # Step 3 — noise removal on generation columns
    TEXT_COLS = ["section_impression_gen", "section_findings_gen"]
    results = apply_noise_fixes(results, TEXT_COLS)

    # Step 4 — train/valid split, save cleaned CSVs, build H5
    split_and_save(results)
    build_h5_datasets()

    print("\n" + "=" * 65)
    print("Pipeline complete.")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()