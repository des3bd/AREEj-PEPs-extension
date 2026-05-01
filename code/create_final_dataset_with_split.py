import random
import pandas as pd
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("data/review/pep_review_dataset_with_evidence.csv")
OUTPUT_CSV = Path("data/final/pep_areej_final_dataset.csv")

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =========================
# HELPERS
# =========================

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_target_output(row) -> str:
    subject = clean_text(row["subject"])
    obj = clean_text(row["object"])
    evidence = clean_text(row["evidence"])

    return f"<bor> {subject} <per> {obj} <concept> position held <rt> {evidence} <e>"


# =========================
# MAIN
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {"id", "sentence", "subject", "relation", "object", "evidence"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean text columns
    for col in ["sentence", "subject", "relation", "object", "evidence"]:
        df[col] = df[col].apply(clean_text)

    # Keep only complete rows
    before_complete = len(df)
    df = df[
        (df["sentence"] != "")
        & (df["subject"] != "")
        & (df["relation"] != "")
        & (df["object"] != "")
        & (df["evidence"] != "")
    ].copy()

    dropped_incomplete = before_complete - len(df)

    # Force final relation name
    df["relation"] = "position_held"

    # Remove exact duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["sentence", "subject", "relation", "object", "evidence"]
    ).copy()
    dropped_duplicates = before_dedup - len(df)

    # Split by subject/person
    subjects = sorted(df["subject"].unique().tolist())

    random.seed(RANDOM_SEED)
    random.shuffle(subjects)

    n_subjects = len(subjects)
    train_end = int(n_subjects * TRAIN_RATIO)
    val_end = train_end + int(n_subjects * VAL_RATIO)

    train_subjects = set(subjects[:train_end])
    val_subjects = set(subjects[train_end:val_end])
    test_subjects = set(subjects[val_end:])

    def assign_split(subject):
        if subject in train_subjects:
            return "train"
        if subject in val_subjects:
            return "validation"
        return "test"

    df["split"] = df["subject"].apply(assign_split)

    # Build AREEj-style target output
    df["target_output"] = df.apply(build_target_output, axis=1)

    # Rebuild IDs
    df = df.reset_index(drop=True)
    df["id"] = range(1, len(df) + 1)

    final_columns = [
        "id",
        "sentence",
        "subject",
        "relation",
        "object",
        "evidence",
        "target_output",
        "split",
    ]

    df = df[final_columns]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Input rows: {before_complete}")
    print(f"Dropped incomplete rows: {dropped_incomplete}")
    print(f"Dropped duplicate rows: {dropped_duplicates}")
    print(f"Final rows: {len(df)}")
    print(f"Unique subjects: {n_subjects}")
    print(f"Output saved to: {OUTPUT_CSV}")

    print("\nSplit counts:")
    print(df["split"].value_counts())

    print("\nSubject counts by split:")
    print(df.groupby("split")["subject"].nunique())


if __name__ == "__main__":
    main()