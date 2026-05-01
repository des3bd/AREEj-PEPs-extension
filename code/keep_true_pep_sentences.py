import pandas as pd
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("data/review/pep_review_dataset_gemini_labeled.csv")
OUTPUT_CSV = Path("data/review/pep_review_dataset_true_only.csv")


def normalize_bool_value(value):
    """
    Handles values like:
    True, true, TRUE, 1, yes
    """
    return str(value).strip().lower() in {"true", "1", "yes"}


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    if "pep_sentence" not in df.columns:
        raise ValueError("Column 'pep_sentence' was not found in the CSV.")

    before_count = len(df)

    filtered = df[df["pep_sentence"].apply(normalize_bool_value)].copy()

    # Remove the helper classification column if you do not want it in final dataset
    filtered = filtered.drop(columns=["pep_sentence"])

    # Rebuild IDs
    filtered["id"] = range(1, len(filtered) + 1)

    # Optional: keep final column order
    final_columns = [
        "id",
        "target_name_ar",
        "sentence",
        "subject",
        "relation",
        "object",
        "evidence",
        "split",
    ]

    existing_columns = [col for col in final_columns if col in filtered.columns]
    filtered = filtered[existing_columns]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Input rows: {before_count}")
    print(f"Kept true rows: {len(filtered)}")
    print(f"Dropped rows: {before_count - len(filtered)}")
    print(f"Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()