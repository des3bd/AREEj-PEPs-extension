import re
import pandas as pd
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("data/review/pep_review_dataset_relations_extracted.csv")

# Optional output files for manual inspection
PROBLEM_ROWS_CSV = Path("data/review/pep_dataset_problem_rows.csv")
SUMMARY_TXT = Path("data/review/pep_dataset_exploration_summary.txt")


# =========================
# HELPERS
# =========================

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_arabic(text: str) -> str:
    text = clean_text(text)

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # remove Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # remove tatweel
    text = text.replace("ـ", "")

    return clean_text(text)


def contains_normalized(sentence: str, phrase: str) -> bool:
    sentence_norm = normalize_arabic(sentence)
    phrase_norm = normalize_arabic(phrase)

    if not sentence_norm or not phrase_norm:
        return False

    return phrase_norm in sentence_norm


def word_count(text: str) -> int:
    text = clean_text(text)
    if not text:
        return 0
    return len(text.split())


def char_count(text: str) -> int:
    return len(clean_text(text))


def preview_rows(df: pd.DataFrame, n: int = 5) -> str:
    if df.empty:
        return "None\n"

    lines = []
    for _, row in df.head(n).iterrows():
        lines.append(f"ID: {row.get('id', '')}")
        lines.append(f"Sentence: {clean_text(row.get('sentence', ''))}")
        lines.append(f"Subject: {clean_text(row.get('subject', ''))}")
        lines.append(f"Relation: {clean_text(row.get('relation', ''))}")
        lines.append(f"Object: {clean_text(row.get('object', ''))}")
        if "evidence" in df.columns:
            lines.append(f"Evidence: {clean_text(row.get('evidence', ''))}")
        lines.append("-" * 60)

    return "\n".join(lines) + "\n"


# =========================
# MAIN
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    # Standardize column names: strip spaces
    df.columns = [c.strip() for c in df.columns]

    required_cols = {"sentence", "subject", "relation", "object"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure expected optional columns exist where useful
    has_evidence = "evidence" in df.columns
    has_split = "split" in df.columns

    # Clean text columns
    text_cols = ["sentence", "subject", "relation", "object"]
    if has_evidence:
        text_cols.append("evidence")
    if has_split:
        text_cols.append("split")

    for col in text_cols:
        df[col] = df[col].apply(clean_text)

    # Basic stats
    total_rows = len(df)

    # Empty counts
    empty_counts = {}
    for col in df.columns:
        empty_counts[col] = int(df[col].apply(lambda x: clean_text(x) == "").sum())

    # Duplicate checks
    duplicate_sentence_count = int(df.duplicated(subset=["sentence"]).sum())
    duplicate_triple_count = int(df.duplicated(subset=["subject", "relation", "object"]).sum())

    # Length stats
    df["_sentence_chars"] = df["sentence"].apply(char_count)
    df["_sentence_words"] = df["sentence"].apply(word_count)
    df["_subject_words"] = df["subject"].apply(word_count)
    df["_object_words"] = df["object"].apply(word_count)

    # Containment checks
    df["_subject_in_sentence"] = df.apply(
        lambda row: contains_normalized(row["sentence"], row["subject"]),
        axis=1
    )

    df["_object_in_sentence"] = df.apply(
        lambda row: contains_normalized(row["sentence"], row["object"]),
        axis=1
    )

    if has_evidence:
        df["_evidence_in_sentence"] = df.apply(
            lambda row: contains_normalized(row["sentence"], row["evidence"]),
            axis=1
        )
    else:
        df["_evidence_in_sentence"] = False

    # Potential problem flags
    df["_problem_empty_main_field"] = df.apply(
        lambda row: (
            clean_text(row["sentence"]) == ""
            or clean_text(row["subject"]) == ""
            or clean_text(row["relation"]) == ""
            or clean_text(row["object"]) == ""
        ),
        axis=1
    )

    df["_problem_subject_not_in_sentence"] = ~df["_subject_in_sentence"]
    df["_problem_object_not_in_sentence"] = ~df["_object_in_sentence"]

    if has_evidence:
        df["_problem_evidence_not_in_sentence"] = (
            df["evidence"].apply(clean_text) != ""
        ) & (~df["_evidence_in_sentence"])
    else:
        df["_problem_evidence_not_in_sentence"] = False

    df["_problem_sentence_too_short"] = df["_sentence_words"] < 3
    df["_problem_sentence_too_long"] = df["_sentence_words"] > 35

    df["_has_any_problem"] = (
        df["_problem_empty_main_field"]
        | df["_problem_subject_not_in_sentence"]
        | df["_problem_object_not_in_sentence"]
        | df["_problem_evidence_not_in_sentence"]
        | df["_problem_sentence_too_short"]
        | df["_problem_sentence_too_long"]
    )

    problem_rows = df[df["_has_any_problem"]].copy()

    # Save problem rows for inspection
    PROBLEM_ROWS_CSV.parent.mkdir(parents=True, exist_ok=True)

    export_cols = [c for c in df.columns if not c.startswith("_")]
    debug_cols = [
        "_subject_in_sentence",
        "_object_in_sentence",
        "_evidence_in_sentence",
        "_sentence_words",
        "_problem_empty_main_field",
        "_problem_subject_not_in_sentence",
        "_problem_object_not_in_sentence",
        "_problem_evidence_not_in_sentence",
        "_problem_sentence_too_short",
        "_problem_sentence_too_long",
    ]

    problem_rows[export_cols + debug_cols].to_csv(
        PROBLEM_ROWS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # Build summary
    lines = []

    lines.append("==============================")
    lines.append("PEP DATASET EXPLORATION REPORT")
    lines.append("==============================\n")

    lines.append("File")
    lines.append("----")
    lines.append(f"Input file: {INPUT_CSV}")
    lines.append(f"Total rows: {total_rows}")
    lines.append(f"Columns: {', '.join(df.columns)}\n")

    lines.append("Empty Values")
    lines.append("------------")
    for col, count in empty_counts.items():
        lines.append(f"{col}: {count}")
    lines.append("")

    lines.append("Duplicates")
    lines.append("----------")
    lines.append(f"Duplicate sentences: {duplicate_sentence_count}")
    lines.append(f"Duplicate subject-relation-object triples: {duplicate_triple_count}\n")

    lines.append("Sentence Length")
    lines.append("---------------")
    lines.append(f"Min words: {df['_sentence_words'].min()}")
    lines.append(f"Max words: {df['_sentence_words'].max()}")
    lines.append(f"Mean words: {df['_sentence_words'].mean():.2f}")
    lines.append(f"Median words: {df['_sentence_words'].median():.2f}")
    lines.append(f"Min chars: {df['_sentence_chars'].min()}")
    lines.append(f"Max chars: {df['_sentence_chars'].max()}")
    lines.append(f"Mean chars: {df['_sentence_chars'].mean():.2f}\n")

    lines.append("Relation Distribution")
    lines.append("---------------------")
    relation_counts = df["relation"].value_counts(dropna=False)
    for rel, count in relation_counts.items():
        lines.append(f"{rel}: {count}")
    lines.append("")

    if has_split:
        lines.append("Split Distribution")
        lines.append("------------------")
        split_counts = df["split"].value_counts(dropna=False)
        for split, count in split_counts.items():
            split_label = split if clean_text(split) else "(empty)"
            lines.append(f"{split_label}: {count}")
        lines.append("")

    lines.append("Containment Checks")
    lines.append("------------------")
    lines.append(f"Subject appears in sentence: {df['_subject_in_sentence'].sum()} / {total_rows}")
    lines.append(f"Object appears in sentence: {df['_object_in_sentence'].sum()} / {total_rows}")

    if has_evidence:
        non_empty_evidence = int((df["evidence"].apply(clean_text) != "").sum())
        lines.append(f"Non-empty evidence rows: {non_empty_evidence} / {total_rows}")
        lines.append(f"Evidence appears in sentence: {df['_evidence_in_sentence'].sum()} / {total_rows}")
    lines.append("")

    lines.append("Problem Flags")
    lines.append("-------------")
    lines.append(f"Rows with any problem: {df['_has_any_problem'].sum()} / {total_rows}")
    lines.append(f"Empty main field: {df['_problem_empty_main_field'].sum()}")
    lines.append(f"Subject not in sentence: {df['_problem_subject_not_in_sentence'].sum()}")
    lines.append(f"Object not in sentence: {df['_problem_object_not_in_sentence'].sum()}")
    lines.append(f"Evidence not in sentence: {df['_problem_evidence_not_in_sentence'].sum()}")
    lines.append(f"Sentence too short (<3 words): {df['_problem_sentence_too_short'].sum()}")
    lines.append(f"Sentence too long (>35 words): {df['_problem_sentence_too_long'].sum()}\n")

    lines.append("Sample Rows")
    lines.append("-----------")
    lines.append(preview_rows(df, 5))

    lines.append("Sample Problem Rows")
    lines.append("-------------------")
    lines.append(preview_rows(problem_rows, 10))

    lines.append(f"Problem rows saved to: {PROBLEM_ROWS_CSV}")
    lines.append("Done.")

    report = "\n".join(lines)

    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSummary saved to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()