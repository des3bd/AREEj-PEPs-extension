import re
import pandas as pd
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("data/final/pep_areej_finetuned_test_predictions.csv")
OUTPUT_CSV = Path("data/final/pep_areej_finetuned_test_evaluated.csv")
SUMMARY_TXT = Path("data/final/pep_areej_finetuned_test_summary.txt")

PRED_COL = "areej_finetuned_prediction"


# =========================
# HELPERS
# =========================

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def remove_special_tokens(text: str) -> str:
    text = clean_text(text)
    text = text.replace("</s>", " ")
    text = text.replace("<s>", " ")
    text = text.replace("<pad>", " ")
    text = text.replace("ar_AR", " ")
    return clean_text(text)


def normalize_arabic(text: str) -> str:
    text = remove_special_tokens(text)

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

    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    return clean_text(text)


def contains_match(long_text: str, short_text: str) -> bool:
    long_norm = normalize_arabic(long_text)
    short_norm = normalize_arabic(short_text)

    if not long_norm or not short_norm:
        return False

    return short_norm in long_norm


def relation_match(pred_relation: str, gold_relation: str) -> bool:
    pred = clean_text(pred_relation).lower().replace("_", " ")
    gold = clean_text(gold_relation).lower().replace("_", " ")

    acceptable = {
        "position held",
        "position_held",
        "held position",
        "holds position",
        "holds_position",
    }

    if pred in acceptable and gold in acceptable:
        return True

    return pred == gold


def parse_first_areej_relation(prediction: str) -> dict:
    text = remove_special_tokens(prediction)

    result = {
        "pred_subject": "",
        "pred_subject_type": "",
        "pred_object": "",
        "pred_object_type": "",
        "pred_relation": "",
        "pred_evidence": "",
    }

    if "<bor>" not in text:
        return result

    relation_block = text.split("<bor>", 1)[1]

    if "<e>" in relation_block:
        relation_block = relation_block.split("<e>", 1)[0]

    relation_block = clean_text(relation_block)

    type_tags = [
        "per", "org", "loc", "concept", "eve", "time",
        "misc", "product", "work", "law", "language", "date", "media", "unk"
    ]

    type_pattern = "|".join(type_tags)

    pattern = rf"^(.*?)<({type_pattern})>(.*?)<({type_pattern})>(.*?)<rt>(.*)$"

    match = re.search(pattern, relation_block)

    if not match:
        if "<rt>" in relation_block:
            before_rt, after_rt = relation_block.split("<rt>", 1)
            result["pred_relation"] = clean_text(before_rt)
            result["pred_evidence"] = clean_text(after_rt)
        return result

    result["pred_subject"] = clean_text(match.group(1))
    result["pred_subject_type"] = clean_text(match.group(2))
    result["pred_object"] = clean_text(match.group(3))
    result["pred_object_type"] = clean_text(match.group(4))
    result["pred_relation"] = clean_text(match.group(5))
    result["pred_evidence"] = clean_text(match.group(6))

    return result


# =========================
# MAIN
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {
        "id",
        "sentence",
        "subject",
        "relation",
        "object",
        "evidence",
        "target_output",
        PRED_COL,
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    parsed_rows = []

    for _, row in df.iterrows():
        parsed_rows.append(parse_first_areej_relation(row[PRED_COL]))

    parsed_df = pd.DataFrame(parsed_rows)
    df = pd.concat([df, parsed_df], axis=1)

    df["subject_match"] = df.apply(
        lambda row: contains_match(row["pred_subject"], row["subject"])
        or contains_match(row["subject"], row["pred_subject"]),
        axis=1
    )

    df["object_match"] = df.apply(
        lambda row: contains_match(row["pred_object"], row["object"])
        or contains_match(row["object"], row["pred_object"]),
        axis=1
    )

    df["relation_match"] = df.apply(
        lambda row: relation_match(row["pred_relation"], row["relation"]),
        axis=1
    )

    df["evidence_exact_or_contains_match"] = df.apply(
        lambda row: contains_match(row["pred_evidence"], row["evidence"])
        or contains_match(row["evidence"], row["pred_evidence"]),
        axis=1
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    total = len(df)

    metrics = {
        "total_rows": total,
        "subject_match": df["subject_match"].mean(),
        "object_match": df["object_match"].mean(),
        "relation_match": df["relation_match"].mean(),
        "evidence_exact_or_contains_match": df["evidence_exact_or_contains_match"].mean(),
    }

    lines = []
    lines.append("==============================")
    lines.append("FINE-TUNED AREEj TEST EVALUATION")
    lines.append("==============================\n")

    lines.append(f"Input: {INPUT_CSV}")
    lines.append(f"Output: {OUTPUT_CSV}")
    lines.append(f"Total test rows: {total}\n")

    lines.append("Metrics")
    lines.append("-------")

    for key, value in metrics.items():
        if key == "total_rows":
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {value:.4f}")

    lines.append("\nRelation predictions distribution")
    lines.append("---------------------------------")
    for rel, count in df["pred_relation"].value_counts(dropna=False).items():
        rel = rel if clean_text(rel) else "(empty)"
        lines.append(f"{rel}: {count}")

    lines.append("\nSample predictions")
    lines.append("------------------")
    for _, row in df.head(10).iterrows():
        lines.append(f"\nID: {row['id']}")
        lines.append(f"Sentence: {clean_text(row['sentence'])}")
        lines.append(f"Gold: {clean_text(row['target_output'])}")
        lines.append(f"Prediction: {clean_text(row[PRED_COL])}")
        lines.append(
            f"Parsed prediction: "
            f"{clean_text(row['pred_subject'])} | "
            f"{clean_text(row['pred_relation'])} | "
            f"{clean_text(row['pred_object'])} | "
            f"evidence={clean_text(row['pred_evidence'])}"
        )
        lines.append(
            f"Matches: subject={row['subject_match']}, "
            f"object={row['object_match']}, "
            f"relation={row['relation_match']}, "
            f"evidence={row['evidence_exact_or_contains_match']}"
        )

    report = "\n".join(lines)

    SUMMARY_TXT.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved evaluated CSV to: {OUTPUT_CSV}")
    print(f"Saved summary to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()