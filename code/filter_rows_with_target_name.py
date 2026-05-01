import pandas as pd
import re
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("data/review/pep_review_dataset.csv")
OUTPUT_CSV = Path("data/review/pep_review_dataset_target_only.csv")

TARGET_COL = "target_name_ar"
SENTENCE_COL = "sentence"

# Require at least this many consecutive name tokens to match
# 2 is flexible, 3 is stricter
MIN_MATCH_TOKENS = 2


# =========================
# HELPERS
# =========================

def normalize_space(text: str) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_arabic(text: str) -> str:
    """
    Light Arabic normalization for safer matching.
    Example:
    أ / إ / آ -> ا
    ى -> ي
    ة -> ه
    removes diacritics
    """
    text = normalize_space(text)

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

    return normalize_space(text)


def arabic_name_tokens(name: str) -> list[str]:
    """
    Extract clean Arabic tokens from target name.
    """
    name = normalize_arabic(name)

    # Keep Arabic letters and spaces only
    name = re.sub(r"[^\u0600-\u06FF\s]", " ", name)
    name = normalize_space(name)

    tokens = name.split()

    # Remove very short tokens if any
    tokens = [t for t in tokens if len(t) >= 2]

    return tokens


def generate_name_parts(tokens: list[str], min_tokens: int = 2) -> list[str]:
    """
    Generate consecutive parts of the name.
    Example:
    سالم صباح السالم المبارك الصباح
    creates:
    سالم صباح
    صباح السالم
    السالم المبارك
    ...
    سالم صباح السالم
    صباح السالم المبارك
    etc.
    """
    parts = []

    n = len(tokens)

    for size in range(n, min_tokens - 1, -1):
        for start in range(0, n - size + 1):
            part = " ".join(tokens[start:start + size])
            parts.append(part)

    return parts


def target_name_or_part_in_sentence(target_name: str, sentence: str) -> bool:
    target_norm = normalize_arabic(target_name)
    sentence_norm = normalize_arabic(sentence)

    if not target_norm or not sentence_norm:
        return False

    # Full normalized name match
    if target_norm in sentence_norm:
        return True

    tokens = arabic_name_tokens(target_name)

    if len(tokens) < MIN_MATCH_TOKENS:
        return False

    name_parts = generate_name_parts(tokens, MIN_MATCH_TOKENS)

    for part in name_parts:
        # Use word-boundary-like matching for Arabic using spaces/non-Arabic chars
        pattern = rf"(?<![\u0600-\u06FF]){re.escape(part)}(?![\u0600-\u06FF])"
        if re.search(pattern, sentence_norm):
            return True

    return False


# =========================
# MAIN
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {TARGET_COL, SENTENCE_COL}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before_count = len(df)

    df["_has_target_name_or_part"] = df.apply(
        lambda row: target_name_or_part_in_sentence(
            row[TARGET_COL],
            row[SENTENCE_COL]
        ),
        axis=1
    )

    filtered = df[df["_has_target_name_or_part"]].copy()
    filtered = filtered.drop(columns=["_has_target_name_or_part"])

    # Remove exact duplicate target + sentence rows
    filtered = filtered.drop_duplicates(subset=[TARGET_COL, SENTENCE_COL]).copy()

    # Rebuild IDs
    filtered["id"] = range(1, len(filtered) + 1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Input rows: {before_count}")
    print(f"Kept rows: {len(filtered)}")
    print(f"Dropped rows: {before_count - len(filtered)}")
    print(f"Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()