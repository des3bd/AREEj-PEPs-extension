import json
import csv
import re
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_DIR = Path("data/raw")
OUTPUT_CSV = Path("data/review/pep_review_dataset.csv")

MIN_LEN = 35
MAX_LEN = 500
MIN_ARABIC_RATIO = 0.40


# =========================
# REGEX
# =========================

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")


# =========================
# HELPERS
# =========================

def normalize_space(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_arabic(text: str) -> str:
    if not text:
        return ""

    text = normalize_space(text)

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    return text


def arabic_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z\u0600-\u06FF]", text or "")
    if not letters:
        return 0.0

    arabic_letters = ARABIC_RE.findall(text or "")
    return len(arabic_letters) / len(letters)


def get_nested(data, path, default=None):
    current = data

    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def target_present_flag(item: dict) -> bool:
    flags = [
        item.get("sentence_has_subject_name_match"),
        item.get("document_name_supported"),
        item.get("name_matched"),
        item.get("person_name_found"),
    ]

    return any(flag is True for flag in flags)


def target_name_in_sentence(target_name_ar: str, sentence: str) -> bool:
    target_norm = normalize_arabic(target_name_ar)
    sent_norm = normalize_arabic(sentence)

    if not target_norm or not sent_norm:
        return False

    if target_norm in sent_norm:
        return True

    tokens = target_norm.split()

    # Some pages may include only part of the full name.
    if len(tokens) >= 4:
        first_four = " ".join(tokens[:4])
        if first_four in sent_norm:
            return True

    if len(tokens) >= 3:
        first_three = " ".join(tokens[:3])
        if first_three in sent_norm:
            return True

    return False


def looks_like_noise(text: str) -> bool:
    lower = text.lower()

    bad_phrases = [
        "list of persons",
        "type fsd id",
        "copy link",
        "email facebook",
        "twitter telegram",
        "share share options",
        "updates about my website",
        "من ويكيبيديا، الموسوعه الحره معلومات شخصيه",
    ]

    return any(phrase in lower for phrase in bad_phrases)


def is_clean_arabic_text(text: str) -> bool:
    text = normalize_space(text)

    if len(text) < MIN_LEN:
        return False

    if len(text) > MAX_LEN:
        return False

    if arabic_ratio(text) < MIN_ARABIC_RATIO:
        return False

    if looks_like_noise(text):
        return False

    return True


def split_arabic_chunks(text: str) -> list[str]:
    """
    Splits long scraped Arabic text into smaller chunks.
    This is not perfect sentence segmentation, but it helps reduce huge blocks.
    """
    text = normalize_space(text)

    # Split on punctuation and common separators.
    parts = re.split(r"[.!؟?؛;،]\s+|\s+-\s+|\s+\|\s+", text)

    chunks = []

    for part in parts:
        part = normalize_space(part)

        if not part:
            continue

        # If still very long, split around relation-relevant phrases.
        if len(part) > MAX_LEN:
            subparts = re.split(
                r"(?=(?:شارك|شغل|يشغل|تولى|عين|عُين|انتخب|فاز|نايب|نائب|وزير|رئيس|عضو|محافظ|سفير))",
                part
            )

            for sub in subparts:
                sub = normalize_space(sub)
                if sub:
                    chunks.append(sub)
        else:
            chunks.append(part)

    return chunks


def extract_candidate_items(data: dict):
    candidates = []

    metadata = get_nested(data, ["engine_result", "raw_output", "metadata"], {})

    role_traces = metadata.get("role_dependency_traces", [])
    if isinstance(role_traces, list):
        for item in role_traces:
            if isinstance(item, dict) and item.get("sentence"):
                candidates.append({
                    "text": item.get("sentence", ""),
                    "target_flag": target_present_flag(item),
                })

    role_signals = metadata.get("transparency_document_role_signals", [])
    if isinstance(role_signals, list):
        for item in role_signals:
            if isinstance(item, dict) and item.get("snippet_text"):
                candidates.append({
                    "text": item.get("snippet_text", ""),
                    "target_flag": target_present_flag(item),
                })

    evidence_items = get_nested(data, ["engine_result", "raw_output", "evidence"], [])
    if isinstance(evidence_items, list):
        for item in evidence_items:
            if not isinstance(item, dict):
                continue

            for field in ["sentence", "snippet", "text", "evidence", "snippet_text"]:
                if item.get(field):
                    candidates.append({
                        "text": item.get(field, ""),
                        "target_flag": True,
                    })

    return candidates


# =========================
# MAIN
# =========================

def main():
    json_files = sorted(INPUT_DIR.glob("*.json"))

    rows = []
    seen = set()

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue

        target_name_ar = get_nested(data, ["case", "name_ar"], "")

        if not target_name_ar:
            continue

        items = extract_candidate_items(data)

        for item in items:
            raw_text = normalize_space(item["text"])
            target_flag = item["target_flag"]

            chunks = split_arabic_chunks(raw_text)

            for chunk in chunks:
                chunk = normalize_space(chunk)

                if not is_clean_arabic_text(chunk):
                    continue

                has_target_name = target_name_in_sentence(target_name_ar, chunk)

                # Important: do not keep positive case text alone.
                # We keep only if target appears by name or JSON says target is present.
                if not (has_target_name or target_flag):
                    continue

                key = (target_name_ar, chunk)

                if key in seen:
                    continue

                seen.add(key)

                rows.append({
                    "target_name_ar": target_name_ar,
                    "sentence": chunk,
                })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "target_name_ar",
        "sentence",
        "subject",
        "relation",
        "object",
        "evidence",
        "split",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            writer.writerow({
                "id": idx,
                "target_name_ar": row["target_name_ar"],
                "sentence": row["sentence"],
                "subject": "",
                "relation": "",
                "object": "",
                "evidence": "",
                "split": "",
            })

    print("Done.")
    print(f"JSON files scanned: {len(json_files)}")
    print(f"Rows saved: {len(rows)}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()