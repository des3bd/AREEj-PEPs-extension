import json
import re
from pathlib import Path
from collections import Counter


# =========================
# CONFIG
# =========================

INPUT_DIR = Path("data/raw")

MIN_LEN = 40
MAX_LEN = 700
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
    """
    Light Arabic normalization to make name matching easier.
    """
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

    # remove Arabic diacritics
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    return text


def arabic_ratio(text: str) -> float:
    """
    Ratio of Arabic letters to all letters.
    This avoids treating a mostly-English paragraph as Arabic.
    """
    if not text:
        return 0.0

    letters = re.findall(r"[A-Za-z\u0600-\u06FF]", text)
    if not letters:
        return 0.0

    arabic_letters = ARABIC_RE.findall(text)
    return len(arabic_letters) / len(letters)


def latin_ratio(text: str) -> float:
    if not text:
        return 0.0

    letters = re.findall(r"[A-Za-z\u0600-\u06FF]", text)
    if not letters:
        return 0.0

    latin_letters = LATIN_RE.findall(text)
    return len(latin_letters) / len(letters)


def get_nested(data, path, default=None):
    current = data

    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def is_positive_prediction(data: dict) -> bool:
    verdict = (
        get_nested(data, ["engine_result", "verdict"])
        or get_nested(data, ["engine_result", "raw_output", "verdict"])
        or get_nested(data, ["scored_result", "prediction", "verdict"])
        or ""
    )

    raw_is_pep = get_nested(data, ["engine_result", "raw_output", "is_pep"])
    passed_gates = get_nested(data, ["engine_result", "raw_output", "passed_gates"])

    verdict = str(verdict).lower().strip()

    positive_verdicts = {
        "pep",
        "true_pep",
        "confirmed_pep",
        "likely_pep",
        "review",
    }

    return verdict in positive_verdicts or raw_is_pep is True or passed_gates is True


def target_present_flag(item: dict) -> bool:
    flags = [
        item.get("sentence_has_subject_name_match"),
        item.get("document_name_supported"),
        item.get("name_matched"),
        item.get("person_name_found"),
    ]

    return any(flag is True for flag in flags)


def target_name_in_sentence(target_name_ar: str, sentence: str) -> bool:
    """
    Checks if the Arabic target name appears in the sentence.
    Also checks partial name match using first two/three tokens.
    """
    target_norm = normalize_arabic(target_name_ar)
    sent_norm = normalize_arabic(sentence)

    if not target_norm or not sent_norm:
        return False

    if target_norm in sent_norm:
        return True

    tokens = target_norm.split()

    if len(tokens) >= 3:
        first_three = " ".join(tokens[:3])
        if first_three in sent_norm:
            return True

    if len(tokens) >= 2:
        first_two = " ".join(tokens[:2])
        if first_two in sent_norm:
            return True

    return False


def looks_like_clean_arabic_sentence(sentence: str) -> bool:
    sentence = normalize_space(sentence)

    if len(sentence) < MIN_LEN:
        return False

    if len(sentence) > MAX_LEN:
        return False

    if arabic_ratio(sentence) < MIN_ARABIC_RATIO:
        return False

    # reject obvious scraped/list/table noise
    bad_phrases = [
        "List of persons",
        "Type FSD ID",
        "Page ",
        "Share Share Options",
        "Copy Link",
        "Email Facebook",
        "Twitter Telegram",
        "Updates About My Website",
    ]

    for phrase in bad_phrases:
        if phrase.lower() in sentence.lower():
            return False

    return True


def extract_candidate_texts(data: dict):
    candidates = []

    metadata = get_nested(data, ["engine_result", "raw_output", "metadata"], {})

    role_traces = metadata.get("role_dependency_traces", [])
    if isinstance(role_traces, list):
        for item in role_traces:
            if not isinstance(item, dict):
                continue

            sentence = normalize_space(item.get("sentence", ""))

            if sentence:
                candidates.append({
                    "source_section": "role_dependency_traces",
                    "sentence": sentence,
                    "target_flag": target_present_flag(item),
                })

    role_signals = metadata.get("transparency_document_role_signals", [])
    if isinstance(role_signals, list):
        for item in role_signals:
            if not isinstance(item, dict):
                continue

            sentence = normalize_space(item.get("snippet_text", ""))

            if sentence:
                candidates.append({
                    "source_section": "transparency_document_role_signals",
                    "sentence": sentence,
                    "target_flag": target_present_flag(item),
                })

    evidence_items = get_nested(data, ["engine_result", "raw_output", "evidence"], [])
    if isinstance(evidence_items, list):
        for item in evidence_items:
            if not isinstance(item, dict):
                continue

            for field in ["sentence", "snippet", "text", "evidence", "snippet_text"]:
                sentence = normalize_space(item.get(field, ""))

                if sentence:
                    candidates.append({
                        "source_section": f"evidence.{field}",
                        "sentence": sentence,
                        "target_flag": True,
                    })

    return candidates


# =========================
# MAIN
# =========================

def main():
    json_files = sorted(INPUT_DIR.glob("*.json"))

    stats = Counter()
    section_counter = Counter()

    examples_clean = []
    examples_target_name_match = []
    examples_rejected_by_ratio = []
    examples_rejected_by_length = []

    unique_clean_sentences = set()

    for file_path in json_files:
        stats["json_files"] += 1

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        case_positive = is_positive_prediction(data)
        target_name_ar = get_nested(data, ["case", "name_ar"], "")
        case_id = data.get("case_id") or get_nested(data, ["case", "case_id"], file_path.stem)

        if case_positive:
            stats["positive_cases"] += 1

        candidates = extract_candidate_texts(data)

        for cand in candidates:
            sentence = cand["sentence"]
            section = cand["source_section"]

            stats["total_candidates"] += 1
            section_counter[section] += 1

            ratio = arabic_ratio(sentence)

            if ARABIC_RE.search(sentence):
                stats["contains_any_arabic"] += 1

            if len(sentence) > MAX_LEN:
                stats["too_long"] += 1
                if len(examples_rejected_by_length) < 3:
                    examples_rejected_by_length.append((case_id, target_name_ar, len(sentence), sentence[:300]))
                continue

            if len(sentence) < MIN_LEN:
                stats["too_short"] += 1
                continue

            if ratio < MIN_ARABIC_RATIO:
                stats["low_arabic_ratio"] += 1
                if len(examples_rejected_by_ratio) < 3:
                    examples_rejected_by_ratio.append((case_id, target_name_ar, ratio, sentence[:300]))
                continue

            stats["clean_arabic_length_ratio"] += 1

            has_target_name = target_name_in_sentence(target_name_ar, sentence)
            has_target_flag = cand["target_flag"]

            if has_target_name:
                stats["clean_with_target_name_match"] += 1
                if len(examples_target_name_match) < 5:
                    examples_target_name_match.append((case_id, target_name_ar, sentence))

            if has_target_flag:
                stats["clean_with_json_target_flag"] += 1

            if case_positive:
                stats["clean_from_positive_case"] += 1

            # likely useful for your dataset
            if has_target_name or has_target_flag or case_positive:
                stats["likely_dataset_candidates"] += 1
                unique_clean_sentences.add((target_name_ar, sentence))

                if len(examples_clean) < 5:
                    examples_clean.append((case_id, target_name_ar, sentence))

    print("\n==============================")
    print("JSON SENTENCE EXPLORATION V2")
    print("==============================\n")

    print("Files")
    print("-----")
    print(f"Total JSON files: {stats['json_files']}")
    print(f"Positive prediction cases: {stats['positive_cases']}")

    print("\nCandidates")
    print("----------")
    print(f"Total candidates: {stats['total_candidates']}")
    print(f"Contains any Arabic: {stats['contains_any_arabic']}")
    print(f"Too short: {stats['too_short']}")
    print(f"Too long: {stats['too_long']}")
    print(f"Rejected by low Arabic ratio: {stats['low_arabic_ratio']}")
    print(f"Clean Arabic by length + ratio: {stats['clean_arabic_length_ratio']}")

    print("\nTarget / Useful Counts")
    print("----------------------")
    print(f"Clean with target name match: {stats['clean_with_target_name_match']}")
    print(f"Clean with JSON target flag: {stats['clean_with_json_target_flag']}")
    print(f"Clean from positive case: {stats['clean_from_positive_case']}")
    print(f"Likely dataset candidates: {stats['likely_dataset_candidates']}")
    print(f"Unique likely dataset candidates: {len(unique_clean_sentences)}")

    print("\nSource Sections")
    print("---------------")
    for section, count in section_counter.most_common():
        print(f"{section}: {count}")

    print("\nExamples: Likely Dataset Candidates")
    print("-----------------------------------")
    for case_id, target, sent in examples_clean:
        print(f"\nCase: {case_id}")
        print(f"Target: {target}")
        print(f"Sentence: {sent}")

    print("\nExamples: Target Name Match")
    print("---------------------------")
    for case_id, target, sent in examples_target_name_match:
        print(f"\nCase: {case_id}")
        print(f"Target: {target}")
        print(f"Sentence: {sent}")

    print("\nExamples: Rejected by Low Arabic Ratio")
    print("--------------------------------------")
    for case_id, target, ratio, sent in examples_rejected_by_ratio:
        print(f"\nCase: {case_id}")
        print(f"Target: {target}")
        print(f"Arabic ratio: {ratio:.2f}")
        print(f"Text: {sent}")

    print("\nExamples: Rejected by Length")
    print("----------------------------")
    for case_id, target, length, sent in examples_rejected_by_length:
        print(f"\nCase: {case_id}")
        print(f"Target: {target}")
        print(f"Length: {length}")
        print(f"Text: {sent}")

    print("\nDone.")


if __name__ == "__main__":
    main()