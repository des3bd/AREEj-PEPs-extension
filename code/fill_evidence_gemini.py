import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel
from google import genai
from google.genai import types


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("data/review/pep_review_dataset_relations_extracted.csv")
OUTPUT_CSV = Path("data/review/pep_review_dataset_with_evidence.csv")

MODEL_NAME = "gemini-2.5-pro"

# Test first with 20 rows, then change to None
MAX_ROWS = None

# If True, starts fresh from INPUT_CSV
# If False, resumes from OUTPUT_CSV if it exists
OVERWRITE_OUTPUT = True


# =========================
# STRUCTURED OUTPUT SCHEMA
# =========================

class EvidenceExtraction(BaseModel):
    evidence: str


# =========================
# GEMINI SETUP
# =========================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in your environment variables. "
        "Check it in PowerShell using: echo $env:GEMINI_API_KEY"
    )

client = genai.Client(api_key=api_key)


# =========================
# HELPERS
# =========================

def is_empty(value) -> bool:
    if pd.isna(value):
        return True

    value = str(value).strip()
    return value == "" or value.lower() in {"nan", "none", "null"}


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


# =========================
# PROMPT
# =========================

def build_prompt(sentence: str, subject: str, relation: str, obj: str) -> str:
    return f"""
You are preparing evidence annotations for an Arabic relation extraction dataset in the style of AREEj.

Your task:
Extract the shortest useful evidence phrase from the sentence that proves the relation.

Sentence:
{sentence}

Subject:
{subject}

Relation:
{relation}

Object:
{obj}

Meaning of the relation:
The subject holds or held the object position/title.

Evidence rules:
- Return only the evidence phrase.
- The evidence must be copied from the sentence as much as possible.
- Do not invent words or facts.
- Do not use outside knowledge.
- Prefer the shortest phrase that clearly proves the relation.
- If the role phrase alone clearly proves the relation, use the role phrase.
- If the sentence uses a trigger such as "تم تعيين", "شغل", "تولى", "كان", "هو", include the trigger only if needed to prove the relation.
- If the object is normalized differently from the sentence, use the wording that appears in the sentence.

Good examples:

Sentence: "مبارك حمود سعدون الطشه نائب في مجلس الأمة الكويتي."
Subject: "مبارك حمود سعدون الطشه"
Object: "نائب في مجلس الأمة الكويتي"
Evidence: "نائب في مجلس الأمة الكويتي"

Sentence: "اللواء منصور العوضي هو وكيل وزارة الداخلية المساعد لشؤون أمن المنافذ."
Subject: "اللواء منصور العوضي"
Object: "وكيل وزارة الداخلية المساعد لشؤون أمن المنافذ"
Evidence: "هو وكيل وزارة الداخلية المساعد لشؤون أمن المنافذ"

Sentence: "تم تعيين محمد ابطيحان الدويهيس وزيرا للتخطيط."
Subject: "محمد ابطيحان الدويهيس"
Object: "وزير التخطيط"
Evidence: "تم تعيين محمد ابطيحان الدويهيس وزيرا للتخطيط"

Sentence: "عبدالله متعب مسفر مرزوق العرادة كان عضوا في مجلس الأمة الكويتي."
Subject: "عبدالله متعب مسفر مرزوق العرادة"
Object: "عضو في مجلس الأمة الكويتي"
Evidence: "كان عضوا في مجلس الأمة الكويتي"

Return only the structured JSON field:
evidence.
"""


# =========================
# GEMINI CALL
# =========================

def extract_evidence(sentence: str, subject: str, relation: str, obj: str) -> str:
    prompt = build_prompt(sentence, subject, relation, obj)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvidenceExtraction,
            temperature=0,
        ),
    )

    if response.parsed:
        return response.parsed.evidence.strip()

    data = json.loads(response.text)
    return str(data.get("evidence", "")).strip()


# =========================
# MAIN
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    if OUTPUT_CSV.exists() and not OVERWRITE_OUTPUT:
        print(f"Existing output found. Resuming from: {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
    else:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {"sentence", "subject", "relation", "object"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "evidence" not in df.columns:
        df["evidence"] = ""

    # Force string/object dtype to avoid pandas warnings
    for col in ["sentence", "subject", "relation", "object", "evidence"]:
        df[col] = df[col].fillna("").astype("object")

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS).copy()

    processed = 0
    skipped = 0
    errors = 0

    for idx in tqdm(df.index, desc="Extracting evidence"):
        if not is_empty(df.at[idx, "evidence"]):
            skipped += 1
            continue

        sentence = safe_text(df.at[idx, "sentence"])
        subject = safe_text(df.at[idx, "subject"])
        relation = safe_text(df.at[idx, "relation"])
        obj = safe_text(df.at[idx, "object"])

        try:
            evidence = extract_evidence(sentence, subject, relation, obj)
            df.at[idx, "evidence"] = evidence
            processed += 1

        except Exception as e:
            errors += 1
            print(f"\nError at row {idx}: {e}")

        # Save after every row
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Processed rows: {processed}")
    print(f"Skipped rows: {skipped}")
    print(f"Errors: {errors}")
    print(f"Rows saved: {len(df)}")
    print(f"Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()