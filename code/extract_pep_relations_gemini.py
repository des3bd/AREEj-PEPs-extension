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

INPUT_CSV = Path("data/review/pep_review_dataset_true_only.csv")
OUTPUT_CSV = Path("data/review/pep_review_dataset_relations_extracted.csv")

MODEL_NAME = "gemini-2.5-pro"

# Test first with 20 rows, then change to None for all rows
MAX_ROWS = None

# If True, starts fresh and ignores old output file
OVERWRITE_OUTPUT = True


# =========================
# STRUCTURED OUTPUT SCHEMA
# =========================

class PepRelationExtraction(BaseModel):
    cleaned_sentence: str
    subject: str
    relation: str
    object: str
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

def build_prompt(target_name_ar: str, sentence: str) -> str:
    return f"""
You are preparing Arabic relation extraction data in the style of AREEj.

Each example must contain:
- one clean sentence
- a subject/source entity
- a relation type
- an object/target entity
- an evidence phrase from the sentence that explains the relation

The input may be a long Arabic biography block with many details.
Your task is to extract ONLY ONE clean PEP/public-role relation line from it.

Target person:
{target_name_ar}

Original text:
{sentence}

Allowed relation type:
holds_position

Main task:
Find the clearest sentence or phrase in the original text that says the target person held a public, political, government, parliamentary, judicial, diplomatic, military, or senior state-related role.

If the text contains many possible roles:
- choose only ONE relation
- prefer the clearest PEP/public role
- prefer government/parliament/ministry/diplomatic roles over academic or ordinary jobs
- prefer a role directly attached to the target person
- prefer the shortest clear line that still contains the subject and the role

Examples of strong role objects:
- نائب سابق في مجلس الأمة الكويتي
- نائب في مجلس الأمة
- عضو مجلس الأمة
- وزير المالية
- سفير الكويت لدى الولايات المتحدة
- رئيس مجلس الإدارة
- رئيس هيئة حكومية
- محافظ
- أمين عام
- وكيل وزارة
- قاض
- مدير عام في جهة حكومية
- مهندس بوزارة الكهرباء والماء

Minimal cleaning rules:
- You may extract a shorter relation line from the long original text.
- You may add the target person name to the extracted line if the role phrase depends on previous context and the target is clearly the subject.
- Remove obvious scraping or biography noise, such as:
  "من ويكيبيديا، الموسوعة الحرة", "من ويكيبيديا، الموسوعه الحره", "تعديل مصدري", page/menu text, dates of birth, education, election vote counts, references, repeated website text.
- Do not invent facts.
- Do not add a role that is not in the original text.
- Do not use outside knowledge.
- Keep the Arabic meaning unchanged.
- The cleaned_sentence should be one short clean relation line only.
- The evidence should be the smallest useful phrase that proves the relation, usually the role phrase or the clean relation line.

For the output:
- subject must be the target person as written in the text if possible.
- relation must be exactly: holds_position
- object must be the role/title only.
- evidence must prove the subject-role relation.
- cleaned_sentence must contain only the selected relation line, not the full biography block.

Example:

Original text:
"حمود عبدالله الرقبه من مواليد 1951 ... اما المناصب الرسميه التي شغلها الفقيد فهي كالتالي 1974-1975 مهندس بوزاره الكهرباء والماء 1981 استاذ بكليه الهندسه والبترول"

Good output:
cleaned_sentence: "حمود عبدالله الرقبه مهندس بوزارة الكهرباء والماء."
subject: "حمود عبدالله الرقبه"
relation: "holds_position"
object: "مهندس بوزارة الكهرباء والماء"
evidence: "مهندس بوزارة الكهرباء والماء"

Bad output:
cleaned_sentence: full biography block
object: education degree
object: ordinary biography details

Return only the structured JSON fields:
cleaned_sentence, subject, relation, object, evidence.
"""


# =========================
# EXTRACTION FUNCTION
# =========================

def extract_relation(target_name_ar: str, sentence: str) -> dict:
    prompt = build_prompt(target_name_ar, sentence)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PepRelationExtraction,
            temperature=0,
        ),
    )

    if response.parsed:
        parsed = response.parsed
        return {
            "cleaned_sentence": parsed.cleaned_sentence.strip(),
            "subject": parsed.subject.strip(),
            "relation": parsed.relation.strip(),
            "object": parsed.object.strip(),
            "evidence": parsed.evidence.strip(),
        }

    data = json.loads(response.text)
    return {
        "cleaned_sentence": str(data.get("cleaned_sentence", "")).strip(),
        "subject": str(data.get("subject", "")).strip(),
        "relation": str(data.get("relation", "")).strip(),
        "object": str(data.get("object", "")).strip(),
        "evidence": str(data.get("evidence", "")).strip(),
    }


# =========================
# MAIN
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {"id", "target_name_ar", "sentence"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if OUTPUT_CSV.exists() and not OVERWRITE_OUTPUT:
        print(f"Existing output found. Resuming from: {OUTPUT_CSV}")
        out_df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
    else:
        out_df = df.copy()

    # Ensure text columns are proper object dtype, not float NaN columns
    for col in ["subject", "relation", "object", "evidence", "cleaned_sentence"]:
        if col not in out_df.columns:
            out_df[col] = ""
        out_df[col] = out_df[col].fillna("").astype("object")

    for col in ["target_name_ar", "sentence"]:
        out_df[col] = out_df[col].fillna("").astype("object")

    if MAX_ROWS is not None:
        out_df = out_df.head(MAX_ROWS).copy()

    processed = 0
    skipped = 0
    errors = 0

    for idx in tqdm(out_df.index, desc="Extracting relations"):
        already_done = (
            not is_empty(out_df.at[idx, "subject"])
            and not is_empty(out_df.at[idx, "relation"])
            and not is_empty(out_df.at[idx, "object"])
            and not is_empty(out_df.at[idx, "evidence"])
        )

        if already_done:
            skipped += 1
            continue

        target_name_ar = safe_text(out_df.at[idx, "target_name_ar"])
        sentence = safe_text(out_df.at[idx, "sentence"])

        try:
            result = extract_relation(target_name_ar, sentence)

            out_df.at[idx, "cleaned_sentence"] = result["cleaned_sentence"]
            out_df.at[idx, "subject"] = result["subject"]
            out_df.at[idx, "relation"] = "holds_position"
            out_df.at[idx, "object"] = result["object"]
            out_df.at[idx, "evidence"] = result["evidence"]

            # Replace original long/noisy sentence with the clean extracted relation line
            if result["cleaned_sentence"]:
                out_df.at[idx, "sentence"] = result["cleaned_sentence"]

            processed += 1

        except Exception as e:
            errors += 1
            print(f"\nError at row {idx}: {e}")

        # Save after each row so progress is not lost
        out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Drop helper column before final save
    if "cleaned_sentence" in out_df.columns:
        out_df = out_df.drop(columns=["cleaned_sentence"])

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

    final_columns = [col for col in final_columns if col in out_df.columns]
    out_df = out_df[final_columns]

    out_df["id"] = range(1, len(out_df) + 1)

    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Processed rows: {processed}")
    print(f"Skipped rows: {skipped}")
    print(f"Errors: {errors}")
    print(f"Rows saved: {len(out_df)}")
    print(f"Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()