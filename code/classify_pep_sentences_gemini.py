import os
import time
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

INPUT_CSV = Path("data/review/pep_review_dataset_target_only.csv")
OUTPUT_CSV = Path("data/review/pep_review_dataset_gemini_labeled.csv")

MODEL_NAME = "gemini-2.5-flash"

MAX_ROWS = None

SLEEP_SECONDS = 0.1


# =========================
# STRUCTURED OUTPUT SCHEMA
# =========================

class PepSentenceDecision(BaseModel):
    pep_sentence: bool


# =========================
# GEMINI SETUP
# =========================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in your environment variables. "
        "Check the variable name in PowerShell."
    )

client = genai.Client(api_key=api_key)


# =========================
# PROMPT
# =========================

def build_prompt(target_name_ar: str, sentence: str) -> str:
    return f"""
You are labeling Arabic relation extraction data for a machine learning project.

Task:
Decide whether the Arabic sentence clearly expresses a PEP-relevant relation for the target person.

Target person:
{target_name_ar}

Sentence:
{sentence}

Label pep_sentence = true ONLY if the sentence clearly states that the target person has or had a politically exposed/public role, such as:
- member of parliament / مجلس الأمة / نائب
- minister / وزير
- ambassador / سفير
- head/chairman/president of a government body / رئيس
- governor / محافظ
- judge/prosecutor/senior official
- senior military/security/government position
- elected public office

Label pep_sentence = false if:
- the sentence only mentions the person without a role
- the person is a writer, student, athlete, artist, private businessperson, or ordinary person
- the role belongs to another person
- the sentence is about elections, voting, news, death, family, crime, or biography without clearly saying the target held a public role
- the sentence is too noisy or unclear
- the target name appears but there is no clear person-role relation

Important:
The decision must be based only on this sentence.
Return only the structured JSON field pep_sentence.
Do not use outside knowledge.
"""


# =========================
# CLASSIFICATION FUNCTION
# =========================

def classify_row(target_name_ar: str, sentence: str) -> bool:
    prompt = build_prompt(target_name_ar, sentence)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PepSentenceDecision,
            temperature=0,
        ),
    )

    if response.parsed:
        return bool(response.parsed.pep_sentence)

    data = json.loads(response.text)
    return bool(data["pep_sentence"])


# =========================
# MAIN
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {"target_name_ar", "sentence"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Resume if output already exists
    if OUTPUT_CSV.exists():
        print(f"Existing output found. Resuming from: {OUTPUT_CSV}")
        out_df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")
    else:
        out_df = df.copy()
        out_df["pep_sentence"] = ""

    if MAX_ROWS is not None:
        out_df = out_df.head(MAX_ROWS).copy()

    for idx in tqdm(out_df.index, desc="Classifying"):
        current_value = str(out_df.at[idx, "pep_sentence"]).strip().lower()

        if current_value in {"true", "false"}:
            continue

        target_name_ar = str(out_df.at[idx, "target_name_ar"])
        sentence = str(out_df.at[idx, "sentence"])

        try:
            result = classify_row(target_name_ar, sentence)
            out_df.at[idx, "pep_sentence"] = result

        except Exception as e:
            out_df.at[idx, "pep_sentence"] = "ERROR"
            print(f"\nError at row {idx}: {e}")

        # Save after every row so progress is not lost
        out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        time.sleep(SLEEP_SECONDS)

    print("Done.")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nLabel counts:")
    print(out_df["pep_sentence"].value_counts(dropna=False))


if __name__ == "__main__":
    main()