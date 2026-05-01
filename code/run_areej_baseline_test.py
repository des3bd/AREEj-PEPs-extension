import re
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("data/final/pep_areej_final_dataset.csv")
OUTPUT_CSV = Path("data/final/pep_areej_baseline_test_predictions.csv")

MODEL_NAME = "U4RASD/AREEj"

BATCH_SIZE = 4

MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 160

NUM_BEAMS = 4


# =========================
# HELPERS
# =========================

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# =========================
# MAIN
# =========================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {"id", "sentence", "target_output", "split"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use test split only for fair comparison with fine-tuned model
    df["split"] = df["split"].apply(clean_text)
    df = df[df["split"] == "test"].copy()

    if df.empty:
        raise ValueError("No rows found with split = test")

    print(f"Test rows to process: {len(df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading baseline model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    sentences = [clean_text(x) for x in df["sentence"].tolist()]
    predictions = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sentences), BATCH_SIZE), desc="Running AREEj baseline on test split"):
            batch_sentences = sentences[start:start + BATCH_SIZE]

            inputs = tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
            ).to(device)

            generated_ids = model.generate(
                **inputs,
                max_length=MAX_OUTPUT_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )

            batch_outputs = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )

            batch_outputs = [clean_text(x) for x in batch_outputs]
            predictions.extend(batch_outputs)

    df["areej_baseline_prediction"] = predictions

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nSample predictions:")
    for i in range(min(5, len(df))):
        print("\n---")
        print("Sentence:", clean_text(df.iloc[i]["sentence"]))
        print("Gold:", clean_text(df.iloc[i]["target_output"]))
        print("Baseline:", clean_text(df.iloc[i]["areej_baseline_prediction"]))


if __name__ == "__main__":
    main()