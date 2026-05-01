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
OUTPUT_CSV = Path("data/final/pep_areej_finetuned_test_predictions.csv")

MODEL_DIR = Path("models/areej_pep_finetuned")

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


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Fine-tuned model folder not found: {MODEL_DIR}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {"id", "sentence", "target_output", "split"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use test split only
    df["split"] = df["split"].apply(clean_text)
    test_df = df[df["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("No rows found with split = test")

    print(f"Test rows to process: {len(test_df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading fine-tuned model from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))

    model.to(device)
    model.eval()

    sentences = [clean_text(x) for x in test_df["sentence"].tolist()]
    predictions = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sentences), BATCH_SIZE), desc="Running fine-tuned AREEj"):
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

    test_df["areej_finetuned_prediction"] = predictions

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nSample predictions:")
    for i in range(min(5, len(test_df))):
        print("\n---")
        print("Sentence:", clean_text(test_df.iloc[i]["sentence"]))
        print("Gold:", clean_text(test_df.iloc[i]["target_output"]))
        print("Fine-tuned:", clean_text(test_df.iloc[i]["areej_finetuned_prediction"]))


if __name__ == "__main__":
    main()