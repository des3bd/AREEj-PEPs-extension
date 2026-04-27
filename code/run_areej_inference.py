import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pathlib import Path

DATA_PATH = Path("data/draft_pep_test_dataset_50.csv")
OUTPUT_PATH = Path("results/areej_predictions.csv")

MODEL_NAME = "U4RASD/AREEj"

df = pd.read_csv(DATA_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading AREEj model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def build_prompt(row):
    return (
        "استخرج العلاقات من النص التالي مع الدليل:\n"
        f"{row['sentence']}"
    )

predictions = []

for index, row in df.iterrows():
    prompt = build_prompt(row)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            max_length=None
        )

    prediction_text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    predictions.append(prediction_text)

    print(f"Processed {index + 1}/{len(df)}")

df["prediction_raw"] = predictions

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"Saved predictions to: {OUTPUT_PATH}")