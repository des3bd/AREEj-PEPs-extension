import re
import pandas as pd
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


# =========================
# CONFIG
# =========================

INPUT_CSV = Path("data/final/pep_areej_final_dataset.csv")
MODEL_NAME = "U4RASD/AREEj"

OUTPUT_DIR = Path("models/areej_pep_finetuned")

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 160

NUM_EPOCHS = 15
LEARNING_RATE = 3e-5

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

RANDOM_SEED = 42


# =========================
# HELPERS
# =========================

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def load_split_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {"sentence", "target_output", "split"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["sentence"] = df["sentence"].apply(clean_text)
    df["target_output"] = df["target_output"].apply(clean_text)
    df["split"] = df["split"].apply(clean_text)

    df = df[
        (df["sentence"] != "")
        & (df["target_output"] != "")
        & (df["split"] != "")
    ].copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"].isin(["validation", "val"])].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty:
        raise ValueError("Train split is empty.")

    if val_df.empty:
        raise ValueError("Validation split is empty.")

    print("Dataset loaded.")
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")

    return train_df, val_df, test_df


class PepRelationDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer):
        self.inputs = dataframe["sentence"].tolist()
        self.targets = dataframe["target_output"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        source_text = self.inputs[idx]
        target_text = self.targets[idx]

        model_inputs = self.tokenizer(
            source_text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )

        labels = self.tokenizer(
            text_target=target_text,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


def main():
    torch.manual_seed(RANDOM_SEED)

    train_df, val_df, _ = load_split_data()

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_dataset = PepRelationDataset(train_df, tokenizer)
    val_dataset = PepRelationDataset(val_df, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    use_fp16 = torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,

        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=use_fp16,

        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Saving final model...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("Done.")
    print(f"Fine-tuned model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()