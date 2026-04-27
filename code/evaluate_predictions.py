import pandas as pd
from pathlib import Path

PRED_PATH = Path("results/areej_predictions.csv")
RESULTS_PATH = Path("results/preliminary_results.csv")

df = pd.read_csv(PRED_PATH)

def extract_relation_match(row):
    gold = str(row["relation"]).strip()
    pred = str(row["prediction_raw"]).strip()
    return 1 if gold in pred else 0

def extract_evidence_match(row):
    gold = str(row["evidence_span"]).strip()
    pred = str(row["prediction_raw"]).strip()
    return 1 if gold in pred else 0

df["relation_correct"] = df.apply(extract_relation_match, axis=1)
df["evidence_correct"] = df.apply(extract_evidence_match, axis=1)

relation_accuracy = df["relation_correct"].mean()
evidence_accuracy = df["evidence_correct"].mean()

results = pd.DataFrame([
    {
        "model": "AREEj without fine-tuning",
        "test_size": len(df),
        "relation_accuracy": round(relation_accuracy, 3),
        "evidence_match_accuracy": round(evidence_accuracy, 3),
        "notes": "Preliminary no-fine-tuning evaluation on draft Arabic PEP test set"
    }
])

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

print(results)
print(f"Saved results to: {RESULTS_PATH}")