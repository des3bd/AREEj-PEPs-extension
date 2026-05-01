# Extending AREEj for Arabic PEP Relation Extraction

This project extends **AREEj**, an Arabic evidence-aware relation extraction model, to a specialized **Politically Exposed Person (PEP)** relation extraction task.

The target relation is:

```text
person → position_held → office/role
```

Example:

```text
Sentence:
مبارك حمود سعدون الطشه نائب في مجلس الأمة الكويتي.

Target output:
<bor> مبارك حمود سعدون الطشه <per> نائب في مجلس الأمة الكويتي <concept> position held <rt> نائب في مجلس الأمة الكويتي <e>
```

---

## Project Overview

The original AREEj model was trained on broad Arabic relation extraction data. This project tests whether AREEj can be adapted to a narrower PEP-related task.


---


## Dataset

The original collection contained **1,414 JSON files**. After filtering, classification, annotation, and manual cleaning, the final dataset contained **197 relation instances**.

Each final row contains:

```text
id
sentence
subject
relation
object
evidence
target_output
split
```

The relation label used in this project is:

```text
position_held
```

The `target_output` column follows the AREEj-style linearized format:

```text
<bor> subject <per> object <concept> position held <rt> evidence <e>
```

---

## Requirements

Install the required packages:

```bash
pip install pandas torch transformers tqdm sentencepiece accelerate scikit-learn google-genai pydantic
```

The project uses the Hugging Face model:

```text
U4RASD/AREEj
```

Gemini API was used during dataset annotation. If rerunning the Gemini scripts, set your API key as an environment variable:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

---


## Baseline Experiment


Baseline test results:

| Metric | Score |
|---|---:|
| Subject match | 0.7667 |
| Object match | 0.7000 |
| Relation match | 0.4333 |
| Evidence exact/contains match | 0.7667 |

---

## Fine-Tuning


Fine-tuned test results:

| Metric | Score |
|---|---:|
| Subject match | 0.8000 |
| Object match | 0.7000 |
| Relation match | 0.8333 |
| Evidence exact/contains match | 0.8667 |

---

## Results Summary

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Subject match | 0.7667 | 0.8000 |
| Object match | 0.7000 | 0.7000 |
| Relation match | 0.4333 | 0.8333 |
| Evidence exact/contains match | 0.7667 | 0.8667 |


---

