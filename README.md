# Extending AREEj for Specialized Arabic PEP-Relation Extraction

This repository contains the implementation files for a preliminary extension of AREEj to specialized Arabic PEP-related relation extraction.

## Project Description

The project evaluates whether AREEj can be applied without fine-tuning to a smaller Arabic PEP-relation schema. The target relations include:

- person -> holds_position -> office
- office -> belongs_to -> government/international organization
- person -> member_of / heads / serves_in -> state body


## Project Structure

```text
data/
  draft_pep_test_dataset_50.csv

code/
  run_areej_inference.py
  evaluate_predictions.py

results/
  areej_predictions.csv
  preliminary_results.csv
```

## How to Run

### Prerequisites

- Python 3.8 or higher
- An internet connection (to download the AREEj model from Hugging Face on first run)
- A GPU is optional but recommended for faster inference

### 1. Install Dependencies

From the repository root, install the required packages:

```bash
pip install -r requirments.txt
```

### 2. Run Inference

This script loads the AREEj model, runs it on the test dataset, and saves the raw predictions to `results/areej_predictions.csv`.

```bash
python code/run_areej_inference.py
```

### 3. Evaluate Predictions

After inference completes, run the evaluation script to compute relation and evidence accuracy. Results are saved to `results/preliminary_results.csv`.

```bash
python code/evaluate_predictions.py
```