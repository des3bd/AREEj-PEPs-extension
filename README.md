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