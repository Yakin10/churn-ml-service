# Customer Churn Prediction — ML Service

An end-to-end machine learning project that predicts customer churn using a reproducible scikit-learn pipeline.

## Problem Statement

Customer churn occurs when customers stop using a service. This project explores how customer attributes and subscription details can be used to identify customers who may be at risk of leaving.

The model is a learning and portfolio project, not a validated production decision system.

## Dataset

Telco Customer Churn dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The commonly distributed dataset contains 7,043 customer records and 21 columns. The target is `Churn`.

NOTE: The raw dataset is not committed to this repository.

## Approach

- Convert `TotalCharges` to numeric and handle missing values.
- Exclude the customer identifier and target from model features.
- Split data into training and test sets using stratification.
- Impute missing numeric and categorical values.
- Scale numeric features and one-hot encode categorical features.
- Train a logistic regression baseline.
- Evaluate predictions using accuracy, precision, recall, F1 and ROC-AUC.

Preprocessing and model training are combined in a scikit-learn Pipeline to ensure transformations are learned from training data.

## Baseline Results

| Metric | Score |
|---|---:|
| Accuracy | 0.806 |
| Churn precision | 0.657 |
| Churn recall | 0.559 |
| Churn F1 | 0.604 |
| ROC-AUC | 0.727 |

These results use a stratified 80/20 train-test split with random_state=42. Results may vary with dataset or library changes.

## Project Structure

    data/                  Local dataset files
    models/                Saved model artifacts
    notebooks/             Exploratory analysis
    src/churn_ml/          Reusable ML code
    tests/                 Automated tests
    requirements.txt       Python dependencies
    README.md              Project documentation

## Setup

Requires a compatible Python version and Git.

Create and activate a virtual environment on Windows:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

Download the dataset from Kaggle and place the CSV at:

    data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Run training and evaluation from the repository root:

    python src/churn_ml/train.py

Run tests:

    python -m pytest tests/ -v

## Current Status

- [x] Data loading and basic cleaning
- [x] Preprocessing and model training pipeline
- [x] Baseline evaluation
- [x] Automated pipeline test
- [x] Model artifact persistence
- [x] Prediction API
- [x] Docker packaging
- [x] Continuous integration
- [ ] Deployment and monitoring

## Limitations

The current model is a baseline. Its performance has not been
validated for real-world customer retention decisions. Further work
is needed on threshold selection, model comparisons, monitoring,
and production safeguards.