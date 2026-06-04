[README (1).md](https://github.com/user-attachments/files/28600416/README.1.md)

# Telco Customer Churn Prediction

Predict whether a telecom customer will churn (leave the service) based on account
demographics, subscribed services, and billing information. The trained
**Random Forest** classifier is the final model — it can be reloaded from a
pickle file and used to score new customers.

The full pipeline lives in
`0e6d708d__d391edbf-441d-42c2-9516-59e7be6ed31e.ipynb`.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Data Loading & Cleaning](#1-data-loading--cleaning)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Handling Class Imbalance](#4-handling-class-imbalance)
  - [5. Model Training & Selection](#5-model-training--selection)
  - [6. Model Evaluation](#6-model-evaluation)
  - [7. Inference on New Data](#7-inference-on-new-data)
- [Results](#results)
- [Saved Artifacts](#saved-artifacts)
- [Possible Improvements](#possible-improvements)
- [License](#license)

---

## Overview

Customer churn is a top concern for subscription-based businesses. This project
builds a binary classifier that flags customers who are likely to churn so that
retention teams can act on them. The notebook covers the full lifecycle —
cleaning, EDA, preprocessing, training, evaluation, and a small inference demo.

## Dataset

- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Shape:** 7,043 rows × 21 columns (20 features + target)
- **Target:** `Churn` — `Yes` / `No` (re-encoded to `1` / `0`)
- **Class distribution:** 5,174 retained vs. 1,869 churned (≈ 73 / 27 split — imbalanced)
- **Feature mix:**
  - Numerical: `tenure`, `MonthlyCharges`, `TotalCharges`
  - Categorical: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`,
    `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
    `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`,
    `PaperlessBilling`, `PaymentMethod`

> The original notebook was authored in Google Colab and reads the CSV from
> `/content/`. Update the path in **Cell 3** before running locally.

## Project Structure

```
.
├── 0e6d708d__d391edbf-441d-42c2-9516-59e7be6ed31e.ipynb   # Main notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv                    # Dataset (download separately)
├── encoders.pkl                                            # Saved LabelEncoders (generated)
├── customer_churn_model.pkl                                # Trained RandomForest + feature list (generated)
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

Launch Jupyter and open the notebook:

```bash
jupyter notebook
```

## Pipeline Walkthrough

### 1. Data Loading & Cleaning
- Loaded the CSV with pandas.
- Dropped `customerID` (identifier, not a feature).
- `TotalCharges` had 11 rows containing blank strings → replaced with `"0.0"`
  and cast to `float`. All other columns are non-null.

### 2. Exploratory Data Analysis
- `df.describe()` for numerical summaries.
- Distribution + box plots for `tenure`, `MonthlyCharges`, `TotalCharges`.
- Correlation heatmap across the three numerical features.
- Count plots for every categorical column.
- Confirmed class imbalance in `Churn`.

### 3. Data Preprocessing
- Replaced target values: `Yes → 1`, `No → 0`.
- `LabelEncoder` fit on every object-dtype column; each fitted encoder is
  persisted in a dict and dumped to **`encoders.pkl`** so the same mapping can
  be applied at inference time.

### 4. Handling Class Imbalance
- Split 80 / 20 (`random_state=42`).
- Applied **SMOTE** (`imblearn.over_sampling.SMOTE`) on the training set only,
  balancing the minority class so the model doesn't bias toward "No Churn".

### 5. Model Training & Selection
Trained three classifiers with default hyperparameters and ran **5-fold
cross-validation** on the resampled training set:

| Model         | Mean CV Accuracy |
|---------------|------------------|
| Decision Tree | ~0.78            |
| Random Forest | **~0.85**        |
| XGBoost       | ~0.84            |

Random Forest won, so it was retrained on the full SMOTE-augmented training
set and used for evaluation + inference.

### 6. Model Evaluation
Evaluated the final Random Forest on the held-out test set:

- **Accuracy**
- **Confusion matrix**
- **Precision / Recall / F1** via `classification_report`

The trained model and the list of feature names are saved to
**`customer_churn_model.pkl`**.

### 7. Inference on New Data
A demo cell constructs a single-customer dictionary, wraps it in a
DataFrame, applies the saved encoders, and calls
`model.predict` / `model.predict_proba` to return:

```text
Prediction: Churn / No Churn
Prediction Probability: [P(No Churn), P(Churn)]
```

## Results

Random Forest is the best performer. From the notebook:

- **Cross-validation accuracy (5-fold on SMOTE data):** ~0.85
- **Test set accuracy:** reported inside Cell 79 — the confusion matrix shows
  the model leans slightly better on the majority class, which is the
  expected trade-off without hyperparameter tuning.

(See the notebook's printed outputs in Cells 72 and 79 for exact numbers.)

## Saved Artifacts

| File                       | Contents                                            |
|----------------------------|-----------------------------------------------------|
| `encoders.pkl`             | Dict of `column → fitted LabelEncoder`              |
| `customer_churn_model.pkl` | Dict: `{"model": RandomForest, "features_names": []}` |

Example inference script:

```python
import pickle, pandas as pd

with open("customer_churn_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model, feature_names = bundle["model"], bundle["features_names"]

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

sample = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}

df = pd.DataFrame([sample])
for col, enc in encoders.items():
    df[col] = enc.transform(df[col])

print("Churn" if model.predict(df)[0] == 1 else "No Churn")
print("Probabilities:", model.predict_proba(df))
```

## Possible Improvements

The notebook's own "To do" list (Cell 87):

1. Hyperparameter tuning (`GridSearchCV` / `Optuna`).
2. Better model selection / ensembling.
3. Try **downsampling** the majority class as an alternative to SMOTE.
4. Reduce overfitting (limit depth, regularization, more trees with lower variance).
5. **Stratified k-fold CV** to preserve the class ratio per fold.

Other ideas worth trying:

- One-hot encoding for nominal features (avoid imposing an ordinal meaning).
- Feature engineering: `tenure × MonthlyCharges`, service count, average monthly change.
- Calibration curves and threshold tuning to favor recall on the churn class.
- Persist preprocessing in a `Pipeline` / `ColumnTransformer` so train and
  inference never drift.

## License

This project is for educational purposes. The Telco Customer Churn dataset is
publicly available from IBM / Kaggle — please follow the original source's
licensing terms.
