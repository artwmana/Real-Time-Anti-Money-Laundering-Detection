# Real-Time AML Detection

This repo trains and serves a model that flags risky money‑laundering transactions.  
It works with batch files today and is ready to plug into a streaming source later.

## What’s inside
- Feature pipeline that flattens JSON metadata, builds time features, and handles high-cardinality IDs.
- Two training options:
  - Scikit-learn (XGBoost/LightGBM/CatBoost) for quick local experiments.
  - PySpark Gradient Boosted Trees with class weights for very imbalanced data.
- MkDocs docs with data description and ML system design notes.

## Quick start (local CPU)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs core deps
pip install -e .[dev]       # optional: notebooks/linting
```

Place your data
```
data/raw/AMLNet_August_2025.csv      # or parquet splits in data/processed/
```
Target column is `isMoneyLaundering` (1 = suspicious, 0 = normal).

### Run the sklearn demo
```bash
python main.py
```
It logs pipeline steps to `logs/feature_pipeline.log`.

### Train the PySpark model (handles heavy imbalance)
```bash
python -m aml.pipelines.spark_training
```
The script:
1) Loads splits from `configs/dataset.yaml` (falls back to `data/raw` if parquet splits are absent).
2) Builds class weights from label counts.
3) Fits a GBT classifier with `weightCol`.
4) Saves the model and `metrics.json` to `models/spark_gbt_aml/`.

## Data expectation
- Schema and feature lists live in `configs/dataset.yaml` and `docs/DATASET.md`.
- Important columns: amounts/balances, customer/merchant/device IDs, derived time features.
- Leakage columns (`fraud_probability`, `laundering_typology`, etc.) are dropped before training.

## How to extend
- Add new engineered features in `src/aml/pipelines/feature_pipeline.py` under `_feature_engineering`.
- Tune PySpark GBT hyperparameters in `src/aml/pipelines/spark_training.py`.
- Swap in another classifier that supports `weightCol` if you prefer (e.g., LogisticRegression).

## Repo layout
- `src/aml/features` – flattening, downcasting, target encoding.
- `src/aml/pipelines` – sklearn pipeline + PySpark trainer.
- `notebooks/` – EDA and quick experiments.
- `docs/` – MkDocs site (architecture, dataset, ML system design).
- `configs/` – dataset paths, schema, feature groups.

## Minimal checklist before using in prod
- Recompute class weights on your own data.
- Track experiments in MLflow (hooks are ready to add).
- Calibrate threshold on precision/recall for your regulator policy.
- Add monitoring for class drift and prediction drift.
