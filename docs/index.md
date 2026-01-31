# Real-Time AML Detection (Overview)

This project builds a real-time anti–money-laundering (AML) detector for transaction streams.  
It combines a reusable feature pipeline, class-imbalance aware training, and a design ready for streaming ingestion.

## Core components
- **Feature pipeline** (pandas + sklearn) — flattens JSON metadata, builds time features, target-encodes high-cardinality IDs.
- **PySpark trainer** — gradient-boosted trees with `weightCol` to fight the extreme class imbalance in AML labels.
- **Config-first data contract** — schema, paths, and feature groups live in `configs/dataset.yaml`.
- **Docs** — architecture, dataset notes, and ML system design for productionizing the model.

## Quick links
- Architecture: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- Dataset: [docs/DATASET.md](DATASET.md)
- ML system design: [docs/ML_SYSTEM_DESIGN.md](ML_SYSTEM_DESIGN.md)
- PySpark training script: `src/aml/pipelines/spark_training.py`

## Local try-out
1. Install: `pip install -e .` (optional: `pip install -e .[dev]`)
2. Put data under `data/raw/` or parquet splits under `data/processed/`.
3. Run sklearn demo: `python main.py`
4. Train with PySpark + class weights: `python -m aml.pipelines.spark_training`

## Why class weighting
Money-laundering labels are tiny (~0.1%). The PySpark trainer builds inverse-frequency class weights so the model sees positives more often without oversampling. This keeps inference fast and avoids synthetic examples that might leak into production logic.
