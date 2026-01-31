# ML System Design

This page sketches how to run the AML detector as a reliable, low-latency service while keeping training reproducible and traceable.

## Goals and constraints
- **Low latency**: <150 ms per transaction for online scoring.
- **High precision first**: regulators prefer fewer false negatives; we bias toward precision and alert triage.
- **Extreme imbalance**: ~0.1% positives; handled via class weights and threshold tuning.
- **Traceability**: every decision should point to model version, feature payload, and upstream event ID.

## Data flow
```
Payment Gateway -> Kafka topic `transactions_raw`
                 -> Spark Structured Streaming (feature prep)
                 -> Feature Store (historical aggregates)
                 -> Online Model (PySpark/MLlib or exported to ONNX)
                 -> Alerts topic `aml_alerts`
                 -> Case Management / Investigator UI
```

### Offline training
1. **Ingest**: Land CSV/parquet into `data/raw` or `data/processed`.
2. **Feature build**: `FeaturePipeline` (pandas) creates time signals, ratios, and target encodings for IDs.
3. **Class imbalance**: PySpark trainer adds inverse-frequency `class_weight` and fits GBT with `weightCol`.
4. **Validation**: PR-AUC and ROC-AUC on a holdout split; pick operating threshold by desired precision/recall.
5. **Registry**: Save model, metrics.json, and feature schema to an artifact store (e.g., MLflow).

### Online inference
- **Ingestion**: Spark Structured Streaming reads from Kafka, parses JSON, and applies lightweight feature prep (time fields, ratios, known encodings).
- **Feature parity**: keep the same logic as offline by exporting encoder params (buckets, target encodings). Store them alongside the model artifact.
- **Scoring**: GBTClassifier (Spark) or a converted model served via FastAPI if using a microservice pattern.
- **Thresholding**: configurable per jurisdiction; default from offline calibration.
- **Outputs**: alert payload includes score, top contributing features, and references to model version + event ID.

## Reliability and monitoring
- **Drift**: Evidently or custom Spark jobs track population drift and label delay; alert when PSI/JS divergence exceeds set bounds.
- **Data quality**: enforce schema checks (missing timestamps, invalid amounts) before scoring; route failures to a dead-letter queue.
- **Throughput**: size shuffle partitions based on ingestion volume; autoscale executors on backpressure.
- **Alert budget**: cap alert rate per merchant/region to prevent investigator overload.

## Extending the model
- Try other classifiers with `weightCol` (LogisticRegression, RandomForest) inside `spark_training.py`.
- Add rolling aggregates (e.g., 24h txn count per `nameOrig`) via Spark window functions before the model stage.
- Export to ONNX/PMML if you need to serve on a non-Spark stack; keep the same feature contract.
