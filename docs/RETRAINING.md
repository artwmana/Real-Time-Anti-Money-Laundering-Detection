# Retraining

## Current training entrypoint

```bash
aml-train
```

This command:

1. loads the raw AML dataset;
2. builds features using the existing `FeaturePipeline`;
3. applies a chronological split;
4. tunes and fits the AML ensemble;
5. saves model artifacts;
6. writes a serving-ready `inference_bundle.joblib`;
7. logs the run to MLflow when `MLFLOW_TRACKING_URI` is enabled.

## Key environment variables

- `AML_TARGET_COL`
- `AML_N_TRIALS`
- `AML_SAMPLE_ROWS`
- `DATA_PATH`
- `MODEL_PATH`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`

## Outputs

After a successful run, these artifacts should exist:

- `models/aml_ensemble/ensemble.joblib`
- `models/base_models_only/ensemble_deploy.joblib`
- `models/inference_bundle.joblib`

And, in infrastructure mode, the training run should be visible in MLflow.

## Why the inference bundle matters

The serving path needs more than just the fitted model.

It also needs:

- fitted preprocessing state;
- learned target encoding statistics;
- encoded feature names;
- verdict thresholds;
- schema metadata.

That is why the product serves from `inference_bundle.joblib` rather than only from the legacy deployment artifact.

## Fast development training

For local iteration:

```bash
AML_SAMPLE_ROWS=100000 AML_N_TRIALS=2 aml-train
```

## Fuller training run

For a more realistic local training pass:

```bash
AML_N_TRIALS=10 aml-train
```

## MLflow-enabled training

```bash
AML_STORAGE_BACKEND=postgres \
AML_POSTGRES_DSN=postgresql://aml:aml@localhost:5432/aml \
MLFLOW_TRACKING_URI=http://localhost:5000 \
MLFLOW_EXPERIMENT_NAME=aml-realtime \
aml-train
```

## Feedback loop status

The product API already supports alert resolution:

- `POST /alerts/{alert_id}/resolution`

These resolution records are the natural basis for the next retraining iteration.  
The next production-hardening step is to join resolved alerts back into the labeled training set and automate retraining windows.
