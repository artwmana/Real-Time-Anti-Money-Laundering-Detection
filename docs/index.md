# Real-Time AML Detection (Overview)

This repository now exposes an infrastructure-backed end-to-end AML product, not only offline ML experiments.

Implemented flow:

`synthetic or Kafka event -> Redis state -> feature engineering -> ensemble scoring -> business verdict -> PostgreSQL -> ClickHouse -> monitoring`

## Product components

- **Training pipeline** — Optuna-tuned AML ensemble and inference bundle generation.
- **Inference bundle** — fitted preprocessing plus model artifact for serving.
- **FastAPI scoring service** — sync scoring, alert resolution, audit retrieval, metrics.
- **Kafka worker** — async processing from `transactions_raw`.
- **Spark bridge** — Spark Structured Streaming path from Kafka to the scoring API.
- **Redis feature store** — 24h counters for customer and merchant activity.
- **Operational store** — PostgreSQL for raw events, feature snapshots, predictions, alerts, DLQ.
- **Analytical/log sink** — ClickHouse for event, prediction, alert, and dead-letter logs.
- **MLflow** — training run tracking and artifact logging.
- **Monitoring** — Prometheus-compatible metrics and persisted monitoring summary.

## Quick links

- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Dataset: [DATASET.md](DATASET.md)
- ML system design: [ML_SYSTEM_DESIGN.md](ML_SYSTEM_DESIGN.md)
- API: [API.md](API.md)
- Runbook: [RUNBOOK.md](RUNBOOK.md)
- Monitoring: [MONITORING.md](MONITORING.md)
- Retraining: [RETRAINING.md](RETRAINING.md)
- E2E build roadmap: [BUILD_E2E_ROADMAP.md](BUILD_E2E_ROADMAP.md)

## Fast local validation

```bash
python -m aml.runtime.smoke_test
```

This command generates events, scores them, writes runtime data to the configured backend, and prints a monitoring summary.

## Main commands

- `aml-train`
- `aml-serve`
- `aml-generate --count 100`
- `aml-generate --count 100 --sink kafka`
- `aml-replay --count 100 --mode direct`
- `aml-kafka-worker`
- `aml-spark-stream`
- `aml-monitor-report`
- `python -m aml.runtime.smoke_test`
