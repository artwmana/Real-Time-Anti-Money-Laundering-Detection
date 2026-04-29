# Real-Time AML Detection

A pet project of an end-to-end ML system for real-time detection of suspicious financial transactions — covering model training, event generation, online scoring API, stateful features, Kafka ingestion, operational storage, analytical logging, monitoring, and traffic replay.

[Russian version](README.md)

## Problem

Financial transactions arrive as a stream of events. For each transaction, the system must quickly:

* extract features from the raw payload and the client’s historical state;
* compute the AML risk probability;
* make a business decision: `CLEAR`, `BLOCK`, or `REVIEW`;
* persist the result, audit trail, and alerts;
* return a response via API or process the event asynchronously.

## What This Project Demonstrates

* **ML pipeline**: preprocessing, feature engineering, training, inference bundle.
* **Real-time serving**: FastAPI endpoint `POST /score`.
* **Streaming architecture**: Kafka worker and Spark Structured Streaming bridge.
* **Stateful features**: Redis counters for customers and merchants over a 24-hour window.
* **Operational storage**: PostgreSQL as the system of operational truth.
* **Analytics storage**: ClickHouse for events, predictions, alerts, and dead-letter logs.
* **Monitoring**: Prometheus-compatible `/metrics`, runtime JSON logs, monitoring summary.
* **Local fallback**: SQLite mode for running without infrastructure.
* **Reproducibility**: serving via `models/inference_bundle.joblib`.

## Main Flow

1. Generate or receive a transaction from Kafka.
2. Enrich the event with online counters from Redis.
3. Pass the event through the fitted feature pipeline and ensemble model.
4. Apply policy thresholds and produce a business verdict.
5. Persist prediction, alert, and audit context.
6. Send events to monitoring and analytical storage.

## API

* `GET /health` — health check
* `GET /ready` — readiness check for bundle and storage backend
* `POST /score` — synchronous transaction scoring
* `GET /events/{event_id}` — retrieve event with prediction/audit context
* `GET /alerts` — list alerts
* `POST /alerts/{alert_id}/resolution` — resolve an alert
* `GET /metrics` — Prometheus metrics
* `GET /monitoring/summary` — aggregated monitoring summary

## Latency

| Mode                       |   Average |       p50 |       p95 |       min |       max |
| -------------------------- | --------: | --------: | --------: | --------: | --------: |
| Docker end-to-end scoring  | `21.7 ms` | `22.0 ms` | `25.5 ms` | `18.3 ms` | `25.9 ms` |
| Inference only             | `10.9 ms` | `10.7 ms` | `12.4 ms` |         - |         - |
| Model `predict_proba` only |  `2.4 ms` |  `2.4 ms` |  `3.2 ms` |  `1.8 ms` |  `4.4 ms` |

## Run

### Full Infrastructure Stack

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
docker compose up --build -d
```

* API: `http://127.0.0.1:8000`
* MLflow: `http://127.0.0.1:5000`
* Kafka: `127.0.0.1:9092`
* PostgreSQL: `127.0.0.1:5432`
* ClickHouse HTTP: `127.0.0.1:8123`
* Redis: `127.0.0.1:6379`
* Spark master UI: `http://127.0.0.1:8080`

### Tests

```bash
AML_STORAGE_BACKEND=postgres \
AML_POSTGRES_DSN=postgresql://aml:aml@localhost:5432/aml \
AML_CLICKHOUSE_HOST=localhost \
AML_CLICKHOUSE_PORT=8123 \
AML_REDIS_URL=redis://localhost:6379/0 \
AML_KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
MLFLOW_TRACKING_URI=http://localhost:5000 \
python -m aml.runtime.smoke_test
```

### Local API Run

```bash
AML_STORAGE_BACKEND=sqlite \
AML_ENABLE_CLICKHOUSE=0 \
AML_ENABLE_KAFKA=0 \
AML_ENABLE_REDIS=0 \
AML_ENABLE_MLFLOW=0 \
python -m uvicorn aml.api.app:app --host 127.0.0.1 --port 8000
```

### Traffic Generation and Replay

```bash
aml-generate --count 100
aml-replay --input runtime/generated_events.jsonl --mode api --url http://127.0.0.1:8000/score
```

## Model Training

```bash
aml-train
```

Main training outputs:

* `models/aml_ensemble/`
* `models/base_models_only/`
* `models/inference_bundle.joblib`

The serving path uses the fitted `inference_bundle.joblib`, since correct runtime scoring requires not only the model but also the preprocessing and feature pipeline state.

## Project Structure

```text
src/aml/api              FastAPI application
src/aml/application      use cases and verdict policy
src/aml/config           runtime settings
src/aml/contracts        event and response schemas
src/aml/generation       synthetic transaction generator
src/aml/inference        inference service and bundle
src/aml/infrastructure   Kafka, Redis, ClickHouse, MLflow adapters
src/aml/models           ensemble model code
src/aml/monitoring       metrics and JSON logging
src/aml/pipelines        feature engineering pipeline
src/aml/runtime          bootstrap, replay, smoke test, workers
src/aml/storage          PostgreSQL, SQLite and composite repositories
src/aml/training         training entrypoints
```

## Documentation

* [Architecture](docs/ARCHITECTURE.md)
* [ML System Design](docs/ML_SYSTEM_DESIGN.md)
* [API](docs/API.md)
* [Runbook](docs/RUNBOOK.md)
* [Monitoring](docs/MONITORING.md)
* [Retraining](docs/RETRAINING.md)
