# Real-Time AML Detection

Pet project that demonstrates an end-to-end ML system for detecting suspicious financial transactions in real time.

The project is not limited to model training. It shows a production-like product loop: event generation, online scoring API, stateful features, Kafka ingestion, operational storage, analytical logs, monitoring, and traffic replay.

[Русская версия](README.md)

## Project Idea

Financial transactions arrive as a stream of events. For every transaction, the system should quickly:

- build features from the raw payload and customer history;
- estimate AML-risk probability;
- return a business verdict: `CLEAR`, `REVIEW`, or `BLOCK`;
- persist prediction, audit trail, and alerts;
- respond through an API or process the event asynchronously.

The main goal is to show how an ML model becomes a real-time product, not just a notebook experiment.

## What The Project Demonstrates

- **ML pipeline**: preprocessing, feature engineering, training, inference bundle.
- **Real-time serving**: FastAPI endpoint `POST /score`.
- **Streaming architecture**: Kafka worker and Spark Structured Streaming bridge.
- **Stateful features**: Redis 24-hour counters for customers and merchants.
- **Operational storage**: PostgreSQL as the operational source of truth.
- **Analytics storage**: ClickHouse for events, predictions, alerts, and dead-letter logs.
- **Monitoring**: Prometheus-compatible `/metrics`, runtime JSON logs, monitoring summary.
- **Local fallback**: SQLite mode for running without the full infrastructure stack.
- **Reproducibility**: serving through `models/inference_bundle.joblib`.

## Architecture

```text
Synthetic generator / Kafka
          |
          v
  Transaction event
          |
          v
  Redis online state ----+
          |              |
          v              |
  Feature pipeline <-----+
          |
          v
  Inference bundle
          |
          v
  Verdict policy
          |
          +--> FastAPI response
          +--> PostgreSQL predictions / alerts
          +--> ClickHouse analytical logs
          +--> Kafka predictions / alerts
          +--> Prometheus metrics
```

## Main User Flow

1. Generate a transaction or consume it from Kafka.
2. Enrich the event with online counters from Redis.
3. Run the event through the fitted feature pipeline and ensemble model.
4. Apply policy thresholds and return a business verdict.
5. Persist prediction, alert, and audit context.
6. Publish events to monitoring and analytical stores.

## API Capabilities

- `GET /health` - health check.
- `GET /ready` - inference bundle and storage backend readiness.
- `POST /score` - synchronous transaction scoring.
- `GET /events/{event_id}` - event details with prediction and audit context.
- `GET /alerts` - alert list.
- `POST /alerts/{alert_id}/resolution` - alert resolution.
- `GET /metrics` - Prometheus metrics.
- `GET /monitoring/summary` - aggregated monitoring summary.

## Measured Latency

`100` sequential events:

| Mode | Mean | p50 | p95 | p99 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct in-process scoring | `11.80 ms` | `11.43 ms` | `12.41 ms` | `19.22 ms` | `10.89 ms` | `32.77 ms` |
| HTTP `POST /score` | `14.55 ms` | `13.46 ms` | `20.04 ms` | `30.57 ms` | `11.64 ms` | `50.35 ms` |

Takeaway: the local synchronous scoring path is about `13.5 ms` p50 and `20.0 ms` p95 at the HTTP layer. The additional HTTP overhead versus a direct in-process call is roughly `2.8 ms` on average.

## Model Quality

Split sizes:

- train: `959,903` rows, `1,544` positives (`0.161%`)
- val: `83,918` rows, `108` positives (`0.129%`)
- test: `30,956` rows, `75` positives (`0.242%`)

Test split metrics:

| Metric | Value |
| --- | ---: |
| ROC-AUC | `0.9844` |
| PR-AUC | `0.9109` |
| F1 @ review threshold | `0.8408` |
| Precision @ review threshold | `0.8049` |
| Recall @ review threshold | `0.8800` |
| Review threshold | `0.65` |
| Block threshold | `0.90` |

Confusion matrix at `review threshold = 0.65`:

```text
TN=30865  FP=16
FN=9      TP=66
```

## How To Run

### Full Infrastructure Stack

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
docker compose up --build -d
```

Available services:

- API: `http://127.0.0.1:8000`
- MLflow: `http://127.0.0.1:5000`
- Kafka: `127.0.0.1:9092`
- PostgreSQL: `127.0.0.1:5432`
- ClickHouse HTTP: `127.0.0.1:8123`
- Redis: `127.0.0.1:6379`
- Spark master UI: `http://127.0.0.1:8080`

### Smoke Test

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

### Local API Without Infrastructure

```bash
AML_STORAGE_BACKEND=sqlite \
AML_ENABLE_CLICKHOUSE=0 \
AML_ENABLE_KAFKA=0 \
AML_ENABLE_REDIS=0 \
AML_ENABLE_MLFLOW=0 \
python -m uvicorn aml.api.app:app --host 127.0.0.1 --port 8000
```

### Generate And Replay Traffic

```bash
aml-generate --count 100
aml-replay --input runtime/generated_events.jsonl --mode api --url http://127.0.0.1:8000/score
```

## Model Training

```bash
aml-train
```

Main training outputs:

- `models/aml_ensemble/`
- `models/base_models_only/`
- `models/inference_bundle.joblib`

The serving path uses the fitted `inference_bundle.joblib`, because runtime scoring requires both the model and the preprocessing/feature pipeline state.

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

## Next Improvements

- move repository writes out of the hot path into an async/background writer;
- add batch inference for Kafka/Spark scenarios;
- profile the feature pipeline by stage and remove expensive per-row `pandas` operations;
- add load testing and a latency SLO dashboard;
- use model distillation for faster production serving;
- add a CI pipeline with smoke test, linting, and minimal API contract tests.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [ML System Design](docs/ML_SYSTEM_DESIGN.md)
- [API](docs/API.md)
- [Runbook](docs/RUNBOOK.md)
- [Monitoring](docs/MONITORING.md)
- [Retraining](docs/RETRAINING.md)
