# Runbook

## Primary startup path: Docker Compose

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Verify data is present

The product expects at least one of:

- `data/raw/AMLNet_August_2025.csv`
- `data/processed/train.parquet`

### 3. Start the infrastructure stack

```bash
docker compose up --build -d
```

This starts PostgreSQL, ClickHouse, Redis, Zookeeper, Kafka, MLflow, API, Kafka worker, and Spark services. Kafka topics are created by the `kafka-init` one-shot service.

### 4. Run a smoke test

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

This checks:

- bundle bootstrap
- synthetic generation
- runtime scoring
- audit persistence
- alert creation

## Starting the API

```bash
aml-serve
```

Default address:

- `http://127.0.0.1:8000`

Primary backend services:

- PostgreSQL: `localhost:5432`
- ClickHouse: `localhost:8123`
- Redis: `localhost:6379`
- Kafka: `localhost:9092`
- MLflow: `localhost:5000`
- Spark master UI: `localhost:8080`

## Generating and replaying traffic

Generate JSONL:

```bash
aml-generate --count 100
```

Generate directly to Kafka:

```bash
aml-generate --count 100 --sink kafka
```

Replay directly into the runtime:

```bash
aml-replay --input runtime/generated_events.jsonl --mode direct
```

Replay through the HTTP API:

```bash
aml-replay --input runtime/generated_events.jsonl --mode api --url http://127.0.0.1:8000/score
```

Replay into Kafka:

```bash
aml-replay --input runtime/generated_events.jsonl --mode kafka
```

Start Kafka worker:

```bash
aml-kafka-worker
```

Start Spark streaming bridge:

```bash
SPARK_MASTER=spark://localhost:7077 aml-spark-stream
```

## Operational files

- SQLite DB fallback: `runtime/aml_runtime.sqlite3`
- Runtime logs: `logs/aml_runtime.jsonl`
- Monitoring summary: `data/monitoring/latest_summary.json`

## Common checks

### Check health

```bash
curl http://127.0.0.1:8000/health
```

### Check readiness

```bash
curl http://127.0.0.1:8000/ready
```

### Check monitoring summary

```bash
curl http://127.0.0.1:8000/monitoring/summary
```

### Check metrics

```bash
curl http://127.0.0.1:8000/metrics
```

## Incident handling

If scoring fails:

1. Check `logs/aml_runtime.jsonl`
2. Inspect `dead_letter_events` in `runtime/aml_runtime.sqlite3`
3. Verify `models/inference_bundle.joblib` exists
4. Check PostgreSQL / ClickHouse / Kafka / Redis connectivity
5. Rebuild the bundle by rerunning `aml-train` or by deleting the bundle and letting runtime bootstrap it from legacy artifacts

## Rebuilding artifacts

Full train path:

```bash
AML_N_TRIALS=10 aml-train
```

Fast development path:

```bash
AML_SAMPLE_ROWS=100000 AML_N_TRIALS=2 aml-train
```

## Inspecting infrastructure data

PostgreSQL:

```bash
psql postgresql://aml:aml@localhost:5432/aml
```

Useful queries:

```sql
\dt
SELECT COUNT(*) FROM raw_events;
SELECT COUNT(*) FROM predictions;
SELECT COUNT(*) FROM alerts;
SELECT event_id, score, verdict, created_at FROM predictions ORDER BY created_at DESC LIMIT 10;
```

ClickHouse:

```bash
clickhouse-client --host localhost --port 9000 --query "SELECT count() FROM prediction_logs"
```

Useful queries:

```bash
clickhouse-client --host localhost --port 9000 --query "SHOW TABLES FROM aml"
clickhouse-client --host localhost --port 9000 --query "SELECT event_id, score, verdict, logged_at FROM aml.prediction_logs ORDER BY logged_at DESC LIMIT 10"
clickhouse-client --host localhost --port 9000 --query "SELECT logged_at, error_message FROM aml.dead_letter_logs ORDER BY logged_at DESC LIMIT 10"
```

Kafka topics:

```bash
docker exec -it $(docker ps -qf name=kafka) kafka-topics --bootstrap-server kafka:29092 --list
```

Read raw events:

```bash
docker exec -it $(docker ps -qf name=kafka) kafka-console-consumer --bootstrap-server kafka:29092 --topic transactions_raw --from-beginning --max-messages 5
```

Redis:

```bash
redis-cli -u redis://localhost:6379/0 KEYS 'aml:*'
```
