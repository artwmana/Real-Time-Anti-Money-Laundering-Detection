# Monitoring

## What is monitored

The product tracks both technical and business-facing runtime signals.

### Technical signals

- total scoring requests
- scoring errors
- average scoring latency
- Kafka prediction and alert topics
- ClickHouse event/prediction/alert log tables
- Redis-backed customer and merchant 24h counters

### Business signals

- verdict distribution
- number of open alerts
- dead-letter events
- average score

## Metrics endpoint

The API exposes Prometheus-compatible metrics at:

- `GET /metrics`

Current metrics:

- `aml_scoring_requests_total`
- `aml_scoring_errors_total`
- `aml_scoring_latency_seconds_avg`
- `aml_verdict_total{verdict="..."}`

## Monitoring summary

The API also exposes a repository-backed operational summary:

- `GET /monitoring/summary`

The summary includes:

- `total_events`
- `total_predictions`
- `open_alerts`
- `dead_letter_events`
- `average_score`
- `average_latency_ms`
- `p95_latency_ms`
- `verdict_counts`

## Persistent monitoring artifact

You can write the current monitoring snapshot to disk:

```bash
aml-monitor-report
```

This writes JSON to:

- `data/monitoring/latest_summary.json`

## Logs

Runtime logs are written as JSON lines to:

- `logs/aml_runtime.jsonl`

Each log record is structured for downstream parsing and contains:

- timestamp
- level
- logger
- message
- event_id where applicable
- verdict where applicable

## Audit data

Monitoring in this repository is intentionally tied to the operational store, so summary metrics can always be reconciled with actual persisted events.

The SQLite store keeps:

- raw events
- feature snapshots
- encoded features
- predictions
- alerts
- dead-letter events

## ClickHouse analytical tables

The infrastructure stack mirrors runtime events to ClickHouse:

- `event_logs`
- `prediction_logs`
- `alert_logs`
- `dead_letter_logs`

Example query:

```sql
SELECT verdict, count()
FROM prediction_logs
GROUP BY verdict
ORDER BY verdict;
```

## PostgreSQL operational tables

Primary runtime truth in infrastructure mode is stored in PostgreSQL:

- `raw_events`
- `feature_snapshots`
- `predictions`
- `alerts`
- `dead_letter_events`

## Kafka topics

- `transactions_raw`
- `aml_predictions`
- `aml_alerts`
- `aml_dead_letter`
