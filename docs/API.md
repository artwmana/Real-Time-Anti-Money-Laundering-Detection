# API

## Endpoints

### `GET /health`

Returns a simple liveness check.

Example response:

```json
{"status": "ok"}
```

### `GET /ready`

Checks whether the inference bundle is available.

### `POST /score`

Scores one transaction event and returns the business verdict.

Example request:

```json
{
  "type": "TRANSFER",
  "amount": 9500.0,
  "category": "Transfer",
  "nameOrig": "C1001",
  "nameDest": "C2002",
  "oldbalanceOrg": 12000.0,
  "newbalanceOrig": 2500.0,
  "metadata": {
    "timestamp": "2026-04-11T12:00:00+00:00",
    "location": {"city": "Warsaw", "state": "Mazowieckie", "country": "PL", "postcode": "00-001"},
    "device_info": {"type": "Mobile", "os": "Android", "ip_address": "10.10.10.10"},
    "payment_method": "CardNumber",
    "risk_indicators": {
      "amount_vs_average": 8.2,
      "customer_risk_score": 92,
      "category_risk": "high",
      "risk_score": 88,
      "unusual_time": true,
      "unusual_location": true
    }
  }
}
```

Example response:

```json
{
  "event_id": "7f8d9d9f-6f52-4c07-8a4c-24f5ef1f4d02",
  "score": 0.72,
  "verdict": "REVIEW",
  "risk_band": "medium",
  "model_version": "aml_ensemble_v1",
  "policy_version": "aml_policy_v1",
  "threshold_review": 0.65,
  "threshold_block": 0.9,
  "reason_codes": ["high_risk_indicator", "unusual_location"],
  "processing_latency_ms": 215.3
}
```

### `GET /events/{event_id}`

Returns the full audit bundle for an event:

- raw payload
- feature snapshot
- encoded features
- prediction
- linked alerts

### `GET /alerts`

Returns current alerts. Optional query parameter:

- `status`

### `POST /alerts/{alert_id}/resolution`

Closes or resolves an alert.

Example request:

```json
{
  "analyst_id": "analyst-01",
  "resolution": "TRUE_POSITIVE",
  "analyst_comment": "Confirmed suspicious cash-out pattern"
}
```

### `GET /monitoring/summary`

Returns an aggregate operational summary from the active operational backend:

- PostgreSQL in infrastructure mode
- SQLite in local fallback mode

### `GET /metrics`

Returns Prometheus-compatible text metrics for request counts, errors, verdict distribution, and average latency.
