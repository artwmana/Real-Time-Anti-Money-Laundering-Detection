from __future__ import annotations

import json
from datetime import UTC, datetime
from statistics import quantiles
from typing import Any
from uuid import uuid4

from aml.contracts.runtime import AlertResolution, AlertView, MonitoringSummary, TransactionEvent


def _serialize(payload: Any) -> str:
    def _default(value: Any):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    return json.dumps(payload, ensure_ascii=False, default=_default)


class PostgresRepository:
    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("psycopg package is required for PostgresRepository") from exc

        self.psycopg = psycopg
        self.dict_row = dict_row
        self.dsn = dsn
        self._initialize()

    def _connect(self):
        return self.psycopg.connect(self.dsn, row_factory=self.dict_row)

    def _initialize(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS raw_events (
                        event_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        ingested_at TIMESTAMPTZ NOT NULL,
                        payload_json JSONB NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS feature_snapshots (
                        event_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        feature_json JSONB NOT NULL,
                        encoded_json JSONB NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS predictions (
                        event_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        score DOUBLE PRECISION NOT NULL,
                        verdict TEXT NOT NULL,
                        risk_band TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        policy_version TEXT NOT NULL,
                        threshold_review DOUBLE PRECISION NOT NULL,
                        threshold_block DOUBLE PRECISION NOT NULL,
                        processing_latency_ms DOUBLE PRECISION NOT NULL,
                        reason_codes_json JSONB NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL REFERENCES predictions(event_id),
                        created_at TIMESTAMPTZ NOT NULL,
                        status TEXT NOT NULL,
                        verdict TEXT NOT NULL,
                        score DOUBLE PRECISION NOT NULL,
                        reason_codes_json JSONB NOT NULL,
                        analyst_id TEXT,
                        analyst_resolution TEXT,
                        analyst_comment TEXT,
                        resolved_at TIMESTAMPTZ
                    );

                    CREATE TABLE IF NOT EXISTS dead_letter_events (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        payload_json JSONB NOT NULL,
                        error_message TEXT NOT NULL
                    );
                    """
                )
            conn.commit()

    def save_raw_event(self, event: TransactionEvent) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO raw_events(event_id, source, ingested_at, payload_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (event_id) DO UPDATE
                    SET source = EXCLUDED.source,
                        ingested_at = EXCLUDED.ingested_at,
                        payload_json = EXCLUDED.payload_json
                    """,
                    (event.event_id, event.source, event.ingested_at, _serialize(event.model_dump(mode="json"))),
                )
            conn.commit()

    def save_feature_snapshot(self, event_id: str, feature_snapshot: dict[str, Any], encoded_features: dict[str, Any]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feature_snapshots(event_id, created_at, feature_json, encoded_json)
                    VALUES (%s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (event_id) DO UPDATE
                    SET created_at = EXCLUDED.created_at,
                        feature_json = EXCLUDED.feature_json,
                        encoded_json = EXCLUDED.encoded_json
                    """,
                    (event_id, datetime.now(UTC), _serialize(feature_snapshot), _serialize(encoded_features)),
                )
            conn.commit()

    def save_prediction(self, event: TransactionEvent, outcome: Any) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions(
                        event_id, created_at, score, verdict, risk_band, model_version, policy_version,
                        threshold_review, threshold_block, processing_latency_ms, reason_codes_json
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (event_id) DO UPDATE
                    SET created_at = EXCLUDED.created_at,
                        score = EXCLUDED.score,
                        verdict = EXCLUDED.verdict,
                        risk_band = EXCLUDED.risk_band,
                        model_version = EXCLUDED.model_version,
                        policy_version = EXCLUDED.policy_version,
                        threshold_review = EXCLUDED.threshold_review,
                        threshold_block = EXCLUDED.threshold_block,
                        processing_latency_ms = EXCLUDED.processing_latency_ms,
                        reason_codes_json = EXCLUDED.reason_codes_json
                    """,
                    (
                        event.event_id,
                        datetime.now(UTC),
                        outcome.score,
                        outcome.verdict,
                        outcome.risk_band,
                        outcome.model_version,
                        outcome.policy_version,
                        outcome.threshold_review,
                        outcome.threshold_block,
                        outcome.processing_latency_ms,
                        _serialize(outcome.reason_codes),
                    ),
                )
            conn.commit()

    def save_alert(self, event: TransactionEvent, outcome: Any) -> str:
        alert_id = str(uuid4())
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO alerts(
                        alert_id, event_id, created_at, status, verdict, score, reason_codes_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        alert_id,
                        event.event_id,
                        datetime.now(UTC),
                        "OPEN",
                        outcome.verdict,
                        outcome.score,
                        _serialize(outcome.reason_codes),
                    ),
                )
            conn.commit()
        return alert_id

    def save_dlq(self, payload: dict[str, Any], error_message: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dead_letter_events(created_at, payload_json, error_message)
                    VALUES (%s, %s::jsonb, %s)
                    """,
                    (datetime.now(UTC), _serialize(payload), error_message),
                )
            conn.commit()

    def fetch_event_bundle(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM raw_events WHERE event_id = %s", (event_id,))
                raw = cur.fetchone()
                cur.execute("SELECT * FROM predictions WHERE event_id = %s", (event_id,))
                pred = cur.fetchone()
                cur.execute("SELECT * FROM feature_snapshots WHERE event_id = %s", (event_id,))
                feat = cur.fetchone()
                cur.execute("SELECT * FROM alerts WHERE event_id = %s ORDER BY created_at DESC", (event_id,))
                alerts = cur.fetchall()

        if raw is None and pred is None:
            return None

        prediction_payload = None
        if pred:
            prediction_payload = dict(pred)
            prediction_payload["reason_codes"] = pred["reason_codes_json"]

        return {
            "event_id": event_id,
            "raw_event": raw["payload_json"] if raw else None,
            "prediction": prediction_payload,
            "feature_snapshot": feat["feature_json"] if feat else None,
            "encoded_features": feat["encoded_json"] if feat else None,
            "alerts": [
                {
                    **dict(alert),
                    "reason_codes": alert["reason_codes_json"],
                }
                for alert in alerts
            ],
        }

    def list_alerts(self, status: str | None = None, limit: int = 50) -> list[AlertView]:
        query = "SELECT * FROM alerts"
        params: list[Any] = []
        if status:
            query += " WHERE status = %s"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [
            AlertView(
                alert_id=row["alert_id"],
                event_id=row["event_id"],
                created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
                status=row["status"],
                verdict=row["verdict"],
                score=row["score"],
                reason_codes=row["reason_codes_json"],
                analyst_resolution=row["analyst_resolution"],
                analyst_comment=row["analyst_comment"],
            )
            for row in rows
        ]

    def resolve_alert(self, alert_id: str, resolution: AlertResolution) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE alerts
                    SET status = %s,
                        analyst_id = %s,
                        analyst_resolution = %s,
                        analyst_comment = %s,
                        resolved_at = %s
                    WHERE alert_id = %s
                    """,
                    (
                        "RESOLVED",
                        resolution.analyst_id,
                        resolution.resolution,
                        resolution.analyst_comment,
                        datetime.now(UTC),
                        alert_id,
                    ),
                )
            conn.commit()

    def monitoring_summary(self, last_hours: int = 24) -> MonitoringSummary:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS total FROM raw_events")
                total_events = cur.fetchone()["total"]
                cur.execute("SELECT COUNT(*) AS total FROM predictions")
                total_predictions = cur.fetchone()["total"]
                cur.execute("SELECT COUNT(*) AS total FROM alerts WHERE status = 'OPEN'")
                open_alerts = cur.fetchone()["total"]
                cur.execute("SELECT COUNT(*) AS total FROM dead_letter_events")
                dead_letter_events = cur.fetchone()["total"]
                cur.execute("SELECT COALESCE(AVG(score), 0.0) AS avg_score, COALESCE(AVG(processing_latency_ms), 0.0) AS avg_latency FROM predictions")
                avg_row = cur.fetchone()
                cur.execute("SELECT verdict, COUNT(*) AS total FROM predictions GROUP BY verdict ORDER BY verdict")
                verdict_rows = cur.fetchall()
                cur.execute("SELECT processing_latency_ms FROM predictions")
                latency_rows = cur.fetchall()

        latencies = [float(row["processing_latency_ms"]) for row in latency_rows]
        p95 = quantiles(latencies, n=20)[-1] if len(latencies) >= 2 else (latencies[0] if latencies else 0.0)
        verdict_counts = {row["verdict"]: int(row["total"]) for row in verdict_rows}
        return MonitoringSummary(
            total_events=int(total_events),
            total_predictions=int(total_predictions),
            open_alerts=int(open_alerts),
            dead_letter_events=int(dead_letter_events),
            average_score=float(avg_row["avg_score"] or 0.0),
            average_latency_ms=float(avg_row["avg_latency"] or 0.0),
            p95_latency_ms=float(p95 or 0.0),
            verdict_counts=verdict_counts,
        )
