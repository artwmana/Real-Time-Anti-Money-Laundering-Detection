from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from statistics import quantiles
from typing import Any
from uuid import uuid4

from aml.contracts.runtime import AlertResolution, AlertView, MonitoringSummary, TransactionEvent


def _serialize(payload: Any) -> str:
    def _default(value: Any):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    return json.dumps(payload, ensure_ascii=False, default=_default)


class SQLiteRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_events (
                    event_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    event_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    feature_json TEXT NOT NULL,
                    encoded_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    event_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    risk_band TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    threshold_review REAL NOT NULL,
                    threshold_block REAL NOT NULL,
                    processing_latency_ms REAL NOT NULL,
                    reason_codes_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    score REAL NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    analyst_id TEXT,
                    analyst_resolution TEXT,
                    analyst_comment TEXT,
                    resolved_at TEXT,
                    FOREIGN KEY(event_id) REFERENCES predictions(event_id)
                );

                CREATE TABLE IF NOT EXISTS dead_letter_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    error_message TEXT NOT NULL
                );
                """
            )

    def save_raw_event(self, event: TransactionEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO raw_events(event_id, source, ingested_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.source,
                    event.ingested_at.isoformat(),
                    _serialize(event.model_dump()),
                ),
            )

    def save_feature_snapshot(self, event_id: str, feature_snapshot: dict[str, Any], encoded_features: dict[str, Any]) -> None:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_snapshots(event_id, created_at, feature_json, encoded_json)
                VALUES (?, ?, ?, ?)
                """,
                (event_id, created_at, _serialize(feature_snapshot), _serialize(encoded_features)),
            )

    def save_prediction(self, event: TransactionEvent, outcome: Any) -> None:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO predictions(
                    event_id, created_at, score, verdict, risk_band, model_version, policy_version,
                    threshold_review, threshold_block, processing_latency_ms, reason_codes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    created_at,
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

    def save_alert(self, event: TransactionEvent, outcome: Any) -> str:
        alert_id = str(uuid4())
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts(
                    alert_id, event_id, created_at, status, verdict, score, reason_codes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_id,
                    event.event_id,
                    created_at,
                    "OPEN",
                    outcome.verdict,
                    outcome.score,
                    _serialize(outcome.reason_codes),
                ),
            )
        return alert_id

    def save_dlq(self, payload: dict[str, Any], error_message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dead_letter_events(created_at, payload_json, error_message)
                VALUES (?, ?, ?)
                """,
                (datetime.now(UTC).isoformat(), _serialize(payload), error_message),
            )

    def fetch_event_bundle(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            raw = conn.execute("SELECT * FROM raw_events WHERE event_id = ?", (event_id,)).fetchone()
            pred = conn.execute("SELECT * FROM predictions WHERE event_id = ?", (event_id,)).fetchone()
            feat = conn.execute("SELECT * FROM feature_snapshots WHERE event_id = ?", (event_id,)).fetchone()
            alerts = conn.execute("SELECT * FROM alerts WHERE event_id = ? ORDER BY created_at DESC", (event_id,)).fetchall()

        if raw is None and pred is None:
            return None

        prediction_payload = None
        if pred:
            prediction_payload = dict(pred)
            prediction_payload["reason_codes"] = json.loads(pred["reason_codes_json"])

        return {
            "event_id": event_id,
            "raw_event": json.loads(raw["payload_json"]) if raw else None,
            "prediction": prediction_payload,
            "feature_snapshot": json.loads(feat["feature_json"]) if feat else None,
            "encoded_features": json.loads(feat["encoded_json"]) if feat else None,
            "alerts": [
                {
                    **dict(alert),
                    "reason_codes": json.loads(alert["reason_codes_json"]),
                }
                for alert in alerts
            ],
        }

    def list_alerts(self, status: str | None = None, limit: int = 50) -> list[AlertView]:
        query = "SELECT * FROM alerts"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()

        return [
            AlertView(
                alert_id=row["alert_id"],
                event_id=row["event_id"],
                created_at=row["created_at"],
                status=row["status"],
                verdict=row["verdict"],
                score=row["score"],
                reason_codes=json.loads(row["reason_codes_json"]),
                analyst_resolution=row["analyst_resolution"],
                analyst_comment=row["analyst_comment"],
            )
            for row in rows
        ]

    def resolve_alert(self, alert_id: str, resolution: AlertResolution) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE alerts
                SET status = ?, analyst_id = ?, analyst_resolution = ?, analyst_comment = ?, resolved_at = ?
                WHERE alert_id = ?
                """,
                (
                    "RESOLVED",
                    resolution.analyst_id,
                    resolution.resolution,
                    resolution.analyst_comment,
                    datetime.now(UTC).isoformat(),
                    alert_id,
                ),
            )

    def monitoring_summary(self, last_hours: int = 24) -> MonitoringSummary:
        with self._connect() as conn:
            total_events = conn.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]
            total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            open_alerts = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'OPEN'").fetchone()[0]
            dead_letter_events = conn.execute("SELECT COUNT(*) FROM dead_letter_events").fetchone()[0]
            avg_row = conn.execute(
                "SELECT COALESCE(AVG(score), 0.0), COALESCE(AVG(processing_latency_ms), 0.0) FROM predictions"
            ).fetchone()
            verdict_rows = conn.execute(
                "SELECT verdict, COUNT(*) AS total FROM predictions GROUP BY verdict ORDER BY verdict"
            ).fetchall()
            latency_rows = conn.execute("SELECT processing_latency_ms FROM predictions").fetchall()

        latencies = [float(row["processing_latency_ms"]) for row in latency_rows]
        p95 = quantiles(latencies, n=20)[-1] if len(latencies) >= 2 else (latencies[0] if latencies else 0.0)
        verdict_counts = {row["verdict"]: int(row["total"]) for row in verdict_rows}

        return MonitoringSummary(
            total_events=int(total_events),
            total_predictions=int(total_predictions),
            open_alerts=int(open_alerts),
            dead_letter_events=int(dead_letter_events),
            average_score=float(avg_row[0] or 0.0),
            average_latency_ms=float(avg_row[1] or 0.0),
            p95_latency_ms=float(p95 or 0.0),
            verdict_counts=verdict_counts,
        )
