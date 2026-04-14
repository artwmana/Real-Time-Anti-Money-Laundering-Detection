from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


class NullClickHouseSink:
    def log_raw_event(self, event: Any) -> None:
        return None

    def log_prediction(self, event: Any, outcome: Any) -> None:
        return None

    def log_alert(self, alert_id: str, event: Any, outcome: Any) -> None:
        return None

    def log_dead_letter(self, payload: dict[str, Any], error_message: str) -> None:
        return None

    def log_alert_resolution(self, alert_id: str, resolution: Any) -> None:
        return None


class ClickHouseSink:
    def __init__(self, host: str, port: int, database: str, username: str, password: str) -> None:
        try:
            import clickhouse_connect
        except ImportError as exc:
            raise RuntimeError("clickhouse-connect package is required for ClickHouseSink") from exc

        self.client = clickhouse_connect.get_client(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
        )
        self._initialize()

    def _initialize(self) -> None:
        self.client.command(
            """
            CREATE TABLE IF NOT EXISTS event_logs (
                logged_at DateTime,
                event_id String,
                source String,
                payload_json String
            ) ENGINE = MergeTree ORDER BY (logged_at, event_id)
            """
        )
        self.client.command(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                logged_at DateTime,
                event_id String,
                score Float64,
                verdict String,
                risk_band String,
                model_version String,
                policy_version String,
                latency_ms Float64,
                reason_codes_json String,
                feature_json String
            ) ENGINE = MergeTree ORDER BY (logged_at, event_id)
            """
        )
        self.client.command(
            """
            CREATE TABLE IF NOT EXISTS alert_logs (
                logged_at DateTime,
                alert_id String,
                event_id String,
                verdict String,
                score Float64,
                reason_codes_json String,
                resolution_json String
            ) ENGINE = MergeTree ORDER BY (logged_at, alert_id)
            """
        )
        self.client.command(
            """
            CREATE TABLE IF NOT EXISTS dead_letter_logs (
                logged_at DateTime,
                payload_json String,
                error_message String
            ) ENGINE = MergeTree ORDER BY logged_at
            """
        )

    def log_raw_event(self, event: Any) -> None:
        self.client.insert(
            "event_logs",
            [[datetime.now(UTC), event.event_id, event.source, json.dumps(event.model_dump(mode='json'), ensure_ascii=False)]],
            column_names=["logged_at", "event_id", "source", "payload_json"],
        )

    def log_prediction(self, event: Any, outcome: Any) -> None:
        self.client.insert(
            "prediction_logs",
            [[
                datetime.now(UTC),
                event.event_id,
                float(outcome.score),
                outcome.verdict,
                outcome.risk_band,
                outcome.model_version,
                outcome.policy_version,
                float(outcome.processing_latency_ms),
                json.dumps(outcome.reason_codes, ensure_ascii=False),
                json.dumps(outcome.feature_snapshot, ensure_ascii=False, default=str),
            ]],
            column_names=[
                "logged_at",
                "event_id",
                "score",
                "verdict",
                "risk_band",
                "model_version",
                "policy_version",
                "latency_ms",
                "reason_codes_json",
                "feature_json",
            ],
        )

    def log_alert(self, alert_id: str, event: Any, outcome: Any) -> None:
        self.client.insert(
            "alert_logs",
            [[
                datetime.now(UTC),
                alert_id,
                event.event_id,
                outcome.verdict,
                float(outcome.score),
                json.dumps(outcome.reason_codes, ensure_ascii=False),
                "",
            ]],
            column_names=["logged_at", "alert_id", "event_id", "verdict", "score", "reason_codes_json", "resolution_json"],
        )

    def log_dead_letter(self, payload: dict[str, Any], error_message: str) -> None:
        self.client.insert(
            "dead_letter_logs",
            [[datetime.now(UTC), json.dumps(payload, ensure_ascii=False, default=str), error_message]],
            column_names=["logged_at", "payload_json", "error_message"],
        )

    def log_alert_resolution(self, alert_id: str, resolution: Any) -> None:
        self.client.insert(
            "alert_logs",
            [[
                datetime.now(UTC),
                alert_id,
                "",
                "",
                0.0,
                "[]",
                json.dumps(resolution.model_dump(mode="json"), ensure_ascii=False),
            ]],
            column_names=["logged_at", "alert_id", "event_id", "verdict", "score", "reason_codes_json", "resolution_json"],
        )
