from __future__ import annotations

from typing import Any


class CompositeRepository:
    def __init__(self, primary: Any, clickhouse_sink: Any = None, kafka_publisher: Any = None) -> None:
        self.primary = primary
        self.clickhouse_sink = clickhouse_sink
        self.kafka_publisher = kafka_publisher

    def save_raw_event(self, event) -> None:
        self.primary.save_raw_event(event)
        if self.clickhouse_sink:
            self.clickhouse_sink.log_raw_event(event)

    def save_feature_snapshot(self, event_id: str, feature_snapshot: dict[str, Any], encoded_features: dict[str, Any]) -> None:
        self.primary.save_feature_snapshot(event_id, feature_snapshot, encoded_features)

    def save_prediction(self, event, outcome) -> None:
        self.primary.save_prediction(event, outcome)
        if self.clickhouse_sink:
            self.clickhouse_sink.log_prediction(event, outcome)
        if self.kafka_publisher:
            self.kafka_publisher.publish_prediction(outcome.to_response().model_dump(mode="json"))

    def save_alert(self, event, outcome) -> str:
        alert_id = self.primary.save_alert(event, outcome)
        if self.clickhouse_sink:
            self.clickhouse_sink.log_alert(alert_id, event, outcome)
        if self.kafka_publisher:
            payload = outcome.to_response().model_dump(mode="json")
            payload["alert_id"] = alert_id
            self.kafka_publisher.publish_alert(payload)
        return alert_id

    def save_dlq(self, payload: dict[str, Any], error_message: str) -> None:
        self.primary.save_dlq(payload, error_message)
        if self.clickhouse_sink:
            self.clickhouse_sink.log_dead_letter(payload, error_message)
        if self.kafka_publisher:
            self.kafka_publisher.publish_dead_letter({"payload": payload, "error_message": error_message})

    def fetch_event_bundle(self, event_id: str):
        return self.primary.fetch_event_bundle(event_id)

    def list_alerts(self, status: str | None = None, limit: int = 50):
        return self.primary.list_alerts(status=status, limit=limit)

    def resolve_alert(self, alert_id: str, resolution) -> None:
        self.primary.resolve_alert(alert_id, resolution)
        if self.clickhouse_sink:
            self.clickhouse_sink.log_alert_resolution(alert_id, resolution)

    def monitoring_summary(self, last_hours: int = 24):
        return self.primary.monitoring_summary(last_hours=last_hours)
