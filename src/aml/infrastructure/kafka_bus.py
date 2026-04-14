from __future__ import annotations

import json
from typing import Any


class NullKafkaPublisher:
    def publish_raw_event(self, payload: dict[str, Any]) -> None:
        return None

    def publish_prediction(self, payload: dict[str, Any]) -> None:
        return None

    def publish_alert(self, payload: dict[str, Any]) -> None:
        return None

    def publish_dead_letter(self, payload: dict[str, Any]) -> None:
        return None


class KafkaPublisher:
    def __init__(
        self,
        bootstrap_servers: str,
        raw_topic: str,
        predictions_topic: str,
        alerts_topic: str,
        dead_letter_topic: str,
    ) -> None:
        try:
            from kafka import KafkaProducer
        except ImportError as exc:
            raise RuntimeError("kafka-python package is required for KafkaPublisher") from exc

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        )
        self.raw_topic = raw_topic
        self.predictions_topic = predictions_topic
        self.alerts_topic = alerts_topic
        self.dead_letter_topic = dead_letter_topic

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.producer.send(topic, payload)
        self.producer.flush()

    def publish_raw_event(self, payload: dict[str, Any]) -> None:
        self._publish(self.raw_topic, payload)

    def publish_prediction(self, payload: dict[str, Any]) -> None:
        self._publish(self.predictions_topic, payload)

    def publish_alert(self, payload: dict[str, Any]) -> None:
        self._publish(self.alerts_topic, payload)

    def publish_dead_letter(self, payload: dict[str, Any]) -> None:
        self._publish(self.dead_letter_topic, payload)
