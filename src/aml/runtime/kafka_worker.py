from __future__ import annotations

import argparse
import json
import logging

from aml.runtime.bootstrap import build_runtime

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume AML events from Kafka and process them through the runtime")
    parser.add_argument("--group-id", type=str, default="aml-runtime-worker")
    args = parser.parse_args()

    runtime = build_runtime()
    settings = runtime.settings

    try:
        from kafka import KafkaConsumer
    except ImportError as exc:
        raise RuntimeError("kafka-python package is required for aml.runtime.kafka_worker") from exc

    consumer = KafkaConsumer(
        settings.kafka_topic_raw,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        group_id=args.group_id,
        auto_offset_reset="earliest",
    )

    for message in consumer:
        payload = message.value
        try:
            runtime.score_use_case.execute_payload(payload)
        except Exception as exc:
            runtime.metrics.record_error()
            runtime.repository.save_dlq(payload, str(exc))
            logger.exception("Kafka worker failed to process event")


if __name__ == "__main__":
    main()
