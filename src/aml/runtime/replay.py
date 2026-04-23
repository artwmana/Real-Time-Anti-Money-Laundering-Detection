from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request

from aml.contracts import TransactionEvent
from aml.generation.synthetic import EventGenerator
from aml.runtime.bootstrap import build_runtime


def replay_direct(path: str | Path) -> dict[str, int]:
    runtime = build_runtime()
    processed = 0
    alerts = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            result = runtime.score_use_case.execute(TransactionEvent(**payload))
            processed += 1
            if result.verdict in {"REVIEW", "BLOCK"}:
                alerts += 1
    return {"processed": processed, "alerts": alerts}


def replay_via_api(path: str | Path, url: str) -> dict[str, int]:
    processed = 0
    alerts = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line).copy()
            req = request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            processed += 1
            if body["verdict"] in {"REVIEW", "BLOCK"}:
                alerts += 1
    return {"processed": processed, "alerts": alerts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay generated AML events through the product pipeline")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--mode", choices=["direct", "api", "kafka"], default="direct")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/score")
    args = parser.parse_args()

    if args.input is None:
        runtime = build_runtime()
        generator = EventGenerator(settings=runtime.settings)
        input_path = generator.write_jsonl(runtime.settings.generated_events_path, count=args.count)
    else:
        input_path = Path(args.input)

    if args.mode == "api":
        summary = replay_via_api(input_path, args.url)
    elif args.mode == "kafka":
        from aml.config import get_settings
        from aml.infrastructure.kafka_bus import KafkaPublisher

        settings = get_settings()
        publisher = KafkaPublisher(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            raw_topic=settings.kafka_topic_raw,
            predictions_topic=settings.kafka_topic_predictions,
            alerts_topic=settings.kafka_topic_alerts,
            dead_letter_topic=settings.kafka_topic_dead_letter,
        )
        processed = 0
        with Path(input_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                publisher.publish_raw_event(json.loads(line))
                processed += 1
        summary = {"processed": processed, "alerts": 0}
    else:
        summary = replay_direct(input_path)

    print(json.dumps({"input": str(input_path), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
