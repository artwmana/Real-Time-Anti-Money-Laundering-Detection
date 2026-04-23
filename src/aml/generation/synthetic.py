from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from aml.config import get_settings
from aml.contracts import TransactionEvent
from aml.preprocessing.json_extractor import parse_metadata_payload


class EventGenerator:
    def __init__(
        self,
        settings=None,
        seed_rows: int = 5000,
        random_state: int = 42,
        suspicious_ratio: float = 0.2,
    ) -> None:
        self.settings = settings or get_settings()
        self.random = random.Random(random_state)
        self.seed_frame = pd.read_csv(self.settings.raw_data_path, nrows=seed_rows)
        self.suspicious_ratio = suspicious_ratio

    def _row_to_event(self, row: dict[str, Any], index: int, suspicious: bool = False) -> TransactionEvent:
        metadata = parse_metadata_payload(row.get("metadata", {}))
        base_time = datetime.now(UTC) - timedelta(minutes=index)
        metadata["timestamp"] = base_time.isoformat()
        metadata.setdefault("location", {})
        metadata.setdefault("device_info", {})
        metadata.setdefault("merchant_info", {})
        metadata.setdefault("risk_indicators", {})

        amount_multiplier = self.random.uniform(0.75, 1.35)
        if suspicious:
            amount_multiplier *= self.random.uniform(6.0, 12.0)
        amount = max(1.0, float(row["amount"]) * amount_multiplier)
        old_balance = max(amount + 1.0, float(row["oldbalanceOrg"]) * self.random.uniform(0.7, 1.2))
        new_balance = max(0.0, old_balance - amount)

        metadata["risk_indicators"]["unusual_time"] = suspicious or bool(self.random.random() < 0.1)
        metadata["risk_indicators"]["unusual_location"] = suspicious or bool(self.random.random() < 0.05)
        metadata["risk_indicators"]["risk_score"] = min(
            100.0,
            float(metadata["risk_indicators"].get("risk_score", 20.0))
            * (self.random.uniform(2.5, 4.0) if suspicious else self.random.uniform(0.8, 1.4)),
        )
        metadata["risk_indicators"]["customer_risk_score"] = min(
            100.0,
            float(metadata["risk_indicators"].get("customer_risk_score", 20.0))
            * (self.random.uniform(2.0, 3.5) if suspicious else self.random.uniform(0.9, 1.3)),
        )
        metadata["risk_indicators"]["amount_vs_average"] = (
            float(metadata["risk_indicators"].get("amount_vs_average", 1.0))
            * (self.random.uniform(5.0, 10.0) if suspicious else self.random.uniform(0.8, 1.4))
        )
        if suspicious:
            metadata["merchant_info"] = {
                "merchant_id": f"HR-{self.random.randint(1000, 9999)}",
                "category": "HighRiskTransfer",
                "risk_level": "high",
                "avg_transaction": round(max(10.0, amount / self.random.uniform(10.0, 20.0)), 2),
            }
            row["category"] = "Transfer"

        return TransactionEvent(
            event_id=str(uuid4()),
            source="generator",
            ingested_at=base_time,
            step=int(row.get("step", index)),
            type=str(row["type"]),
            amount=round(amount, 2),
            category=str(row["category"]),
            nameOrig=str(row["nameOrig"]),
            nameDest=str(row["nameDest"]),
            oldbalanceOrg=round(old_balance, 2),
            newbalanceOrig=round(new_balance, 2),
            metadata=metadata,
            isMoneyLaundering=1 if suspicious else int(row.get("isMoneyLaundering", row.get("isFraud", 0))),
            isFraud=1 if suspicious else int(row.get("isFraud", row.get("isMoneyLaundering", 0))),
        )

    def generate(self, count: int = 100) -> list[TransactionEvent]:
        sample = self.seed_frame.sample(n=count, replace=count > len(self.seed_frame), random_state=self.random.randint(1, 10_000))
        events: list[TransactionEvent] = []
        for idx, row in enumerate(sample.to_dict(orient="records")):
            suspicious = self.random.random() < self.suspicious_ratio
            events.append(self._row_to_event(row, idx, suspicious=suspicious))
        return events

    def write_jsonl(self, path: str | Path, count: int = 100) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        events = self.generate(count=count)
        with output.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n")
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AML transaction events as JSONL")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sink", choices=["file", "kafka"], default="file")
    args = parser.parse_args()

    settings = get_settings()
    generator = EventGenerator(settings=settings)
    if args.sink == "kafka":
        from aml.infrastructure.kafka_bus import KafkaPublisher

        publisher = KafkaPublisher(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            raw_topic=settings.kafka_topic_raw,
            predictions_topic=settings.kafka_topic_predictions,
            alerts_topic=settings.kafka_topic_alerts,
            dead_letter_topic=settings.kafka_topic_dead_letter,
        )
        for event in generator.generate(count=args.count):
            publisher.publish_raw_event(event.model_dump(mode="json"))
        print(settings.kafka_topic_raw)
    else:
        output = generator.write_jsonl(args.output or settings.generated_events_path, count=args.count)
        print(output)


if __name__ == "__main__":
    main()
