from __future__ import annotations

import json
from pathlib import Path

from aml.generation.synthetic import EventGenerator
from aml.runtime.bootstrap import build_runtime


def main() -> None:
    runtime = build_runtime()
    generator = EventGenerator(settings=runtime.settings, seed_rows=500)
    path = generator.write_jsonl(runtime.settings.generated_events_path, count=25)

    processed = 0
    review_or_block = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            result = runtime.score_use_case.execute_payload(json.loads(line))
            processed += 1
            if result.verdict in {"REVIEW", "BLOCK"}:
                review_or_block += 1

    summary = runtime.repository.monitoring_summary()
    print(
        json.dumps(
            {
                "processed": processed,
                "review_or_block": review_or_block,
                "total_predictions": summary.total_predictions,
                "open_alerts": summary.open_alerts,
                "db_path": str(runtime.settings.database_path),
                "bundle_path": str(runtime.settings.inference_bundle_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
