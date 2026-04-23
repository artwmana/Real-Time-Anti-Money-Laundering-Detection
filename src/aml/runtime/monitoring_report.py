from __future__ import annotations

import json

from aml.config import get_settings
from aml.runtime.bootstrap import build_runtime


def main() -> None:
    runtime = build_runtime()
    summary = runtime.repository.monitoring_summary()
    settings = get_settings()
    settings.monitoring_snapshot_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(settings.monitoring_snapshot_path)


if __name__ == "__main__":
    main()
