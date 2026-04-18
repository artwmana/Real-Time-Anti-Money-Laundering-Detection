from __future__ import annotations

from collections import Counter
from statistics import mean
from threading import Lock


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self.requests_total = 0
        self.errors_total = 0
        self.verdict_total: Counter[str] = Counter()
        self.latencies: list[float] = []

    def record_success(self, verdict: str, latency_seconds: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.verdict_total[verdict] += 1
            self.latencies.append(float(latency_seconds))

    def record_error(self) -> None:
        with self._lock:
            self.errors_total += 1

    def render_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP aml_scoring_requests_total Total processed scoring requests",
                "# TYPE aml_scoring_requests_total counter",
                f"aml_scoring_requests_total {self.requests_total}",
                "# HELP aml_scoring_errors_total Total scoring errors",
                "# TYPE aml_scoring_errors_total counter",
                f"aml_scoring_errors_total {self.errors_total}",
                "# HELP aml_scoring_latency_seconds_avg Average scoring latency in seconds",
                "# TYPE aml_scoring_latency_seconds_avg gauge",
                f"aml_scoring_latency_seconds_avg {mean(self.latencies) if self.latencies else 0.0}",
                "# HELP aml_verdict_total Total predictions by verdict",
                "# TYPE aml_verdict_total counter",
            ]
            for verdict, total in sorted(self.verdict_total.items()):
                lines.append(f'aml_verdict_total{{verdict="{verdict}"}} {total}')
        return "\n".join(lines) + "\n"
