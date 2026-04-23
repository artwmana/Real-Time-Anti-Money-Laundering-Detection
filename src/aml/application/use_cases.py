from __future__ import annotations

from dataclasses import asdict
from typing import Any

from aml.contracts.runtime import AlertResolution, ScoringResponse, TransactionEvent


class ScoreTransactionUseCase:
    def __init__(self, inference_service: Any, repository: Any, metrics: Any | None = None) -> None:
        self.inference_service = inference_service
        self.repository = repository
        self.metrics = metrics

    def execute(self, event: TransactionEvent) -> ScoringResponse:
        self.repository.save_raw_event(event)
        outcome = self.inference_service.score(event)
        self.repository.save_feature_snapshot(event.event_id, outcome.feature_snapshot, outcome.encoded_features)
        self.repository.save_prediction(event, outcome)
        if outcome.verdict in {"REVIEW", "BLOCK"}:
            self.repository.save_alert(event, outcome)
        if self.metrics is not None:
            self.metrics.record_success(outcome.verdict, outcome.processing_latency_ms / 1000.0)
        return outcome.to_response()

    def execute_payload(self, payload: dict[str, Any]) -> ScoringResponse:
        return self.execute(TransactionEvent(**payload))


class ResolveAlertUseCase:
    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def execute(self, alert_id: str, resolution: AlertResolution) -> dict[str, Any]:
        self.repository.resolve_alert(alert_id, resolution)
        return {
            "alert_id": alert_id,
            "status": "RESOLVED",
            "resolution": resolution.model_dump(),
        }
