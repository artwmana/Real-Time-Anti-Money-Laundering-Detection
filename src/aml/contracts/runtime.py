from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TransactionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str = "api"
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    step: int = 0
    type: str
    amount: float
    category: str
    nameOrig: str
    nameDest: str
    oldbalanceOrg: float
    newbalanceOrig: float

    hour: int | None = None
    day_of_week: int | None = None
    day_of_month: int | None = None
    month: int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    isMoneyLaundering: int | None = None
    isFraud: int | None = None

    def to_feature_record(self) -> dict[str, Any]:
        record = self.model_dump(
            exclude={
                "event_id",
                "source",
                "ingested_at",
                "isMoneyLaundering",
                "isFraud",
            }
        )
        metadata = dict(record.get("metadata", {}))
        metadata.setdefault("timestamp", self.ingested_at.isoformat())
        record["metadata"] = metadata

        ts = datetime.fromisoformat(str(metadata["timestamp"]).replace("Z", "+00:00"))
        record["hour"] = self.hour if self.hour is not None else ts.hour
        record["day_of_week"] = self.day_of_week if self.day_of_week is not None else ts.weekday()
        record["day_of_month"] = self.day_of_month if self.day_of_month is not None else ts.day
        record["month"] = self.month if self.month is not None else ts.month
        return record


class ScoringResponse(BaseModel):
    event_id: str
    score: float
    verdict: Literal["CLEAR", "REVIEW", "BLOCK"]
    risk_band: Literal["low", "medium", "high"]
    model_version: str
    policy_version: str
    threshold_review: float
    threshold_block: float
    reason_codes: list[str] = Field(default_factory=list)
    processing_latency_ms: float


class AlertView(BaseModel):
    alert_id: str
    event_id: str
    created_at: str
    status: str
    verdict: str
    score: float
    reason_codes: list[str] = Field(default_factory=list)
    analyst_resolution: str | None = None
    analyst_comment: str | None = None


class AlertResolution(BaseModel):
    analyst_id: str
    resolution: Literal["TRUE_POSITIVE", "FALSE_POSITIVE", "ESCALATED", "DISMISSED"]
    analyst_comment: str | None = None


class MonitoringSummary(BaseModel):
    total_events: int
    total_predictions: int
    open_alerts: int
    dead_letter_events: int
    average_score: float
    average_latency_ms: float
    p95_latency_ms: float
    verdict_counts: dict[str, int] = Field(default_factory=dict)
