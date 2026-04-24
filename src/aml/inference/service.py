from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from aml.application.verdict_policy import VerdictPolicy
from aml.contracts.runtime import ScoringResponse, TransactionEvent
from aml.inference.bundle import InferenceBundle
from aml.inference.reasons import derive_reason_codes


def _json_safe_dict(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if hasattr(value, "item"):
            safe[key] = value.item()
        elif isinstance(value, (pd.Timestamp,)):
            safe[key] = value.isoformat()
        else:
            safe[key] = value
    return safe


@dataclass
class InferenceOutcome:
    event_id: str
    score: float
    verdict: str
    risk_band: str
    model_version: str
    policy_version: str
    threshold_review: float
    threshold_block: float
    reason_codes: list[str]
    processing_latency_ms: float
    feature_snapshot: dict[str, Any]
    encoded_features: dict[str, float]

    def to_response(self) -> ScoringResponse:
        return ScoringResponse(
            event_id=self.event_id,
            score=self.score,
            verdict=self.verdict,  # type: ignore[arg-type]
            risk_band=self.risk_band,  # type: ignore[arg-type]
            model_version=self.model_version,
            policy_version=self.policy_version,
            threshold_review=self.threshold_review,
            threshold_block=self.threshold_block,
            reason_codes=self.reason_codes,
            processing_latency_ms=self.processing_latency_ms,
        )


class InferenceService:
    def __init__(self, bundle: InferenceBundle, verdict_policy: VerdictPolicy, feature_store: Any | None = None) -> None:
        self.bundle = bundle
        self.verdict_policy = verdict_policy
        self.feature_store = feature_store

    def score(self, event: TransactionEvent) -> InferenceOutcome:
        started = perf_counter()
        df = pd.DataFrame([event.to_feature_record()])
        stateful_profile: dict[str, Any] = {}
        if self.feature_store is not None:
            stateful_profile = self.feature_store.get_customer_profile(event.nameOrig, event.nameDest)

        if hasattr(self.bundle.pipeline, "transform_with_features"):
            feature_frame, encoded = self.bundle.pipeline.transform_with_features(df)
        else:
            feature_frame = self.bundle.pipeline.build_feature_frame(df)
            encoded = self.bundle.pipeline.transform(df)
        if not isinstance(encoded, pd.DataFrame):
            encoded = pd.DataFrame(encoded, columns=self.bundle.feature_names)

        score = float(self.bundle.model.predict_proba(encoded)[0])
        verdict = self.verdict_policy.decide(score)
        risk_band = self.verdict_policy.risk_band(score)
        feature_snapshot = _json_safe_dict(feature_frame.iloc[0].to_dict())
        feature_snapshot.update(stateful_profile)
        encoded_features = {col: float(encoded.iloc[0][col]) for col in encoded.columns}
        reasons = derive_reason_codes(feature_snapshot, score, verdict)
        latency_ms = round((perf_counter() - started) * 1000.0, 3)
        if self.feature_store is not None:
            self.feature_store.record_event(event.nameOrig, event.nameDest, event.amount)

        return InferenceOutcome(
            event_id=event.event_id,
            score=score,
            verdict=verdict,
            risk_band=risk_band,
            model_version=self.bundle.model_version,
            policy_version=self.verdict_policy.policy_version,
            threshold_review=self.verdict_policy.review_threshold,
            threshold_block=self.verdict_policy.block_threshold,
            reason_codes=reasons,
            processing_latency_ms=latency_ms,
            feature_snapshot=feature_snapshot,
            encoded_features=encoded_features,
        )
