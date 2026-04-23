from __future__ import annotations

from typing import Any


def derive_reason_codes(feature_snapshot: dict[str, Any], score: float, verdict: str) -> list[str]:
    reasons: list[str] = []

    amount = float(feature_snapshot.get("amount", 0) or 0)
    amt_to_balance = float(feature_snapshot.get("amt_to_balance", 0) or 0)
    risk_score = float(feature_snapshot.get("risk_risk_score", 0) or 0)
    customer_risk = float(feature_snapshot.get("risk_customer_risk_score", 0) or 0)

    if amount >= 5_000:
        reasons.append("large_amount")
    if amt_to_balance >= 0.3:
        reasons.append("high_amount_vs_balance")
    if risk_score >= 70:
        reasons.append("high_risk_indicator")
    if customer_risk >= 80:
        reasons.append("high_customer_risk")
    if bool(feature_snapshot.get("risk_unusual_time")):
        reasons.append("unusual_time")
    if bool(feature_snapshot.get("risk_unusual_location")):
        reasons.append("unusual_location")
    if bool(feature_snapshot.get("is_weekend")) and amount >= 2_000:
        reasons.append("weekend_high_value")
    if float(feature_snapshot.get("tx_count_24h", 0) or 0) >= 10:
        reasons.append("high_customer_velocity_24h")
    if float(feature_snapshot.get("amount_sum_24h", 0) or 0) >= 20_000:
        reasons.append("high_customer_volume_24h")
    if float(feature_snapshot.get("merchant_tx_count_24h", 0) or 0) >= 20:
        reasons.append("high_merchant_velocity_24h")
    if score >= 0.9:
        reasons.append("extreme_model_score")
    if verdict in {"REVIEW", "BLOCK"} and not reasons:
        reasons.append("elevated_model_risk")

    return reasons
