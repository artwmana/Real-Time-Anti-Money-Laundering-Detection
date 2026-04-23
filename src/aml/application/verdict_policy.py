from __future__ import annotations


class VerdictPolicy:
    def __init__(self, review_threshold: float, block_threshold: float, policy_version: str) -> None:
        if review_threshold >= block_threshold:
            raise ValueError("review_threshold must be lower than block_threshold")

        self.review_threshold = float(review_threshold)
        self.block_threshold = float(block_threshold)
        self.policy_version = policy_version

    def decide(self, score: float) -> str:
        if score >= self.block_threshold:
            return "BLOCK"
        if score >= self.review_threshold:
            return "REVIEW"
        return "CLEAR"

    def risk_band(self, score: float) -> str:
        if score >= self.block_threshold:
            return "high"
        if score >= self.review_threshold:
            return "medium"
        return "low"
