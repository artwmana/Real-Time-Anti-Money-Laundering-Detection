from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class NullRedisFeatureStore:
    def get_customer_profile(self, customer_id: str, merchant_id: str | None = None) -> dict[str, Any]:
        return {
            "tx_count_24h": 0,
            "amount_sum_24h": 0.0,
            "merchant_tx_count_24h": 0,
        }

    def record_event(self, customer_id: str, merchant_id: str | None, amount: float) -> None:
        return None


class RedisFeatureStore:
    def __init__(self, redis_url: str, ttl_seconds: int = 86_400) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("redis package is required for RedisFeatureStore") from exc

        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    def _customer_count_key(self, customer_id: str) -> str:
        return f"aml:customer:{customer_id}:tx_count_24h"

    def _customer_amount_key(self, customer_id: str) -> str:
        return f"aml:customer:{customer_id}:amount_sum_24h"

    def _merchant_count_key(self, merchant_id: str | None) -> str | None:
        if not merchant_id:
            return None
        return f"aml:merchant:{merchant_id}:tx_count_24h"

    def get_customer_profile(self, customer_id: str, merchant_id: str | None = None) -> dict[str, Any]:
        merchant_key = self._merchant_count_key(merchant_id)
        keys = [self._customer_count_key(customer_id), self._customer_amount_key(customer_id)]
        if merchant_key:
            keys.append(merchant_key)
        values = self.redis.mget(keys)
        merchant_count = int(values[2] or 0) if merchant_key else 0
        return {
            "tx_count_24h": int(values[0] or 0),
            "amount_sum_24h": float(values[1] or 0.0),
            "merchant_tx_count_24h": merchant_count,
        }

    def record_event(self, customer_id: str, merchant_id: str | None, amount: float) -> None:
        pipe = self.redis.pipeline()
        count_key = self._customer_count_key(customer_id)
        amount_key = self._customer_amount_key(customer_id)
        pipe.incr(count_key)
        pipe.expire(count_key, self.ttl_seconds)
        pipe.incrbyfloat(amount_key, float(amount))
        pipe.expire(amount_key, self.ttl_seconds)
        merchant_key = self._merchant_count_key(merchant_id)
        if merchant_key:
            pipe.incr(merchant_key)
            pipe.expire(merchant_key, self.ttl_seconds)
        pipe.execute()
