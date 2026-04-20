from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from aml.application import ResolveAlertUseCase, ScoreTransactionUseCase, VerdictPolicy
from aml.config import get_settings
from aml.infrastructure import (
    ClickHouseSink,
    KafkaPublisher,
    NullClickHouseSink,
    NullKafkaPublisher,
    NullRedisFeatureStore,
    RedisFeatureStore,
)
from aml.inference import InferenceService, ensure_inference_bundle
from aml.monitoring import MetricsRegistry, configure_json_logging
from aml.storage import CompositeRepository, PostgresRepository, SQLiteRepository


@dataclass
class RuntimeContext:
    settings: Any
    bundle: Any
    repository: Any
    metrics: MetricsRegistry
    inference_service: InferenceService
    score_use_case: ScoreTransactionUseCase
    resolve_alert_use_case: ResolveAlertUseCase


def _build_primary_repository(settings):
    backend = settings.storage_backend
    if backend == "postgres":
        return PostgresRepository(settings.postgres_dsn)
    return SQLiteRepository(settings.database_path)


def _build_clickhouse_sink(settings):
    if not settings.enable_clickhouse:
        return NullClickHouseSink()
    try:
        return ClickHouseSink(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            database=settings.clickhouse_database,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
        )
    except Exception:
        logging.getLogger(__name__).exception("ClickHouse sink initialization failed; using null sink")
        return NullClickHouseSink()


def _build_kafka_publisher(settings):
    if not settings.enable_kafka:
        return NullKafkaPublisher()
    try:
        return KafkaPublisher(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            raw_topic=settings.kafka_topic_raw,
            predictions_topic=settings.kafka_topic_predictions,
            alerts_topic=settings.kafka_topic_alerts,
            dead_letter_topic=settings.kafka_topic_dead_letter,
        )
    except Exception:
        logging.getLogger(__name__).exception("Kafka publisher initialization failed; using null publisher")
        return NullKafkaPublisher()


def _build_feature_store(settings):
    if not settings.enable_redis:
        return NullRedisFeatureStore()
    try:
        return RedisFeatureStore(settings.redis_url)
    except Exception:
        logging.getLogger(__name__).exception("Redis feature store initialization failed; using null feature store")
        return NullRedisFeatureStore()


def build_runtime(settings=None) -> RuntimeContext:
    settings = settings or get_settings()
    configure_json_logging(settings.runtime_log_path)
    logging.getLogger("aml.pipelines.feature_pipeline").setLevel(logging.WARNING)
    primary_repository = _build_primary_repository(settings)
    repository = CompositeRepository(
        primary=primary_repository,
        clickhouse_sink=_build_clickhouse_sink(settings),
        kafka_publisher=_build_kafka_publisher(settings),
    )
    feature_store = _build_feature_store(settings)
    metrics = MetricsRegistry()
    bundle = ensure_inference_bundle(settings)
    verdict_policy = VerdictPolicy(
        review_threshold=bundle.threshold_review,
        block_threshold=bundle.threshold_block,
        policy_version=bundle.policy_version,
    )
    inference_service = InferenceService(bundle=bundle, verdict_policy=verdict_policy, feature_store=feature_store)
    score_use_case = ScoreTransactionUseCase(inference_service=inference_service, repository=repository, metrics=metrics)
    resolve_alert_use_case = ResolveAlertUseCase(repository=repository)
    return RuntimeContext(
        settings=settings,
        bundle=bundle,
        repository=repository,
        metrics=metrics,
        inference_service=inference_service,
        score_use_case=score_use_case,
        resolve_alert_use_case=resolve_alert_use_case,
    )
