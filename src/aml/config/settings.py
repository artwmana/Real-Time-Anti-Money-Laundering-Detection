from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").exists()
        and (path / "src").exists()
        and ((path / "data").exists() or (path / "models").exists())
    )


def _discover_project_root() -> Path:
    env_root = os.getenv("AML_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _looks_like_project_root(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _looks_like_project_root(candidate):
            return candidate

    package_root = Path(__file__).resolve()
    for candidate in package_root.parents:
        if _looks_like_project_root(candidate):
            return candidate

    return cwd


@dataclass(frozen=True)
class Settings:
    project_root: Path
    storage_backend: str
    data_path: Path
    raw_data_path: Path
    processed_train_path: Path
    processed_val_path: Path
    processed_test_path: Path
    models_path: Path
    legacy_model_dir: Path
    inference_bundle_path: Path
    database_path: Path
    logs_dir: Path
    runtime_log_path: Path
    generated_events_path: Path
    monitoring_dir: Path
    monitoring_snapshot_path: Path
    postgres_dsn: str
    clickhouse_host: str
    clickhouse_port: int
    clickhouse_database: str
    clickhouse_user: str
    clickhouse_password: str
    redis_url: str
    kafka_bootstrap_servers: str
    kafka_topic_raw: str
    kafka_topic_predictions: str
    kafka_topic_alerts: str
    kafka_topic_dead_letter: str
    mlflow_tracking_uri: str
    mlflow_experiment: str
    api_base_url: str
    enable_clickhouse: bool
    enable_kafka: bool
    enable_redis: bool
    enable_mlflow: bool

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = _discover_project_root()
        load_dotenv(project_root / ".env")

        data_path = Path(os.getenv("DATA_PATH", project_root / "data")).expanduser()
        models_path = Path(os.getenv("MODEL_PATH", project_root / "models")).expanduser()
        runtime_dir = Path(os.getenv("AML_RUNTIME_DIR", project_root / "runtime")).expanduser()
        logs_dir = Path(os.getenv("AML_LOG_DIR", project_root / "logs")).expanduser()
        monitoring_dir = Path(os.getenv("AML_MONITORING_DIR", data_path / "monitoring")).expanduser()
        storage_backend = os.getenv("AML_STORAGE_BACKEND", "sqlite").strip().lower()

        runtime_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        monitoring_dir.mkdir(parents=True, exist_ok=True)

        database_path = Path(os.getenv("AML_DB_PATH", runtime_dir / "aml_runtime.sqlite3")).expanduser()
        generated_events_path = Path(
            os.getenv("AML_GENERATED_EVENTS_PATH", runtime_dir / "generated_events.jsonl")
        ).expanduser()
        api_host = os.getenv("AML_API_HOST", "127.0.0.1")
        api_port = int(os.getenv("AML_API_PORT", "8000"))

        return cls(
            project_root=project_root,
            storage_backend=storage_backend,
            data_path=data_path,
            raw_data_path=data_path / "raw/AMLNet_August_2025.csv",
            processed_train_path=data_path / "processed/train.parquet",
            processed_val_path=data_path / "processed/val.parquet",
            processed_test_path=data_path / "processed/test.parquet",
            models_path=models_path,
            legacy_model_dir=models_path / "base_models_only",
            inference_bundle_path=models_path / "inference_bundle.joblib",
            database_path=database_path,
            logs_dir=logs_dir,
            runtime_log_path=logs_dir / "aml_runtime.jsonl",
            generated_events_path=generated_events_path,
            monitoring_dir=monitoring_dir,
            monitoring_snapshot_path=monitoring_dir / "latest_summary.json",
            postgres_dsn=os.getenv("AML_POSTGRES_DSN", "postgresql://aml:aml@localhost:5432/aml"),
            clickhouse_host=os.getenv("AML_CLICKHOUSE_HOST", "localhost"),
            clickhouse_port=int(os.getenv("AML_CLICKHOUSE_PORT", "8123")),
            clickhouse_database=os.getenv("AML_CLICKHOUSE_DATABASE", "aml"),
            clickhouse_user=os.getenv("AML_CLICKHOUSE_USER", "default"),
            clickhouse_password=os.getenv("AML_CLICKHOUSE_PASSWORD", ""),
            redis_url=os.getenv("AML_REDIS_URL", "redis://localhost:6379/0"),
            kafka_bootstrap_servers=os.getenv("AML_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            kafka_topic_raw=os.getenv("AML_KAFKA_TOPIC_RAW", "transactions_raw"),
            kafka_topic_predictions=os.getenv("AML_KAFKA_TOPIC_PREDICTIONS", "aml_predictions"),
            kafka_topic_alerts=os.getenv("AML_KAFKA_TOPIC_ALERTS", "aml_alerts"),
            kafka_topic_dead_letter=os.getenv("AML_KAFKA_TOPIC_DEAD_LETTER", "aml_dead_letter"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT_NAME", "aml-realtime"),
            api_base_url=os.getenv("AML_API_BASE_URL", f"http://{api_host}:{api_port}"),
            enable_clickhouse=os.getenv("AML_ENABLE_CLICKHOUSE", "1").strip() not in {"0", "false", "False"},
            enable_kafka=os.getenv("AML_ENABLE_KAFKA", "1").strip() not in {"0", "false", "False"},
            enable_redis=os.getenv("AML_ENABLE_REDIS", "1").strip() not in {"0", "false", "False"},
            enable_mlflow=os.getenv("AML_ENABLE_MLFLOW", "1").strip() not in {"0", "false", "False"},
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
