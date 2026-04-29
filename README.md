# Real-Time AML Detection

Пет-проект end-to-end ML-системы для обнаружения подозрительных финансовых транзакций в реальном времени от обучения модели, генерацию событий, online scoring API, stateful features, Kafka ingestion, операционное хранение, аналитические логи, мониторинг и replay трафика.

[English version](README-en.md)

## Задача

Финансовые транзакции поступают как поток событий. Для каждой транзакции система должна быстро:

- собрать признаки из raw payload и исторического состояния клиента;
- посчитать вероятность AML-risk;
- принять бизнес-вердикт `CLEAR`, `BLOCK` или `REVIEW`;
- сохранить результат, audit trail и alerts;
- отдать ответ через API или обработать событие асинхронно.

## Что демонстрирует проект

- **ML pipeline**: preprocessing, feature engineering, training, inference bundle.
- **Real-time serving**: FastAPI endpoint `POST /score`.
- **Streaming architecture**: Kafka worker и Spark Structured Streaming bridge.
- **Stateful features**: Redis-счетчики по клиентам и мерчантам за 24 часа.
- **Operational storage**: PostgreSQL как источник операционной правды.
- **Analytics storage**: ClickHouse для событий, predictions, alerts и dead-letter logs.
- **Monitoring**: Prometheus-compatible `/metrics`, runtime JSON logs, monitoring summary.
- **Local fallback**: SQLite-режим для запуска без инфраструктуры.
- **Reproducibility**: serving через `models/inference_bundle.joblib`.

## Основной сценарий

1. Сгенерировать или получить транзакцию из Kafka.
2. Обогатить событие online-счетчиками из Redis.
3. Прогнать событие через fitted feature pipeline и ensemble model.
4. Применить policy thresholds и вернуть бизнес-вердикт.
5. Сохранить prediction, alert и audit context.
6. Отправить события в мониторинг и аналитические хранилища.

## API

- `GET /health` - health check.
- `GET /ready` - проверка готовности bundle и storage backend.
- `POST /score` - синхронный скоринг транзакции.
- `GET /events/{event_id}` - получение события с prediction/audit context.
- `GET /alerts` - список alerts.
- `POST /alerts/{alert_id}/resolution` - закрытие alert.
- `GET /metrics` - Prometheus metrics.
- `GET /monitoring/summary` - агрегированная мониторинговая сводка.

## Задержка

| Режим | Средняя | p50 | p95 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Docker end-to-end scoring | `21.7 ms` | `22.0 ms` | `25.5 ms` | `18.3 ms` | `25.9 ms` |
| Inference only | `10.9 ms` | `10.7 ms` | `12.4 ms` | - | - |
| Model `predict_proba` only | `2.4 ms` | `2.4 ms` | `3.2 ms` | `1.8 ms` | `4.4 ms` |

## Запустить

### Полный infrastructure stack

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
docker compose up --build -d
```

- API: `http://127.0.0.1:8000`
- MLflow: `http://127.0.0.1:5000`
- Kafka: `127.0.0.1:9092`
- PostgreSQL: `127.0.0.1:5432`
- ClickHouse HTTP: `127.0.0.1:8123`
- Redis: `127.0.0.1:6379`
- Spark master UI: `http://127.0.0.1:8080`

### Тесты

```bash
AML_STORAGE_BACKEND=postgres \
AML_POSTGRES_DSN=postgresql://aml:aml@localhost:5432/aml \
AML_CLICKHOUSE_HOST=localhost \
AML_CLICKHOUSE_PORT=8123 \
AML_REDIS_URL=redis://localhost:6379/0 \
AML_KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
MLFLOW_TRACKING_URI=http://localhost:5000 \
python -m aml.runtime.smoke_test
```

### Локальный запуск API

```bash
AML_STORAGE_BACKEND=sqlite \
AML_ENABLE_CLICKHOUSE=0 \
AML_ENABLE_KAFKA=0 \
AML_ENABLE_REDIS=0 \
AML_ENABLE_MLFLOW=0 \
python -m uvicorn aml.api.app:app --host 127.0.0.1 --port 8000
```

### Генерация и replay трафика

```bash
aml-generate --count 100
aml-replay --input runtime/generated_events.jsonl --mode api --url http://127.0.0.1:8000/score
```

## Обучение модели

```bash
aml-train
```

Основные training outputs:

- `models/aml_ensemble/`
- `models/base_models_only/`
- `models/inference_bundle.joblib`

Serving path использует именно fitted `inference_bundle.joblib`, потому что для корректного runtime scoring нужна не только модель, но и состояние preprocessing/feature pipeline.

## Структура проекта

```text
src/aml/api              FastAPI application
src/aml/application      use cases and verdict policy
src/aml/config           runtime settings
src/aml/contracts        event and response schemas
src/aml/generation       synthetic transaction generator
src/aml/inference        inference service and bundle
src/aml/infrastructure   Kafka, Redis, ClickHouse, MLflow adapters
src/aml/models           ensemble model code
src/aml/monitoring       metrics and JSON logging
src/aml/pipelines        feature engineering pipeline
src/aml/runtime          bootstrap, replay, smoke test, workers
src/aml/storage          PostgreSQL, SQLite and composite repositories
src/aml/training         training entrypoints
```

## Документация

- [Architecture](docs/ARCHITECTURE.md)
- [ML System Design](docs/ML_SYSTEM_DESIGN.md)
- [API](docs/API.md)
- [Runbook](docs/RUNBOOK.md)
- [Monitoring](docs/MONITORING.md)
- [Retraining](docs/RETRAINING.md)
