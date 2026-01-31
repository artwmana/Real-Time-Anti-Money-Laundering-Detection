# AML System Architecture (Clean Architecture)

## 1. Purpose

This document describes the architecture of the **Real-Time Anti-Money-Laundering (AML) Detection System** based on the principles of **Clean Architecture** and **Separation of Concerns**.  
The goal is to achieve:

- High testability
- Low coupling between components
- Independent business logic
- Scalable real-time inference
- Clear separation between ML, API, and infrastructure

---

## 2. Architectural Style

The project follows **Clean Architecture (Onion Architecture)** with strict dependency rules:

- **Inner layers never depend on outer layers**
- **Infrastructure depends on Domain**
- **Frameworks are replaceable**
- **Business logic is framework-agnostic**

Dependency direction:

```

API / UI
↓
Application (Use Cases)
↓
Domain (Entities, Business Rules)
↑
Infrastructure (DB, Kafka, MLflow, Models)

```

---

## 3. High-Level System Overview

```

[ Client / External Systems ]
↓
FastAPI API
↓
Application Layer
↓
Domain Layer
↑
Infrastructure Layer
↓
Databases | Kafka | ML Models | MLflow | Monitoring

```

The system processes **real-time financial transactions**, extracts features, evaluates risk using ML models, and produces an AML decision.

---

## 4. Project Folder Structure (src-layout)

```

src/
└── aml/
├── **init**.py
│
├── domain/                   # Pure business logic
│   ├── entities.py           # Transaction, Customer, RiskProfile
│   ├── value_objects.py      # Amount, Currency, GeoLocation
│   └── rules.py              # AML business rules
│
├── application/              # Use cases (orchestration)
│   ├── use_cases.py          # DetectAMLUseCase, ScoreTransactionUseCase
│   ├── dto.py                # Input/output data contracts
│   └── interfaces.py         # Ports for repositories and services
│
├── features/                 # Feature engineering
│   ├── json_extractor.py
│   ├── dtype_downcasting.py
│   └── feature_pipeline.py
│
├── ml/                       # ML models & inference
│   ├── model_loader.py
│   ├── inference.py
│   ├── training.py
│   └── validation.py
│
├── api/                      # FastAPI controllers
│   ├── routes.py
│   ├── schemas.py
│   └── dependencies.py
│
├── infrastructure/           # External systems
│   ├── repositories.py       # PostgreSQL / Feature Store
│   ├── kafka.py              # Stream ingestion
│   ├── mlflow.py             # Experiment tracking
│   └── monitoring.py         # Prometheus / Evidently
│
├── config/                   # Application configuration
│   └── settings.py

```

Outside `src/`:

```

data/          # Raw and processed datasets (NO CODE)
notebooks/     # Research and exploration
tests/         # Unit, integration, and load tests
docs/          # Project documentation

```

---

## 5. Layer Responsibilities

### 5.1 Domain Layer
The **Domain** layer contains **pure business logic** and **entities**.

Responsibilities:
- Define AML entities (Transaction, Customer)
- Define AML rules
- No framework imports (no pandas, no fastapi, no sql)
- 100% unit-testable

Example:
- Fraud thresholds
- Temporal transaction patterns
- Rule-based typology detection

---

### 5.2 Application Layer
The **Application** layer coordinates use-cases.

Responsibilities:
- Orchestrates business workflows
- Applies feature engineering
- Calls ML inference
- Implements use cases:
  - `DetectAMLUseCase`
  - `ScoreTransactionUseCase`

Important:
- Depends only on **Domain**
- Talks to Infrastructure **only via interfaces**

---

### 5.3 Feature Engineering Layer
The **Features** layer prepares transaction data for ML.

Responsibilities:
- JSON flattening
- Type downcasting
- Feature normalization
- Risk indicator extraction

This layer is:
- Shared between offline training & real-time inference
- Independent from ML framework

---

### 5.4 ML Layer
The **ML** layer handles everything related to models.

Responsibilities:
- Model training
- Model versioning
- Feature validation
- Real-time inference
- Threshold calibration

Examples:
- CatBoost / LightGBM scoring
- Probability calibration
- Risk ranking

---

### 5.5 Infrastructure Layer
The **Infrastructure** layer implements all external dependencies.

Responsibilities:
- Database repositories
- Kafka producers & consumers
- MLflow tracking
- Monitoring exporters

Replaceable components:
- PostgreSQL → ClickHouse
- Kafka → Pulsar
- MLflow → other tracking system

---

### 5.6 API Layer
The **API** layer is responsible for communication with clients.

Responsibilities:
- HTTP validation
- Authentication (future)
- Routing
- Request/response mapping
- Dependency injection

Example flow:
`POST /score → Validate → UseCase → Domain → ML → Response`

---

## 6. Data Flow (Real-Time Scoring)

```

1. Transaction arrives via API or Kafka
2. JSON is validated (API Layer)
3. Features are extracted (Features Layer)
4. UseCase is executed (Application Layer)
5. Domain rules applied (Domain Layer)
6. ML model calculates risk (ML Layer)
7. Result is stored (Infrastructure)
8. Metrics are emitted (Monitoring)

```

---

## 7. Offline Training Pipeline

```

Raw Data → Feature Engineering → Train Model
→ Validation → MLflow Registry → Deployment

```

Training and inference **share the same feature pipeline**.

---

## 8. Dependency Inversion Strategy

The system follows **Dependency Inversion Principle**:

- Application depends on **interfaces**
- Infrastructure provides **implementations**
- Implementation is injected at runtime

Example:
- `TransactionRepository` (interface)
- `PostgresTransactionRepository` (implementation)

---

## 9. Configuration Management

- All settings are loaded using **Pydantic Settings**
- No hardcoded secrets
- Environment variables only
- `.env` used for local development

---

## 10. Monitoring & Observability

- **MLflow** — experiment tracking
- **Prometheus** — metrics
- **Evidently** — data drift & model drift
- **Structured logging** — JSON logs

Key metrics:
- Fraud detection rate
- False Positive Rate (FPR)
- Inference latency
- Kafka consumer lag
- Feature drift

---

## 11. Testing Strategy

- **Unit Tests** → Domain & Features
- **Integration Tests** → API + DB + Kafka
- **ML Tests** → Data validation, inference reproducibility
- **Load Tests** → Throughput & latency benchmarks

---

## 12. Deployment Model

- Package-based deployment (`pip install`)
- Docker containerization
- Kubernetes-ready
- Stateless API
- Externalized storage

---

## 13. Key Architectural Principles

- Clean separation of concerns
- Testability at every layer
- Replaceable infrastructure
- Single source of truth for features
- Deterministic ML inference
- Auditability and traceability

---

## 14. Future Extensions

- Graph-based AML detection
- Online learning
- Feature store integration
- Real-time alerting dashboard
- Human-in-the-loop review system

---

## 15. Summary

The AML architecture is designed for:

- **High throughput**
- **Regulatory compliance**
- **Model explainability**
- **Safe ML deployment**
- **Long-term scalability**

The system strictly follows **Clean Architecture** with strong isolation between business logic, ML models, and infrastructure dependencies.
