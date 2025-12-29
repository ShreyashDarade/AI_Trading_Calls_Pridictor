# Architecture Overview

## System Design

The Indian AI Trader platform follows a **microservices architecture** designed for scalability, maintainability, and real-time processing of market data.

## Core Principles

1. **Security First**: All Groww API calls go through the backend; tokens never exposed to browser
2. **Event-Driven**: Services communicate via Kafka/Redpanda for loose coupling
3. **Real-time**: Sub-second latency for market data processing
4. **Reproducibility**: Immutable data snapshots for backtesting accuracy
5. **Observability**: Full tracing, metrics, and logging across all services

## Data Flow

```
┌─────────────────┐
│   Groww APIs    │
│  (REST + Feed)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              MARKET INGESTOR                    │
│  - Streaming feed subscription                  │
│  - REST fallback (Quote/LTP)                   │
│  - Tick normalization                          │
└────────┬───────────────────────────┬───────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐       ┌─────────────────────┐
│  Event Bus      │       │    Market Store     │
│  (Redpanda)     │ ◄───► │   (ClickHouse)      │
│                 │       │                     │
│ market.ticks.*  │       │ - Ticks (raw)       │
│ market.candles.*│       │ - Candles (OHLCV)   │
│ signals.*       │       │ - Snapshots         │
└────────┬────────┘       └─────────────────────┘
         │
    ┌────┴────┬─────────────┐
    │         │             │
    ▼         ▼             ▼
┌───────┐ ┌───────────┐ ┌───────────────┐
│ Bar   │ │ Feature   │ │ Signal        │
│ Aggr. │ │ Service   │ │ Service       │
└───┬───┘ └─────┬─────┘ └───────┬───────┘
    │           │               │
    │           └───────┬───────┘
    │                   │
    ▼                   ▼
┌─────────────────────────────────────────────────┐
│                 API GATEWAY (BFF)               │
│  - REST endpoints                               │
│  - SSE/WebSocket streaming                     │
│  - Authentication                              │
│  - Rate limiting                               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              FRONTEND (HTML/CSS/JS)             │
│  - Static pages                                │
│  - Real-time updates via SSE                   │
│  - No direct API access                        │
└─────────────────────────────────────────────────┘
```

## Service Descriptions

### Tier 1: Data Ingestion

| Service                | Responsibility               | Key Dependencies     |
| ---------------------- | ---------------------------- | -------------------- |
| **Instrument Service** | Symbol/token mapping, search | PostgreSQL           |
| **Market Ingestor**    | Live data from Groww         | Groww Feed, Redpanda |
| **Bar Aggregator**     | Ticks → OHLCV candles        | Redpanda             |
| **Market Store**       | Persistence layer            | ClickHouse           |

### Tier 2: Intelligence

| Service              | Responsibility               | Key Dependencies            |
| -------------------- | ---------------------------- | --------------------------- |
| **Feature Service**  | Online indicator computation | ClickHouse, Redis           |
| **Signal Service**   | ML scoring + rule gating     | Feature Service             |
| **Backtest Service** | Walk-forward backtesting     | ClickHouse, Historical Data |
| **Model Training**   | ML pipeline orchestration    | Airflow, ClickHouse         |

### Tier 3: Execution & Portfolio

| Service                  | Responsibility           | Key Dependencies  |
| ------------------------ | ------------------------ | ----------------- |
| **Portfolio Service**    | Positions, PnL tracking  | PostgreSQL, Redis |
| **Order Service**        | Paper/live order routing | Groww Order API   |
| **Audit Service**        | Immutable decision logs  | ClickHouse        |
| **Notification Service** | Alerts (email, slack)    | Redis queue       |

### Tier 4: Gateway

| Service         | Responsibility           | Key Dependencies |
| --------------- | ------------------------ | ---------------- |
| **API Gateway** | BFF, auth, rate limiting | All services     |

## Technology Stack

### Backend

- **Language**: Python 3.11+
- **Framework**: FastAPI (async, high-performance)
- **Task Queue**: Celery (optional, for heavy tasks)

### Databases

- **PostgreSQL**: Transactional data (users, orders, configs)
- **ClickHouse**: Time-series (ticks, candles, audit logs)
- **Redis**: Cache, sessions, rate limiting

### Messaging

- **Redpanda**: Kafka-compatible, simpler operations

### ML/AI

- **Training**: scikit-learn, XGBoost, LightGBM
- **Serving**: BentoML
- **Feature Store**: (future) Feast

### Observability

- **Tracing**: OpenTelemetry
- **Metrics**: Prometheus
- **Dashboards**: Grafana
- **Logs**: Structured logging (structlog)

## Security Considerations

1. **API Token Protection**

   - Never store tokens in frontend
   - Rotate credentials regularly
   - Use environment variables / secrets manager

2. **Rate Limiting**

   - Protect against abuse
   - Respect Groww API quotas

3. **Authentication**

   - JWT-based session management
   - Secure cookie handling

4. **Audit Trail**
   - Every signal/decision logged
   - Immutable audit packets

## Scaling Considerations

### Horizontal Scaling

- All services are stateless (can run multiple instances)
- Kafka consumer groups for parallel processing
- Redis cluster for distributed caching

### Vertical Scaling

- ClickHouse for high-throughput time-series
- Connection pooling for databases

## Deployment

### Local Development

```bash
docker-compose -f infra/docker/docker-compose.yml up -d
```

### Production (Kubernetes)

- Helm charts in `infra/k8s/`
- HPA for auto-scaling
- PodDisruptionBudgets for availability
