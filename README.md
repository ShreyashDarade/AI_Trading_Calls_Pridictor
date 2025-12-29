# 🚀 Indian AI Trader

**AI-Powered Trading Platform for Indian Stock Markets**

An intelligent, microservices-based trading platform that integrates with the Groww API to provide real-time market data, AI-generated trading signals, automated order execution, and comprehensive backtesting capabilities.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running Services](#running-services)
- [API Documentation](#api-documentation)
- [Frontend](#frontend)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Indian AI Trader is a production-grade trading platform designed specifically for the Indian stock market (NSE/BSE). It combines:

- **Real-time market data** from Groww's trading API
- **AI/ML-powered signal generation** using technical indicators
- **Automated order execution** (paper trading & live trading)
- **Walk-forward backtesting** for strategy validation
- **Beautiful web dashboard** for monitoring and control

### Why This Project?

| Problem                                 | Solution                                                 |
| --------------------------------------- | -------------------------------------------------------- |
| Manual trading is time-consuming        | Automated AI signals with one-click execution            |
| Retail traders lack institutional tools | Professional-grade technical analysis & risk management  |
| No Indian-focused AI trading platforms  | Built specifically for NSE/BSE with IST timezone support |
| Expensive trading software              | Open-source, self-hosted, free forever                   |

---

## ✨ Features

### 📊 Market Data

- Real-time quotes via Groww WebSocket feed
- Historical OHLCV data (1m, 5m, 15m, 1h, 1d)
- Instrument master with 5000+ symbols
- Live tick streaming with Server-Sent Events (SSE)

### 🤖 AI Signal Generation

- ML models (XGBoost, LightGBM, Random Forest)
- 25+ technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- Confidence scoring (0-100%)
- Entry price, stop-loss, and target levels
- Explainable reason codes

### 📈 Trading

- Paper trading (simulated execution)
- Live trading via Groww API
- Multiple order types (Market, Limit, Stop-Loss)
- Position tracking with real-time P&L
- Intraday (MIS) and delivery (CNC) support

### 🔄 Backtesting

- Walk-forward analysis
- Equity curve visualization
- Performance metrics (Sharpe, Sortino, Max Drawdown)
- Trade-by-trade logs

### 🛡️ Risk Management

- Position sizing algorithms (Fixed %, Kelly Criterion, ATR-based)
- Maximum exposure limits
- Daily loss limits
- Drawdown protection

### 🔔 Notifications

- Email alerts
- Slack integration
- Webhook support

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│                    (HTML/CSS/JavaScript)                         │
│                   Dashboard • Watchlist • Signals                │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP/SSE
┌─────────────────────────▼───────────────────────────────────────┐
│                      API GATEWAY                                 │
│                    (FastAPI - BFF)                               │
│              Aggregation • Auth • Rate Limiting                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Signal      │ │   Portfolio   │ │    Order      │
│   Service     │ │   Service     │ │   Service     │
└───────┬───────┘ └───────────────┘ └───────────────┘
        │
        ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Feature     │ │   Market      │ │   Backtest    │
│   Service     │ │   Ingestor    │ │   Service     │
└───────────────┘ └───────┬───────┘ └───────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Groww API  │
                   └─────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│   PostgreSQL (transactional) • ClickHouse (time-series)         │
│   Redis (cache) • Redpanda (event streaming)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
indian-ai-trader/
├── apps/
│   ├── api-gateway/           # Unified API gateway (FastAPI)
│   │   └── src/main.py
│   └── web/                   # Frontend dashboard
│       ├── index.html         # Main dashboard
│       ├── css/               # Stylesheets
│       ├── js/                # JavaScript
│       └── pages/             # Additional pages
│
├── services/
│   ├── instrument-service/    # Symbol/token mapping
│   ├── market-ingestor/       # Real-time data ingestion
│   ├── bar-aggregator/        # Tick → OHLCV aggregation
│   ├── market-store/          # ClickHouse storage
│   ├── feature-service/       # Technical indicators
│   ├── signal-service/        # AI signal generation
│   ├── backtest-service/      # Strategy backtesting
│   ├── portfolio-service/     # Position & P&L tracking
│   ├── order-service/         # Order execution
│   ├── notification-service/  # Alerts (email, Slack)
│   ├── audit-service/         # Decision logging
│   └── model-training/        # ML model training
│
├── libs/
│   ├── common/                # Shared utilities, schemas, config
│   ├── groww-client/          # Groww API client library
│   ├── indicators/            # Technical analysis indicators
│   └── risk/                  # Risk management algorithms
│
├── infra/
│   ├── docker/
│   │   ├── docker-compose.yml # Full infrastructure stack
│   │   └── init-scripts/      # Database initialization
│   └── observability/
│       └── prometheus.yml     # Metrics configuration
│
├── docs/
│   ├── architecture.md        # System design documentation
│   └── api-contracts.md       # API endpoint specifications
│
├── scripts/
│   └── run_all.py             # Start all services locally
│
├── .env.example               # Environment variables template
├── pyproject.toml             # Python project configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 📋 Prerequisites

### Required

- **Python 3.11+**
- **Docker & Docker Compose** (for infrastructure)
- **Groww Trading Account** with API access

### Recommended

- **8GB+ RAM** (for running all services + databases)
- **SSD storage** (for ClickHouse performance)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/indian-ai-trader.git
cd indian-ai-trader
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# REQUIRED: Groww API credentials
GROWW_API_KEY=your_api_key_here
GROWW_ACCESS_TOKEN=your_access_token_here

# Database (Docker defaults)
POSTGRES_PASSWORD=your_secure_password
CLICKHOUSE_PASSWORD=your_secure_password
REDIS_PASSWORD=your_secure_password
```

### 3. Start Infrastructure

```bash
cd infra/docker
docker-compose up -d
```

This starts:

- **PostgreSQL** (port 5432) - Transactional data
- **ClickHouse** (port 9000) - Time-series data
- **Redis** (port 6379) - Caching
- **Redpanda** (port 9092) - Event streaming
- **Prometheus** (port 9090) - Metrics
- **Grafana** (port 3000) - Dashboards

### 4. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 5. Run the API Gateway

```bash
cd apps/api-gateway
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the Dashboard

Navigate to: **http://localhost:8000**

---

## ⚙️ Configuration

### Environment Variables

| Variable                | Description                        | Default    |
| ----------------------- | ---------------------------------- | ---------- |
| `GROWW_API_KEY`         | Your Groww API key                 | (required) |
| `GROWW_ACCESS_TOKEN`    | Groww access token (refresh daily) | (required) |
| `POSTGRES_HOST`         | PostgreSQL hostname                | localhost  |
| `CLICKHOUSE_HOST`       | ClickHouse hostname                | localhost  |
| `REDIS_HOST`            | Redis hostname                     | localhost  |
| `TRADING_MODE`          | `PAPER` or `LIVE`                  | PAPER      |
| `MIN_SIGNAL_CONFIDENCE` | Minimum confidence for signals     | 0.6        |
| `MAX_POSITION_SIZE_PCT` | Max single position size (%)       | 5          |
| `MAX_DAILY_LOSS_PCT`    | Max daily loss before pause (%)    | 3          |

### Getting Groww API Credentials

1. Create a Groww trading account at [groww.in](https://groww.in)
2. Apply for API access at their developer portal
3. Generate API key and access token
4. Access token expires daily - refresh before market open

---

## 🖥️ Running Services

### Option 1: Run All Services (Development)

```bash
python scripts/run_all.py
```

This starts all services on their respective ports.

### Option 2: Run Individual Services

```bash
# API Gateway (port 8000)
cd apps/api-gateway && uvicorn src.main:app --reload --port 8000

# Signal Service (port 8005)
cd services/signal-service && uvicorn src.main:app --reload --port 8005

# Portfolio Service (port 8007)
cd services/portfolio-service && uvicorn src.main:app --reload --port 8007
```

### Service Ports

| Service              | Port | Health Check  |
| -------------------- | ---- | ------------- |
| API Gateway          | 8000 | `/api/health` |
| Instrument Service   | 8001 | `/health`     |
| Market Ingestor      | 8002 | `/health`     |
| Bar Aggregator       | 8003 | `/health`     |
| Feature Service      | 8004 | `/health`     |
| Signal Service       | 8005 | `/health`     |
| Backtest Service     | 8006 | `/health`     |
| Portfolio Service    | 8007 | `/health`     |
| Market Store         | 8008 | `/health`     |
| Order Service        | 8009 | `/health`     |
| Notification Service | 8010 | `/health`     |
| Audit Service        | 8011 | `/health`     |
| Model Training       | 8012 | `/health`     |

---

## � API Documentation

### Interactive Docs

Each service has auto-generated API docs:

- **API Gateway**: http://localhost:8000/docs
- **Signal Service**: http://localhost:8005/docs
- **Portfolio Service**: http://localhost:8007/docs

### Key Endpoints

```
# Market Data
GET  /api/quotes?symbols=RELIANCE,TCS,INFY
GET  /api/quote/NSE/RELIANCE
GET  /api/candles/NSE/RELIANCE?interval=15minute

# Signals
GET  /api/signals/latest?limit=10&min_confidence=0.7
POST /api/signals/generate {"symbol": "RELIANCE", "timeframe": "15m"}

# Portfolio
GET  /api/portfolio
GET  /api/positions
GET  /api/orders

# Orders (Paper Trading)
POST /api/orders {
  "symbol": "RELIANCE",
  "transaction_type": "BUY",
  "quantity": 10,
  "order_type": "MARKET",
  "mode": "PAPER"
}

# SSE Streaming
GET  /api/stream (Server-Sent Events for live prices)
```

---

## 🎨 Frontend

The web dashboard provides:

| Page      | URL                     | Features                               |
| --------- | ----------------------- | -------------------------------------- |
| Dashboard | `/`                     | Portfolio overview, signals, watchlist |
| Watchlist | `/pages/watchlist.html` | Real-time stock grid                   |
| Signals   | `/pages/signals.html`   | AI signal browser with filters         |
| Backtests | `/pages/backtests.html` | Run and analyze backtests              |
| Settings  | `/pages/settings.html`  | API config, risk settings              |

### Tech Stack

- **HTML5** - Semantic markup
- **CSS3** - Custom properties, glassmorphism, animations
- **Vanilla JavaScript** - No framework dependencies
- **Server-Sent Events** - Real-time updates

---

## 🗺️ Development Roadmap

### ✅ Phase 1: Foundation (Completed)

- [x] Microservices architecture
- [x] Groww API integration
- [x] Technical indicator library
- [x] Basic web dashboard
- [x] Paper trading support

### 🔄 Phase 2: Intelligence (In Progress)

- [ ] Train production ML models
- [ ] Feature store with historical data
- [ ] Sentiment analysis (news, social)
- [ ] Options trading support

### 📅 Phase 3: Production (Planned)

- [ ] Live trading with safety guards
- [ ] Mobile app (React Native)
- [ ] Multi-broker support (Zerodha, Upstox)
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline

### 🔮 Phase 4: Advanced (Future)

- [ ] Reinforcement learning strategies
- [ ] Portfolio optimization
- [ ] Social trading features
- [ ] Regulatory compliance (SEBI)

---

## 🛠️ What to Add

### Immediate Priorities

1. **Groww API Authentication**

   - Implement OAuth2 flow
   - Auto-refresh access token

2. **Database Migrations**

   - Alembic for PostgreSQL
   - Version control for schemas

3. **Model Training Pipeline**

   - Collect historical data
   - Feature engineering notebook
   - Train XGBoost/LightGBM models

4. **Testing**
   - Unit tests for indicators
   - Integration tests for services
   - End-to-end tests for trading flow

### Nice to Have

- Advanced charting (TradingView widget)
- Telegram bot for alerts
- Strategy builder (no-code)
- Performance analytics dashboard
- Export trades to Excel/CSV

---

## 👥 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
black .
isort .
mypy .

# Run tests
pytest
```

---

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**This software is for educational purposes only.**

- Trading in financial markets involves substantial risk of loss
- Past performance is not indicative of future results
- The developers are not liable for any financial losses
- Always consult a qualified financial advisor before trading
- This is NOT investment advice

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/indian-ai-trader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/indian-ai-trader/discussions)
- **Email**: support@example.com

---

<p align="center">
  <b>Built with ❤️ for the Indian Trading Community</b>
</p>
