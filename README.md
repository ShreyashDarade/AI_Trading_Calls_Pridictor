# 🇮🇳 Indian AI Trader

**AI-Powered Trading Platform for Indian Stock Markets**

An intelligent trading platform that provides real-time NSE market data, AI-generated trading signals, and comprehensive analysis tools. **Works for FREE with NSE India data** - no API subscription required!

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/indian-ai-trader.git
cd indian-ai-trader

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the platform
python scripts/run_all.py --api

# 4. Open in browser
# http://localhost:8000
```

**That's it!** No API key needed for basic functionality.

---

## 📋 Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Service Ports](#service-ports)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### 📊 Free Market Data (NSE India)

- **Real-time quotes** for NIFTY 50 stocks
- **Market indices** (NIFTY 50, BANK NIFTY, etc.)
- **Top gainers & losers**
- **Stock search** functionality
- **No API key required!**

### 🤖 AI Signal Generation

- **Two Prediction Modes:**
  - **ML Model** - XGBoost, LightGBM, Random Forest
  - **GPT Analysis** - OpenAI GPT-4 for advanced market analysis
- 25+ technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- Confidence scoring (0-100%)
- Entry price, stop-loss, and target levels
- Explainable reason codes

### 📈 Trading (Requires Groww API)

- Paper trading (simulated execution)
- Live trading via Groww API
- Multiple order types (Market, Limit, Stop-Loss)
- Position tracking with real-time P&L

### 🔄 Advanced Backtesting

- Interactive equity curve charts
- Monthly returns visualization
- Win/Loss distribution pie charts
- Drawdown analysis
- Trade-by-trade logs
- Support for both ML and GPT strategies

### 🌐 Modern Web Dashboard

- Dark theme with glassmorphism design
- Responsive layout (mobile-friendly)
- Real-time price updates via SSE
- Interactive charts

### 🔔 Notifications

- Email alerts
- Slack integration
- Webhook support

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Modern web browser

### Step-by-Step

```bash
# Clone the repo
git clone https://github.com/your-repo/indian-ai-trader.git
cd indian-ai-trader

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### Required Python Packages

```
fastapi>=0.100.0
uvicorn>=0.23.0
httpx>=0.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

---

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# For FREE NSE data (no API key needed)
# Just start the server!

# For Groww Trading API (₹499/month)
GROWW_API_KEY=your_jwt_token_here
GROWW_ACCESS_TOKEN=your_jwt_token_here

# Trading settings
TRADING_MODE=PAPER          # PAPER or LIVE
MAX_POSITION_SIZE_PCT=5     # Max 5% per position
MAX_DAILY_LOSS_PCT=3        # Stop at 3% daily loss
MIN_SIGNAL_CONFIDENCE=0.6   # Minimum 60% confidence
```

### Data Sources

| Source                   | Cost       | Features                            |
| ------------------------ | ---------- | ----------------------------------- |
| **NSE India** (Default)  | **FREE**   | Real-time quotes, NIFTY 50, indices |
| **Groww API** (Optional) | ₹499/month | Trading, portfolio, historical data |

---

## 🚀 Usage

### Start the Platform

```bash
# Start only API Gateway (recommended for most users)
python scripts/run_all.py --api

# Start all services
python scripts/run_all.py

# Start specific services
python scripts/run_all.py api-gateway signal-service
```

### Web Dashboard

Open in browser: **http://localhost:8000**

| Page      | URL          | Description                                |
| --------- | ------------ | ------------------------------------------ |
| Dashboard | `/`          | Main overview with portfolio and watchlist |
| Watchlist | `/watchlist` | NIFTY 50 stocks with real-time prices      |
| Signals   | `/signals`   | AI-generated trading signals               |
| Backtests | `/backtests` | Strategy backtesting                       |
| Settings  | `/settings`  | API configuration                          |

### API Endpoints

| Endpoint                  | Method | Description                 |
| ------------------------- | ------ | --------------------------- |
| `/api/health`             | GET    | Health check                |
| `/api/watchlist`          | GET    | NIFTY 50 stocks with prices |
| `/api/quote/{symbol}`     | GET    | Get quote for a symbol      |
| `/api/quotes?symbols=...` | GET    | Get multiple quotes         |
| `/api/indices`            | GET    | Major market indices        |
| `/api/gainers-losers`     | GET    | Top gainers and losers      |
| `/api/signals/latest`     | GET    | Latest AI signals           |
| `/api/market/status`      | GET    | Market open/closed status   |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WEB BROWSER                              │
│                 (Dashboard, Watchlist, Signals)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     API GATEWAY                               │
│                    (Port 8000)                                │
│  • Serves frontend                                            │
│  • Aggregates data from services                              │
│  • SSE streaming for real-time updates                        │
└─────────────────────────────────────────────────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  NSE DATA     │   │ SIGNAL        │   │ BACKTEST      │
│  (Free)       │   │ SERVICE       │   │ SERVICE       │
│  Port 8020    │   │ Port 8005     │   │ Port 8006     │
└───────────────┘   └───────────────┘   └───────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                   NSE INDIA WEBSITE                         │
│              (Free real-time market data)                   │
└───────────────────────────────────────────────────────────┘
```

---

## 🔌 Service Ports

| Service              | Port | Description              |
| -------------------- | ---- | ------------------------ |
| API Gateway          | 8000 | Main web server & API    |
| NSE Data             | 8020 | Free NSE market data     |
| Instrument Service   | 8001 | Stock instruments        |
| Market Ingestor      | 8002 | Real-time data ingestion |
| Feature Service      | 8004 | Technical indicators     |
| Signal Service       | 8005 | AI trading signals       |
| Backtest Service     | 8006 | Strategy backtesting     |
| Portfolio Service    | 8007 | Portfolio management     |
| Order Service        | 8008 | Order execution          |
| Notification Service | 8010 | Email/Slack alerts       |

---

## 📁 Project Structure

```
indian-ai-trader/
├── apps/
│   ├── api-gateway/          # Main API server & frontend
│   │   └── src/main.py
│   └── web/                  # Frontend HTML/CSS/JS
│       ├── index.html
│       ├── css/
│       ├── js/
│       └── pages/
├── services/
│   ├── nse-data/             # Free NSE data service
│   ├── signal-service/       # AI signal generation
│   ├── backtest-service/     # Strategy backtesting
│   └── ...
├── scripts/
│   ├── run_all.py            # Start all services
│   └── test_groww_api.py     # Test API credentials
├── .env.example              # Example configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🛠️ Development

### Running in Development Mode

```bash
# Start with auto-reload
cd apps/api-gateway
python -m uvicorn src.main:app --reload --port 8000
```

### Testing API Credentials

```bash
# Install growwapi package
pip install growwapi

# Test your Groww API credentials
python scripts/test_groww_api.py
```

### Adding a New Service

1. Create service directory: `services/my-service/src/`
2. Create `main.py` with FastAPI app
3. Create `__init__.py`
4. Add to `run_all.py` SERVICES dict
5. Run: `python scripts/run_all.py my-service`

---

## 🔐 Security Notes

- **NEVER commit `.env` file** - it's gitignored
- Use `.env.example` as template
- Generate strong JWT secrets for production
- Enable HTTPS in production
- Use paper trading mode first

---

## 🗺️ Roadmap

- [x] NSE India free data integration
- [x] NIFTY 50 watchlist
- [x] Real-time price updates (SSE)
- [x] Modern dark theme dashboard
- [x] Backtesting interface
- [ ] AI signal generation
- [ ] Groww API trading
- [ ] Portfolio tracking
- [ ] Mobile responsive improvements
- [ ] Push notifications
- [ ] Multi-broker support (Zerodha, Upstox)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading stocks involves significant risk of loss. The creators of this software are not responsible for any financial losses incurred from using this platform. Always do your own research before making trading decisions.

---

## 🙏 Acknowledgments

- [NSE India](https://www.nseindia.com/) for market data
- [FastAPI](https://fastapi.tiangolo.com/) for the awesome framework
- [Groww](https://groww.in/) for trading API

---

**Made with ❤️ for the Indian Trading Community**
