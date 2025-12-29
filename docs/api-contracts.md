# API Contracts

## Overview

All API endpoints are served through the API Gateway at `http://localhost:8000/api/`.

## Authentication

All authenticated endpoints require:

```
Authorization: Bearer <jwt_token>
```

## Common Response Format

### Success Response

```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2024-12-29T10:30:00Z"
}
```

### Error Response

```json
{
  "success": false,
  "error": "ERROR_CODE",
  "message": "Human readable message",
  "details": { ... },
  "timestamp": "2024-12-29T10:30:00Z"
}
```

---

## Endpoints

### Health Check

#### `GET /api/health`

Check API health status.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-12-29T10:30:00Z",
  "version": "1.0.0"
}
```

---

### Market Status

#### `GET /api/market/status`

Get current market status for NSE.

**Response:**

```json
{
  "exchange": "NSE",
  "status": "OPEN",
  "current_time": "2024-12-29T10:30:00+05:30",
  "market_open": "09:15",
  "market_close": "15:30",
  "is_trading_day": true
}
```

---

### Instruments

#### `GET /api/instruments/search`

Search instruments by symbol or name.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| q | string | Yes | Search query |
| exchange | string | No | Filter by exchange (NSE, BSE) |
| limit | integer | No | Max results (default: 20, max: 50) |

**Response:**

```json
{
  "results": [
    {
      "id": "NSE:RELIANCE",
      "exchange": "NSE",
      "segment": "CASH",
      "trading_symbol": "RELIANCE",
      "name": "Reliance Industries Ltd",
      "exchange_token": "2885"
    }
  ]
}
```

---

### Quotes

#### `GET /api/quotes`

Get quotes for multiple symbols.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| symbols | string | Yes | Comma-separated symbols |

**Example:** `/api/quotes?symbols=RELIANCE,TCS,INFY`

**Response:**

```json
{
  "quotes": [
    {
      "symbol": "RELIANCE",
      "name": "Reliance Industries",
      "ltp": 2456.75,
      "change": 1.25,
      "change_percent": 0.05
    }
  ],
  "timestamp": "2024-12-29T10:30:00Z"
}
```

#### `GET /api/quote/{exchange}/{symbol}`

Get detailed quote for a single symbol.

**Response:**

```json
{
  "symbol": "RELIANCE",
  "exchange": "NSE",
  "name": "Reliance Industries",
  "ltp": 2456.75,
  "open": 2443.5,
  "high": 2465.0,
  "low": 2438.2,
  "close": 2455.5,
  "volume": 1234567,
  "change": 1.25,
  "change_percent": 0.05,
  "timestamp": "2024-12-29T10:30:00Z"
}
```

---

### Watchlist

#### `GET /api/watchlist`

Get default watchlist (NIFTY 50).

**Response:**

```json
{
  "name": "NIFTY 50",
  "items": [
    {
      "symbol": "RELIANCE",
      "name": "Reliance Industries",
      "ltp": 2456.75,
      "change": 1.25,
      "change_percent": 0.05
    }
  ],
  "updated_at": "2024-12-29T10:30:00Z"
}
```

---

### Signals

#### `GET /api/signals/latest`

Get latest AI trading signals.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| limit | integer | No | Max results (default: 10, max: 50) |
| min_confidence | float | No | Min confidence (0-1, default: 0.5) |

**Response:**

```json
{
  "signals": [
    {
      "id": "sig_001",
      "symbol": "RELIANCE",
      "exchange": "NSE",
      "action": "LONG",
      "confidence": 0.78,
      "entry_price": 2450.0,
      "stop_loss": 2400.0,
      "target_1": 2520.0,
      "target_2": 2580.0,
      "timeframe": "1h",
      "reason_codes": ["RSI_OVERSOLD", "MACD_BULLISH"],
      "created_at": "2024-12-29T10:30:00Z"
    }
  ]
}
```

#### `GET /api/signals/{signal_id}`

Get specific signal by ID.

---

### Portfolio

#### `GET /api/portfolio`

Get portfolio overview.

**Response:**

```json
{
  "capital": 1000000,
  "invested": 450000,
  "available": 550000,
  "total_pnl": 23450,
  "total_pnl_percent": 5.21,
  "positions": [
    {
      "symbol": "RELIANCE",
      "qty": 50,
      "avg_price": 2400,
      "current_price": 2456.75,
      "pnl": 2837.5,
      "pnl_percent": 2.36
    }
  ]
}
```

---

### Streaming (SSE)

#### `GET /api/stream`

Server-Sent Events endpoint for real-time updates.

**Event Types:**

1. **Price Update**

```json
{
  "type": "price",
  "data": {
    "symbol": "RELIANCE",
    "ltp": 2456.75,
    "change": 1.25,
    "change_percent": 0.05
  }
}
```

2. **Signal Alert**

```json
{
  "type": "signal",
  "data": {
    "id": "sig_001",
    "symbol": "RELIANCE",
    "action": "LONG",
    "confidence": 0.78
  }
}
```

**JavaScript Example:**

```javascript
const eventSource = new EventSource("/api/stream");

eventSource.onmessage = function (event) {
  const data = JSON.parse(event.data);
  console.log("Received:", data);
};
```

---

## Error Codes

| Code                  | HTTP Status | Description           |
| --------------------- | ----------- | --------------------- |
| NOT_FOUND             | 404         | Resource not found    |
| VALIDATION_ERROR      | 422         | Invalid input         |
| RATE_LIMIT_EXCEEDED   | 429         | Too many requests     |
| AUTHENTICATION_FAILED | 401         | Invalid/expired token |
| AUTHORIZATION_FAILED  | 403         | Permission denied     |
| GROWW_API_ERROR       | 502         | Groww API failure     |
| MARKET_CLOSED         | 503         | Market is closed      |
| INTERNAL_ERROR        | 500         | Unexpected error      |

---

## Rate Limits

| Endpoint    | Limit        |
| ----------- | ------------ |
| General API | 60 req/min   |
| SSE Stream  | 1 connection |
| Search      | 30 req/min   |
