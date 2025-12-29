"""
API Gateway (BFF - Backend for Frontend)
Aggregates data from all services and serves to the frontend
Uses REAL data from Groww API - NO MOCK DATA
"""
import asyncio
import logging
import os
import sys
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import httpx

# Add libs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indian AI Trader - API Gateway",
    description="Unified API for the Indian AI Trader platform",
    version="1.0.0"
)


# ============================================
# CONFIGURATION
# ============================================

class Config:
    GROWW_API_KEY = os.getenv("GROWW_API_KEY", "")
    GROWW_ACCESS_TOKEN = os.getenv("GROWW_ACCESS_TOKEN", "")
    
    # Service URLs
    INSTRUMENT_SERVICE = os.getenv("INSTRUMENT_SERVICE_URL", "http://localhost:8001")
    MARKET_INGESTOR = os.getenv("MARKET_INGESTOR_URL", "http://localhost:8002")
    FEATURE_SERVICE = os.getenv("FEATURE_SERVICE_URL", "http://localhost:8004")
    SIGNAL_SERVICE = os.getenv("SIGNAL_SERVICE_URL", "http://localhost:8005")
    BACKTEST_SERVICE = os.getenv("BACKTEST_SERVICE_URL", "http://localhost:8006")
    PORTFOLIO_SERVICE = os.getenv("PORTFOLIO_SERVICE_URL", "http://localhost:8007")
    ORDER_SERVICE = os.getenv("ORDER_SERVICE_URL", "http://localhost:8009")


config = Config()


# ============================================
# GROWW API CLIENT
# ============================================

class GrowwDataProvider:
    """Real-time data provider using Groww API"""
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.base_url = "https://api.groww.in"
        self._http_client: Optional[httpx.AsyncClient] = None
        self._instruments_cache: Dict[str, dict] = {}
        self._quotes_cache: Dict[str, dict] = {}
        self._cache_timestamp: Optional[datetime] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if not self._http_client:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json"
                }
            )
        return self._http_client
    
    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    async def get_instruments(self, exchange: str = "NSE") -> List[dict]:
        """Get instrument master from Groww"""
        try:
            from libs.groww_client.src.instruments import InstrumentDownloader
            
            downloader = InstrumentDownloader(self.api_key, self.access_token)
            instruments = await downloader.download()
            
            # Cache instruments
            for inst in instruments:
                key = f"{inst.get('exchange', 'NSE')}:{inst.get('tradingsymbol', '')}"
                self._instruments_cache[key] = inst
            
            return instruments
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            return []
    
    async def get_quote(self, exchange: str, symbol: str) -> Optional[dict]:
        """Get real-time quote from Groww"""
        try:
            from libs.groww_client.src.live_data import LiveDataClient
            
            client = LiveDataClient(self.api_key, self.access_token)
            quote = await client.get_quote(exchange, symbol)
            
            if quote:
                # Cache quote
                self._quotes_cache[f"{exchange}:{symbol}"] = {
                    **quote,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return quote
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None
    
    async def get_quotes_batch(self, instruments: List[Dict[str, str]]) -> Dict[str, dict]:
        """Get quotes for multiple instruments"""
        try:
            from libs.groww_client.src.live_data import LiveDataClient
            
            client = LiveDataClient(self.api_key, self.access_token)
            quotes = await client.get_ltp_batch(instruments)
            
            # Update cache
            for key, quote in quotes.items():
                self._quotes_cache[key] = {
                    **quote,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return quotes
        except Exception as e:
            logger.error(f"Failed to get batch quotes: {e}")
            return {}
    
    async def get_historical_candles(
        self,
        exchange: str,
        symbol: str,
        interval: str = "15minute",
        from_date: Optional[date] = None,
        to_date: Optional[date] = None
    ) -> List[dict]:
        """Get historical OHLCV data"""
        try:
            from libs.groww_client.src.historical import HistoricalDataClient
            
            client = HistoricalDataClient(self.api_key, self.access_token)
            candles = await client.get_candles(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                from_date=from_date,
                to_date=to_date
            )
            
            return candles
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    def get_cached_quote(self, exchange: str, symbol: str) -> Optional[dict]:
        """Get cached quote if available"""
        return self._quotes_cache.get(f"{exchange}:{symbol}")


# Global data provider
data_provider: Optional[GrowwDataProvider] = None


# ============================================
# MODELS
# ============================================

class QuoteResponse(BaseModel):
    symbol: str
    exchange: str
    name: Optional[str] = None
    ltp: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    change: float
    change_percent: float
    timestamp: str


class SignalResponse(BaseModel):
    id: str
    symbol: str
    exchange: str
    action: str
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    timeframe: str
    reason_codes: List[str]
    created_at: str


# ============================================
# STARTUP / SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup():
    global data_provider
    
    if config.GROWW_API_KEY and config.GROWW_ACCESS_TOKEN:
        data_provider = GrowwDataProvider(
            config.GROWW_API_KEY,
            config.GROWW_ACCESS_TOKEN
        )
        logger.info("Groww data provider initialized")
    else:
        logger.warning("Groww API credentials not configured - API calls will fail")


@app.on_event("shutdown")
async def shutdown():
    global data_provider
    if data_provider:
        await data_provider.close()


# ============================================
# HEALTH & STATUS
# ============================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "groww_configured": data_provider is not None,
        "version": "1.0.0"
    }


@app.get("/api/market/status")
async def get_market_status():
    """Get current market status"""
    now = datetime.now()
    
    # Indian market hours: 9:15 AM - 3:30 PM IST (Mon-Fri)
    is_weekday = now.weekday() < 5
    
    # Convert to IST (UTC+5:30)
    ist_hour = now.hour  # Assuming server is in IST
    ist_minute = now.minute
    
    market_open = (ist_hour == 9 and ist_minute >= 15) or (ist_hour > 9)
    market_close = (ist_hour == 15 and ist_minute <= 30) or (ist_hour < 15)
    
    is_open = is_weekday and market_open and market_close
    
    return {
        "exchange": "NSE",
        "status": "OPEN" if is_open else "CLOSED",
        "current_time": now.isoformat(),
        "market_open": "09:15",
        "market_close": "15:30",
        "is_trading_day": is_weekday,
        "message": "Market is open for trading" if is_open else "Market is closed"
    }


# ============================================
# INSTRUMENTS
# ============================================

@app.get("/api/instruments/search")
async def search_instruments(
    q: str = Query(..., min_length=1),
    exchange: str = "NSE",
    limit: int = Query(default=20, le=50)
):
    """Search instruments by symbol or name"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Groww API not configured")
    
    # Search from cache or fetch
    instruments = list(data_provider._instruments_cache.values())
    
    if not instruments:
        instruments = await data_provider.get_instruments(exchange)
    
    # Filter by query
    query = q.upper()
    results = []
    
    for inst in instruments:
        symbol = inst.get("tradingsymbol", "").upper()
        name = inst.get("name", "").upper()
        
        if query in symbol or query in name:
            results.append({
                "id": f"{inst.get('exchange', 'NSE')}:{inst.get('tradingsymbol', '')}",
                "exchange": inst.get("exchange", "NSE"),
                "segment": inst.get("segment", "CASH"),
                "trading_symbol": inst.get("tradingsymbol", ""),
                "name": inst.get("name", ""),
                "exchange_token": inst.get("exchange_token", ""),
                "lot_size": inst.get("lot_size", 1)
            })
        
        if len(results) >= limit:
            break
    
    return {"results": results, "count": len(results)}


@app.get("/api/instruments/{exchange}/{symbol}")
async def get_instrument(exchange: str, symbol: str):
    """Get instrument details"""
    key = f"{exchange.upper()}:{symbol.upper()}"
    
    if data_provider and key in data_provider._instruments_cache:
        return data_provider._instruments_cache[key]
    
    raise HTTPException(status_code=404, detail="Instrument not found")


# ============================================
# QUOTES (REAL-TIME DATA)
# ============================================

@app.get("/api/quotes")
async def get_quotes(symbols: str = Query(...)):
    """Get real-time quotes for multiple symbols"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Groww API not configured")
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    instruments = [
        {"exchange": "NSE", "trading_symbol": s}
        for s in symbol_list
    ]
    
    quotes = await data_provider.get_quotes_batch(instruments)
    
    results = []
    for symbol in symbol_list:
        key = f"NSE:{symbol}"
        if key in quotes:
            q = quotes[key]
            prev_close = q.get("close", q.get("ltp", 0))
            ltp = q.get("ltp", 0)
            change = ltp - prev_close if prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            results.append({
                "symbol": symbol,
                "exchange": "NSE",
                "name": q.get("name", symbol),
                "ltp": ltp,
                "open": q.get("open"),
                "high": q.get("high"),
                "low": q.get("low"),
                "close": prev_close,
                "volume": q.get("volume", 0),
                "change": round(change, 2),
                "change_percent": round(change_pct, 2)
            })
    
    return {"quotes": results, "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/quote/{exchange}/{symbol}")
async def get_quote(exchange: str, symbol: str):
    """Get detailed quote for a single symbol"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Groww API not configured")
    
    quote = await data_provider.get_quote(exchange.upper(), symbol.upper())
    
    if not quote:
        raise HTTPException(status_code=404, detail=f"No quote found for {symbol}")
    
    prev_close = quote.get("close", quote.get("ltp", 0))
    ltp = quote.get("ltp", 0)
    change = ltp - prev_close if prev_close else 0
    change_pct = (change / prev_close * 100) if prev_close else 0
    
    return {
        "symbol": symbol.upper(),
        "exchange": exchange.upper(),
        "name": quote.get("name", symbol),
        "ltp": ltp,
        "open": quote.get("open"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "close": prev_close,
        "volume": quote.get("volume", 0),
        "bid_price": quote.get("bid_price"),
        "ask_price": quote.get("ask_price"),
        "change": round(change, 2),
        "change_percent": round(change_pct, 2),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================
# HISTORICAL DATA
# ============================================

@app.get("/api/candles/{exchange}/{symbol}")
async def get_candles(
    exchange: str,
    symbol: str,
    interval: str = Query(default="15minute"),
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    limit: int = Query(default=500, le=5000)
):
    """Get historical OHLCV candles"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Groww API not configured")
    
    candles = await data_provider.get_historical_candles(
        exchange=exchange.upper(),
        symbol=symbol.upper(),
        interval=interval,
        from_date=from_date,
        to_date=to_date
    )
    
    return {
        "symbol": symbol.upper(),
        "exchange": exchange.upper(),
        "interval": interval,
        "candles": candles[:limit]
    }


# ============================================
# WATCHLIST
# ============================================

NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "BAJFINANCE", "WIPRO", "HCLTECH", "ULTRACEMCO"
]


@app.get("/api/watchlist")
async def get_watchlist(name: str = "nifty50"):
    """Get watchlist with real-time quotes"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Groww API not configured")
    
    symbols = NIFTY_50_SYMBOLS
    
    instruments = [{"exchange": "NSE", "trading_symbol": s} for s in symbols]
    quotes = await data_provider.get_quotes_batch(instruments)
    
    items = []
    for symbol in symbols:
        key = f"NSE:{symbol}"
        if key in quotes:
            q = quotes[key]
            prev_close = q.get("close", q.get("ltp", 0))
            ltp = q.get("ltp", 0)
            change = ltp - prev_close if prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            items.append({
                "symbol": symbol,
                "name": q.get("name", symbol),
                "ltp": ltp,
                "change": round(change, 2),
                "change_percent": round(change_pct, 2)
            })
    
    return {
        "name": "NIFTY 50",
        "items": items,
        "updated_at": datetime.utcnow().isoformat()
    }


# ============================================
# SIGNALS (from Signal Service)
# ============================================

@app.get("/api/signals/latest")
async def get_latest_signals(
    limit: int = Query(default=10, le=50),
    min_confidence: float = Query(default=0.5, ge=0, le=1)
):
    """Get latest AI trading signals"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{config.SIGNAL_SERVICE}/signals/latest",
                params={"limit": limit, "min_confidence": min_confidence}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Signal service error: {e}")
        raise HTTPException(status_code=503, detail="Signal service unavailable")


@app.get("/api/signals/{signal_id}")
async def get_signal(signal_id: str):
    """Get specific signal"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{config.SIGNAL_SERVICE}/signals/{signal_id}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Signal service error: {e}")
        raise HTTPException(status_code=503, detail="Signal service unavailable")


# ============================================
# PORTFOLIO (from Portfolio Service)
# ============================================

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio summary with real prices"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{config.PORTFOLIO_SERVICE}/portfolio")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Portfolio service error: {e}")
        raise HTTPException(status_code=503, detail="Portfolio service unavailable")


@app.get("/api/positions")
async def get_positions():
    """Get open positions"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{config.PORTFOLIO_SERVICE}/positions")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Portfolio service error: {e}")
        raise HTTPException(status_code=503, detail="Portfolio service unavailable")


# ============================================
# ORDERS (from Order Service)
# ============================================

@app.post("/api/orders")
async def place_order(request: dict):
    """Place a new order"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{config.ORDER_SERVICE}/orders",
                json=request
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Order service error: {e}")
        raise HTTPException(status_code=503, detail="Order service unavailable")


@app.get("/api/orders")
async def get_orders(status: Optional[str] = None, limit: int = 50):
    """Get orders"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{config.ORDER_SERVICE}/orders",
                params={"status": status, "limit": limit}
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Order service error: {e}")
        raise HTTPException(status_code=503, detail="Order service unavailable")


# ============================================
# SSE STREAMING (Real-time updates)
# ============================================

@app.get("/api/stream")
async def stream_updates():
    """Server-Sent Events for real-time price updates"""
    
    async def event_generator():
        """Generate SSE events with real data"""
        while True:
            try:
                if data_provider:
                    # Get real-time quotes
                    instruments = [
                        {"exchange": "NSE", "trading_symbol": s}
                        for s in NIFTY_50_SYMBOLS[:10]  # Stream top 10
                    ]
                    
                    quotes = await data_provider.get_quotes_batch(instruments)
                    
                    for symbol in NIFTY_50_SYMBOLS[:10]:
                        key = f"NSE:{symbol}"
                        if key in quotes:
                            q = quotes[key]
                            prev_close = q.get("close", q.get("ltp", 0))
                            ltp = q.get("ltp", 0)
                            
                            if ltp > 0:
                                import json
                                data = {
                                    "type": "price",
                                    "data": {
                                        "symbol": symbol,
                                        "ltp": ltp,
                                        "change": round(ltp - prev_close, 2) if prev_close else 0,
                                        "change_percent": round((ltp - prev_close) / prev_close * 100, 2) if prev_close else 0
                                    }
                                }
                                yield f"data: {json.dumps(data)}\n\n"
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"SSE error: {e}")
                await asyncio.sleep(5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================
# STATIC FILES (Frontend)
# ============================================

# Serve static files
STATIC_DIR = Path(__file__).parent.parent.parent / "web"

if STATIC_DIR.exists():
    app.mount("/css", StaticFiles(directory=STATIC_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=STATIC_DIR / "js"), name="js")
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
    app.mount("/pages", StaticFiles(directory=STATIC_DIR / "pages"), name="pages")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve main dashboard"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Indian AI Trader</h1><p>Frontend not found</p>")


@app.get("/watchlist", response_class=HTMLResponse)
async def serve_watchlist():
    watchlist_file = STATIC_DIR / "pages" / "watchlist.html"
    if watchlist_file.exists():
        return FileResponse(watchlist_file)
    raise HTTPException(status_code=404)


@app.get("/signals", response_class=HTMLResponse)
async def serve_signals():
    signals_file = STATIC_DIR / "pages" / "signals.html"
    if signals_file.exists():
        return FileResponse(signals_file)
    raise HTTPException(status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
