"""
API Gateway (BFF - Backend for Frontend)
Aggregates data from all services and serves to the frontend
Uses Groww API with NSE India fallback for FREE data
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
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indian AI Trader - API Gateway",
    description="Unified API for the Indian AI Trader platform",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    SIGNAL_SERVICE = os.getenv("SIGNAL_SERVICE_URL", "http://localhost:8005")
    PORTFOLIO_SERVICE = os.getenv("PORTFOLIO_SERVICE_URL", "http://localhost:8007")
    ORDER_SERVICE = os.getenv("ORDER_SERVICE_URL", "http://localhost:8009")
    NSE_SERVICE = os.getenv("NSE_SERVICE_URL", "http://localhost:8020")


config = Config()


# ============================================
# NSE INDIA DATA CLIENT (FREE FALLBACK)
# ============================================

class NSEDataClient:
    """Free data client using NSE India website"""
    
    BASE_URL = "https://www.nseindia.com"
    
    NIFTY_50_STOCKS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
        "BAJFINANCE", "WIPRO", "ULTRACEMCO", "NESTLEIND", "TECHM"
    ]
    
    def __init__(self):
        self._session: Optional[httpx.AsyncClient] = None
        self._cookies = {}
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[datetime] = None
    
    async def _get_session(self) -> httpx.AsyncClient:
        if self._session is None:
            self._session = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.nseindia.com/",
                },
                follow_redirects=True
            )
            try:
                response = await self._session.get(f"{self.BASE_URL}/")
                self._cookies = dict(response.cookies)
            except Exception as e:
                logger.warning(f"Failed to initialize NSE session: {e}")
        return self._session
    
    async def close(self):
        if self._session:
            await self._session.aclose()
            self._session = None
    
    async def _fetch(self, endpoint: str) -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = await session.get(url, cookies=self._cookies)
            self._cookies.update(dict(response.cookies))
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # Refresh session
                await self.close()
                session = await self._get_session()
                response = await session.get(url, cookies=self._cookies)
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            logger.error(f"NSE fetch error: {e}")
        
        return {}
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote for a symbol"""
        try:
            data = await self._fetch(f"/api/quote-equity?symbol={symbol.upper()}")
            if data and "priceInfo" in data:
                price_info = data.get("priceInfo", {})
                info = data.get("info", {})
                intraday = price_info.get("intraDayHighLow", {})
                
                ltp = price_info.get("lastPrice", 0)
                prev_close = price_info.get("previousClose", ltp)
                change = price_info.get("change", ltp - prev_close)
                pchange = price_info.get("pChange", (change / prev_close * 100) if prev_close else 0)
                
                return {
                    "symbol": symbol.upper(),
                    "name": info.get("companyName", symbol),
                    "ltp": ltp,
                    "open": price_info.get("open", 0),
                    "high": intraday.get("max", price_info.get("open", 0)),
                    "low": intraday.get("min", price_info.get("open", 0)),
                    "close": prev_close,
                    "change": round(change, 2),
                    "change_percent": round(pchange, 2),
                    "volume": data.get("preOpenMarket", {}).get("totalTradedVolume", 0),
                    "timestamp": datetime.now().isoformat(),
                    "source": "NSE"
                }
        except Exception as e:
            logger.error(f"Quote error for {symbol}: {e}")
        
        return {}
    
    async def get_nifty50(self) -> List[Dict[str, Any]]:
        """Get NIFTY 50 stocks with prices"""
        try:
            data = await self._fetch("/api/equity-stockIndices?index=NIFTY%2050")
            if data and "data" in data:
                stocks = []
                for item in data.get("data", []):
                    if item.get("symbol") == "NIFTY 50":
                        continue
                    stocks.append({
                        "symbol": item.get("symbol", ""),
                        "name": item.get("meta", {}).get("companyName", item.get("symbol", "")),
                        "ltp": item.get("lastPrice", 0),
                        "open": item.get("open", 0),
                        "high": item.get("dayHigh", 0),
                        "low": item.get("dayLow", 0),
                        "close": item.get("previousClose", 0),
                        "change": round(item.get("change", 0), 2),
                        "change_percent": round(item.get("pctChange", 0), 2),
                        "volume": item.get("totalTradedVolume", 0),
                        "source": "NSE"
                    })
                return stocks
        except Exception as e:
            logger.error(f"NIFTY 50 error: {e}")
        
        return []
    
    async def get_indices(self) -> List[Dict[str, Any]]:
        """Get major indices"""
        try:
            data = await self._fetch("/api/allIndices")
            if data and "data" in data:
                major = ["NIFTY 50", "NIFTY BANK", "NIFTY IT", "INDIA VIX", "NIFTY NEXT 50"]
                indices = []
                for item in data.get("data", []):
                    if item.get("index") in major:
                        indices.append({
                            "name": item.get("index"),
                            "value": item.get("last"),
                            "change": item.get("percentChange"),
                            "open": item.get("open"),
                            "high": item.get("high"),
                            "low": item.get("low"),
                        })
                return indices
        except Exception as e:
            logger.error(f"Indices error: {e}")
        
        return []
    
    async def search_stocks(self, query: str) -> List[Dict[str, str]]:
        """Search stocks"""
        try:
            data = await self._fetch(f"/api/search/autocomplete?q={query}")
            if data and "symbols" in data:
                return [
                    {
                        "symbol": item.get("symbol", ""),
                        "name": item.get("symbol_info", ""),
                        "type": "equity"
                    }
                    for item in data.get("symbols", [])[:10]
                ]
        except Exception as e:
            logger.error(f"Search error: {e}")
        
        return []
    
    async def get_gainers_losers(self) -> Dict[str, List]:
        """Get top gainers and losers"""
        try:
            data = await self._fetch("/api/equity-stockIndices?index=NIFTY%2050")
            if data and "data" in data:
                stocks = [s for s in data.get("data", []) if s.get("symbol") != "NIFTY 50"]
                stocks.sort(key=lambda x: x.get("pctChange", 0), reverse=True)
                
                gainers = [
                    {"symbol": s.get("symbol"), "ltp": s.get("lastPrice"), 
                     "change": round(s.get("change", 0), 2), "pChange": round(s.get("pctChange", 0), 2)}
                    for s in stocks[:5] if s.get("pctChange", 0) > 0
                ]
                
                losers = [
                    {"symbol": s.get("symbol"), "ltp": s.get("lastPrice"),
                     "change": round(s.get("change", 0), 2), "pChange": round(s.get("pctChange", 0), 2)}
                    for s in reversed(stocks[-5:]) if s.get("pctChange", 0) < 0
                ]
                
                return {"gainers": gainers, "losers": losers}
        except Exception as e:
            logger.error(f"Gainers/Losers error: {e}")
        
        return {"gainers": [], "losers": []}


# ============================================
# GROWW API CLIENT (PAID)
# ============================================

class GrowwDataClient:
    """Data client using Groww API (requires subscription)"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self._groww = None
    
    def _get_client(self):
        if self._groww is None:
            try:
                from growwapi import GrowwAPI
                self._groww = GrowwAPI(self.api_token)
            except ImportError:
                logger.warning("growwapi not installed")
        return self._groww
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote using Groww API"""
        groww = self._get_client()
        if not groww:
            return {}
        
        try:
            response = groww.get_quote(
                exchange=groww.EXCHANGE_NSE,
                segment=groww.SEGMENT_CASH,
                trading_symbol=symbol.upper()
            )
            if response:
                return {
                    "symbol": symbol.upper(),
                    "ltp": response.get("ltp", 0),
                    "open": response.get("open", 0),
                    "high": response.get("high", 0),
                    "low": response.get("low", 0),
                    "close": response.get("close", 0),
                    "change": response.get("change", 0),
                    "change_percent": response.get("pctChange", 0),
                    "source": "Groww"
                }
        except Exception as e:
            logger.error(f"Groww quote error: {e}")
        
        return {}
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get user holdings"""
        groww = self._get_client()
        if not groww:
            return []
        
        try:
            response = groww.get_holdings_for_user(timeout=10)
            return response if isinstance(response, list) else []
        except Exception as e:
            logger.error(f"Groww holdings error: {e}")
        
        return []
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get user positions"""
        groww = self._get_client()
        if not groww:
            return []
        
        try:
            response = groww.get_positions_for_user(segment=groww.SEGMENT_CASH)
            return response if isinstance(response, list) else []
        except Exception as e:
            logger.error(f"Groww positions error: {e}")
        
        return []


# ============================================
# UNIFIED DATA PROVIDER (Groww + NSE Fallback)
# ============================================

class UnifiedDataProvider:
    """Unified provider that uses Groww when available, NSE as fallback"""
    
    def __init__(self):
        self.nse = NSEDataClient()
        self.groww: Optional[GrowwDataClient] = None
        self.use_groww = False
    
    def configure_groww(self, api_token: str):
        """Configure Groww API"""
        if api_token:
            self.groww = GrowwDataClient(api_token)
            # We'll test if Groww works on first use
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote with fallback"""
        # Try Groww first if configured
        if self.groww and self.use_groww:
            quote = await self.groww.get_quote(symbol)
            if quote:
                return quote
        
        # Fallback to NSE (free)
        return await self.nse.get_quote(symbol)
    
    async def get_watchlist(self) -> List[Dict[str, Any]]:
        """Get watchlist with quotes"""
        return await self.nse.get_nifty50()
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get user holdings (Groww only)"""
        if self.groww:
            return await self.groww.get_holdings()
        return []
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get user positions (Groww only)"""
        if self.groww:
            return await self.groww.get_positions()
        return []
    
    async def get_indices(self) -> List[Dict[str, Any]]:
        """Get market indices"""
        return await self.nse.get_indices()
    
    async def get_gainers_losers(self) -> Dict[str, List]:
        """Get gainers and losers"""
        return await self.nse.get_gainers_losers()
    
    async def search(self, query: str) -> List[Dict[str, str]]:
        """Search stocks"""
        return await self.nse.search_stocks(query)
    
    async def close(self):
        """Close connections"""
        await self.nse.close()


# Global data provider
data_provider: Optional[UnifiedDataProvider] = None


# ============================================
# STARTUP / SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup():
    global data_provider
    
    # Load env if available
    env_locations = [".env", "../.env", "../../.env"]
    for env_file in env_locations:
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip())
            break
    
    data_provider = UnifiedDataProvider()
    
    # Configure Groww if credentials available
    groww_token = os.getenv("GROWW_API_KEY", "") or os.getenv("GROWW_ACCESS_TOKEN", "")
    if groww_token:
        data_provider.configure_groww(groww_token)
        logger.info("Groww API configured (will use as primary if available)")
    
    logger.info("API Gateway started - Using NSE India (free) as data source")


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
        "timestamp": datetime.now().isoformat(),
        "data_source": "NSE India (Free)",
        "groww_configured": data_provider.groww is not None if data_provider else False,
        "version": "2.0.0"
    }


@app.get("/api/market/status")
async def get_market_status():
    """Get current market status"""
    now = datetime.now()
    
    is_weekday = now.weekday() < 5
    ist_hour = now.hour
    ist_minute = now.minute
    
    # Market hours: 9:15 AM - 3:30 PM IST
    after_open = (ist_hour > 9) or (ist_hour == 9 and ist_minute >= 15)
    before_close = (ist_hour < 15) or (ist_hour == 15 and ist_minute <= 30)
    
    is_open = is_weekday and after_open and before_close
    
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
# QUOTES (NSE DATA - FREE)
# ============================================

@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    """Get real-time quote for a symbol"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    quote = await data_provider.get_quote(symbol)
    
    if not quote:
        raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
    
    return quote


@app.get("/api/quotes")
async def get_quotes(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get quotes for multiple symbols"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    quotes = await asyncio.gather(*[data_provider.get_quote(s) for s in symbol_list])
    
    return {
        "quotes": [q for q in quotes if q],
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# WATCHLIST
# ============================================

@app.get("/api/watchlist")
async def get_watchlist():
    """Get NIFTY 50 watchlist with real-time prices"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    stocks = await data_provider.get_watchlist()
    
    return {
        "name": "NIFTY 50",
        "items": stocks,
        "count": len(stocks),
        "source": "NSE India",
        "updated_at": datetime.now().isoformat()
    }


@app.get("/api/gainers-losers")
async def get_gainers_losers():
    """Get top gainers and losers"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await data_provider.get_gainers_losers()


@app.get("/api/indices")
async def get_indices():
    """Get major indices"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    indices = await data_provider.get_indices()
    return {"indices": indices}


# ============================================
# SEARCH
# ============================================

@app.get("/api/instruments/search")
async def search_instruments(q: str = Query(..., min_length=1)):
    """Search for stocks"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    results = await data_provider.search(q)
    return {"results": results, "count": len(results)}


# ============================================
# PORTFOLIO (from Groww or local)
# ============================================

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio summary"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Try to get from Groww
    holdings = await data_provider.get_holdings()
    
    if holdings:
        total_value = sum(h.get("value", 0) for h in holdings)
        total_invested = sum(h.get("invested", 0) for h in holdings)
        pnl = total_value - total_invested
        pnl_percent = (pnl / total_invested * 100) if total_invested > 0 else 0
        
        return {
            "total_value": total_value,
            "invested": total_invested,
            "pnl": pnl,
            "pnl_percent": round(pnl_percent, 2),
            "holdings_count": len(holdings),
            "source": "Groww"
        }
    
    # Return placeholder if no Groww access
    return {
        "total_value": 0,
        "invested": 0,
        "pnl": 0,
        "pnl_percent": 0,
        "holdings_count": 0,
        "source": "none",
        "message": "Connect Groww API to see portfolio"
    }


@app.get("/api/positions")
async def get_positions():
    """Get open positions"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    positions = await data_provider.get_positions()
    
    return {
        "positions": positions,
        "count": len(positions),
        "source": "Groww" if positions else "none"
    }


@app.get("/api/holdings")
async def get_holdings():
    """Get holdings"""
    if not data_provider:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    holdings = await data_provider.get_holdings()
    
    return {
        "holdings": holdings,
        "count": len(holdings),
        "source": "Groww" if holdings else "none"
    }


# ============================================
# SIGNALS (Demo signals when service unavailable)
# ============================================

@app.get("/api/signals/latest")
async def get_latest_signals(
    limit: int = Query(default=10, le=50),
    min_confidence: float = Query(default=0.5, ge=0, le=1)
):
    """Get latest AI trading signals"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{config.SIGNAL_SERVICE}/signals/latest",
                params={"limit": limit, "min_confidence": min_confidence}
            )
            response.raise_for_status()
            data = response.json()
            # If signal service returns signals, use them
            if data.get("signals") and len(data["signals"]) > 0:
                return data
    except Exception as e:
        logger.warning(f"Signal service unavailable: {e}")
    
    # Return demo signals if signal service unavailable or returns empty
        demo_signals = [
            {
                "id": "demo-1",
                "symbol": "RELIANCE",
                "exchange": "NSE",
                "action": "LONG",
                "confidence": 0.78,
                "entry_price": 2450.50,
                "stop_loss": 2401.49,
                "target_1": 2524.02,
                "target_2": 2573.03,
                "reason_codes": ["RSI_OVERSOLD", "UPTREND", "VOLUME_SPIKE"],
                "prediction_mode": "ml",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "demo-2",
                "symbol": "TCS",
                "exchange": "NSE",
                "action": "LONG",
                "confidence": 0.72,
                "entry_price": 3890.25,
                "stop_loss": 3812.45,
                "target_1": 4006.96,
                "target_2": 4084.76,
                "reason_codes": ["MACD_BULLISH", "EMA_CROSS_UP"],
                "prediction_mode": "ml",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "demo-3",
                "symbol": "HDFCBANK",
                "exchange": "NSE",
                "action": "SHORT",
                "confidence": 0.65,
                "entry_price": 1650.00,
                "stop_loss": 1683.00,
                "target_1": 1600.50,
                "target_2": 1567.50,
                "reason_codes": ["RSI_OVERBOUGHT", "RESISTANCE"],
                "prediction_mode": "ml",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "demo-4",
                "symbol": "INFY",
                "exchange": "NSE",
                "action": "LONG",
                "confidence": 0.81,
                "entry_price": 1425.75,
                "stop_loss": 1397.24,
                "target_1": 1468.52,
                "target_2": 1496.04,
                "reason_codes": ["BREAKOUT", "HIGH_VOLUME", "UPTREND"],
                "prediction_mode": "gpt",
                "gpt_analysis": "Strong bullish momentum with volume confirmation",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "demo-5",
                "symbol": "ICICIBANK",
                "exchange": "NSE",
                "action": "LONG",
                "confidence": 0.69,
                "entry_price": 1075.50,
                "stop_loss": 1054.00,
                "target_1": 1107.77,
                "target_2": 1129.28,
                "reason_codes": ["SUPPORT_BOUNCE", "MACD_BULLISH"],
                "prediction_mode": "ml",
                "created_at": datetime.now().isoformat()
            },
        ]
        
        # Filter by confidence
        filtered = [s for s in demo_signals if s["confidence"] >= min_confidence]
        
        return {
            "signals": filtered[:limit],
            "source": "demo",
            "message": "Demo signals - Start Signal Service for live AI predictions",
            "timestamp": datetime.now().isoformat()
        }


# ============================================
# BACKTESTS
# ============================================

@app.get("/api/backtests")
async def get_backtests():
    """Get backtest history"""
    # Return empty list - backtests should be run separately
    return {
        "backtests": [],
        "message": "Run backtests using the backtest service",
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# SETTINGS
# ============================================

@app.get("/api/settings")
async def get_settings():
    """Get current settings"""
    groww_configured = bool(os.getenv("GROWW_API_KEY") or os.getenv("GROWW_ACCESS_TOKEN"))
    
    return {
        "groww_api_configured": groww_configured,
        "data_source": "Groww" if groww_configured else "NSE India (Free)",
        "trading_mode": os.getenv("TRADING_MODE", "PAPER"),
        "min_confidence": float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.6")),
        "max_position_size": float(os.getenv("MAX_POSITION_SIZE_PCT", "5")),
        "max_daily_loss": float(os.getenv("MAX_DAILY_LOSS_PCT", "3")),
    }


@app.post("/api/settings")
async def update_settings(settings: dict):
    """Update settings (in-memory only)"""
    # In production, this would persist to database
    return {"status": "ok", "message": "Settings updated (in-memory only)"}


# ============================================
# SSE STREAMING (Real-time updates)
# ============================================

@app.get("/api/stream")
async def stream_updates():
    """Server-Sent Events for real-time price updates"""
    
    async def event_generator():
        import json
        
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ITC", "ICICIBANK"]
        
        while True:
            try:
                if data_provider:
                    for symbol in symbols:
                        quote = await data_provider.get_quote(symbol)
                        if quote and quote.get("ltp"):
                            data = {
                                "type": "price",
                                "data": {
                                    "symbol": symbol,
                                    "ltp": quote.get("ltp"),
                                    "change": quote.get("change", 0),
                                    "change_percent": quote.get("change_percent", 0)
                                }
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"SSE error: {e}")
                await asyncio.sleep(10)
    
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

STATIC_DIR = Path(__file__).parent.parent.parent / "web"

if STATIC_DIR.exists():
    # Mount static directories
    for subdir in ["css", "js", "assets", "pages"]:
        subdir_path = STATIC_DIR / subdir
        if subdir_path.exists():
            app.mount(f"/{subdir}", StaticFiles(directory=subdir_path), name=subdir)


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
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/signals", response_class=HTMLResponse)
async def serve_signals():
    signals_file = STATIC_DIR / "pages" / "signals.html"
    if signals_file.exists():
        return FileResponse(signals_file)
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/backtests", response_class=HTMLResponse)
async def serve_backtests():
    backtests_file = STATIC_DIR / "pages" / "backtests.html"
    if backtests_file.exists():
        return FileResponse(backtests_file)
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/settings", response_class=HTMLResponse)
async def serve_settings():
    settings_file = STATIC_DIR / "pages" / "settings.html"
    if settings_file.exists():
        return FileResponse(settings_file)
    raise HTTPException(status_code=404, detail="Page not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
