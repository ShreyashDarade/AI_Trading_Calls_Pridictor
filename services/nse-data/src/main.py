"""
NSE India Data Service
Free fallback data source using official NSE India website
Provides real-time quotes, historical data, and market info
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================
# NSE INDIA CLIENT
# ============================================

class NSEClient:
    """Client for fetching data from NSE India website"""
    
    BASE_URL = "https://www.nseindia.com"
    
    # Common NSE stocks for watchlist
    NIFTY_50_STOCKS = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
        "BAJFINANCE", "WIPRO", "ULTRACEMCO", "NESTLEIND", "TECHM"
    ]
    
    def __init__(self):
        self.session = None
        self._cookies = {}
        
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session with proper headers"""
        if self.session is None:
            self.session = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Referer": "https://www.nseindia.com/",
                    "Origin": "https://www.nseindia.com",
                },
                follow_redirects=True
            )
            # Initialize cookies by visiting homepage
            try:
                response = await self.session.get(f"{self.BASE_URL}/")
                self._cookies = dict(response.cookies)
            except Exception:
                pass
        return self.session
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    async def _fetch(self, endpoint: str) -> Dict[str, Any]:
        """Fetch data from NSE API endpoint"""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = await session.get(url, cookies=self._cookies)
            
            # Update cookies
            self._cookies.update(dict(response.cookies))
            
            if response.status_code == 200:
                return response.json()
            else:
                # Retry with fresh session
                await self.close()
                session = await self._get_session()
                response = await session.get(url, cookies=self._cookies)
                if response.status_code == 200:
                    return response.json()
                    
        except Exception as e:
            print(f"NSE fetch error for {endpoint}: {e}")
        
        return {}
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            data = await self._fetch(f"/api/quote-equity?symbol={symbol.upper()}")
            if data and "priceInfo" in data:
                price_info = data.get("priceInfo", {})
                info = data.get("info", {})
                return {
                    "symbol": symbol.upper(),
                    "name": info.get("companyName", symbol),
                    "ltp": price_info.get("lastPrice", 0),
                    "open": price_info.get("open", 0),
                    "high": price_info.get("intraDayHighLow", {}).get("max", 0),
                    "low": price_info.get("intraDayHighLow", {}).get("min", 0),
                    "close": price_info.get("previousClose", 0),
                    "change": price_info.get("change", 0),
                    "pChange": price_info.get("pChange", 0),
                    "volume": data.get("preOpenMarket", {}).get("totalTradedVolume", 0),
                    "timestamp": datetime.now().isoformat(),
                    "source": "NSE"
                }
        except Exception as e:
            print(f"Quote error for {symbol}: {e}")
        
        return {"symbol": symbol, "error": "Failed to fetch", "source": "NSE"}
    
    async def get_nifty50(self) -> List[Dict[str, Any]]:
        """Get NIFTY 50 index data with all constituents"""
        try:
            data = await self._fetch("/api/equity-stockIndices?index=NIFTY%2050")
            if data and "data" in data:
                stocks = []
                for item in data.get("data", []):
                    stocks.append({
                        "symbol": item.get("symbol", ""),
                        "name": item.get("meta", {}).get("companyName", item.get("symbol", "")),
                        "ltp": item.get("lastPrice", 0),
                        "open": item.get("open", 0),
                        "high": item.get("dayHigh", 0),
                        "low": item.get("dayLow", 0),
                        "close": item.get("previousClose", 0),
                        "change": item.get("change", 0),
                        "pChange": item.get("pctChange", 0),
                        "volume": item.get("totalTradedVolume", 0),
                        "source": "NSE"
                    })
                return stocks
        except Exception as e:
            print(f"NIFTY 50 error: {e}")
        
        return []
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get market status (open/closed)"""
        try:
            data = await self._fetch("/api/marketStatus")
            if data:
                market_state = data.get("marketState", [])
                for state in market_state:
                    if state.get("market") == "Capital Market - Normal":
                        return {
                            "is_open": state.get("marketStatus") == "Open",
                            "status": state.get("marketStatus", "Unknown"),
                            "timestamp": datetime.now().isoformat()
                        }
        except Exception as e:
            print(f"Market status error: {e}")
        
        # Default based on time
        now = datetime.now()
        is_weekday = now.weekday() < 5
        is_market_hours = 9 <= now.hour < 16 or (now.hour == 15 and now.minute <= 30)
        
        return {
            "is_open": is_weekday and is_market_hours,
            "status": "Open" if (is_weekday and is_market_hours) else "Closed",
            "timestamp": now.isoformat()
        }
    
    async def get_gainers_losers(self) -> Dict[str, List]:
        """Get top gainers and losers"""
        try:
            data = await self._fetch("/api/equity-stockIndices?index=NIFTY%2050")
            if data and "data" in data:
                stocks = sorted(data.get("data", []), key=lambda x: x.get("pctChange", 0), reverse=True)
                
                gainers = [
                    {
                        "symbol": s.get("symbol"),
                        "ltp": s.get("lastPrice"),
                        "change": s.get("change"),
                        "pChange": s.get("pctChange")
                    }
                    for s in stocks[:5] if s.get("pctChange", 0) > 0
                ]
                
                losers = [
                    {
                        "symbol": s.get("symbol"),
                        "ltp": s.get("lastPrice"),
                        "change": s.get("change"),
                        "pChange": s.get("pctChange")
                    }
                    for s in stocks[-5:] if s.get("pctChange", 0) < 0
                ]
                
                return {"gainers": gainers, "losers": list(reversed(losers))}
        except Exception as e:
            print(f"Gainers/Losers error: {e}")
        
        return {"gainers": [], "losers": []}
    
    async def search_stocks(self, query: str) -> List[Dict[str, str]]:
        """Search for stocks by name or symbol"""
        try:
            data = await self._fetch(f"/api/search/autocomplete?q={query}")
            if data and "symbols" in data:
                return [
                    {
                        "symbol": item.get("symbol", ""),
                        "name": item.get("symbol_info", ""),
                        "type": item.get("result_type", "equity")
                    }
                    for item in data.get("symbols", [])[:10]
                ]
        except Exception as e:
            print(f"Search error: {e}")
        
        return []
    
    async def get_historical_data(
        self, 
        symbol: str, 
        from_date: str = None, 
        to_date: str = None
    ) -> List[Dict[str, Any]]:
        """Get historical OHLCV data for a symbol"""
        try:
            # NSE provides limited historical data via web API
            # For more data, we'd need to use alternative sources
            data = await self._fetch(f"/api/historical/cm/equity?symbol={symbol.upper()}")
            if data and "data" in data:
                candles = []
                for item in data.get("data", []):
                    candles.append({
                        "date": item.get("CH_TIMESTAMP"),
                        "open": float(item.get("CH_OPENING_PRICE", 0)),
                        "high": float(item.get("CH_TRADE_HIGH_PRICE", 0)),
                        "low": float(item.get("CH_TRADE_LOW_PRICE", 0)),
                        "close": float(item.get("CH_CLOSING_PRICE", 0)),
                        "volume": int(item.get("CH_TOT_TRADED_QTY", 0)),
                    })
                return candles
        except Exception as e:
            print(f"Historical data error for {symbol}: {e}")
        
        return []
    
    async def get_indices(self) -> List[Dict[str, Any]]:
        """Get major indices data"""
        try:
            data = await self._fetch("/api/allIndices")
            if data and "data" in data:
                major_indices = ["NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY NEXT 50", "INDIA VIX"]
                indices = []
                for item in data.get("data", []):
                    if item.get("index") in major_indices:
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
            print(f"Indices error: {e}")
        
        return []


# ============================================
# FASTAPI APP
# ============================================

nse_client = NSEClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    yield
    await nse_client.close()


app = FastAPI(
    title="NSE India Data Service",
    description="Free data service using NSE India website",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "nse-data", "timestamp": datetime.now().isoformat()}


@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    """Get real-time quote for a symbol"""
    quote = await nse_client.get_quote(symbol)
    if "error" in quote:
        raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
    return quote


@app.get("/api/quotes")
async def get_quotes(symbols: str = Query(..., description="Comma-separated symbols")):
    """Get quotes for multiple symbols"""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    quotes = await asyncio.gather(*[nse_client.get_quote(s) for s in symbol_list])
    return {"quotes": quotes}


@app.get("/api/nifty50")
async def get_nifty50():
    """Get NIFTY 50 stocks with prices"""
    stocks = await nse_client.get_nifty50()
    return {"stocks": stocks, "count": len(stocks)}


@app.get("/api/market-status")
async def get_market_status():
    """Get current market status"""
    return await nse_client.get_market_status()


@app.get("/api/gainers-losers")
async def get_gainers_losers():
    """Get top gainers and losers"""
    return await nse_client.get_gainers_losers()


@app.get("/api/search")
async def search_stocks(q: str = Query(..., min_length=1)):
    """Search for stocks"""
    results = await nse_client.search_stocks(q)
    return {"results": results}


@app.get("/api/historical/{symbol}")
async def get_historical(
    symbol: str,
    from_date: str = None,
    to_date: str = None
):
    """Get historical data for a symbol"""
    data = await nse_client.get_historical_data(symbol, from_date, to_date)
    return {"symbol": symbol, "candles": data}


@app.get("/api/indices")
async def get_indices():
    """Get major indices"""
    indices = await nse_client.get_indices()
    return {"indices": indices}


@app.get("/api/watchlist")
async def get_watchlist():
    """Get default watchlist with prices"""
    default_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ITC", "ICICIBANK", "KOTAKBANK"]
    quotes = await asyncio.gather(*[nse_client.get_quote(s) for s in default_symbols])
    return {"watchlist": [q for q in quotes if "error" not in q]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
