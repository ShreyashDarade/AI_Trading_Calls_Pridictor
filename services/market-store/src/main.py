"""
Market Store Service
Stores and retrieves market data from ClickHouse
"""
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Market Store Service",
    description="Time-series storage for market data",
    version="1.0.0"
)


# ============================================
# MODELS
# ============================================

class Tick(BaseModel):
    exchange_token: str
    symbol: str
    exchange: str = "NSE"
    timestamp: datetime
    ltp: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    volume: int = 0


class Candle(BaseModel):
    exchange_token: str
    symbol: str
    exchange: str = "NSE"
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None


class CandleQuery(BaseModel):
    symbol: str
    exchange: str = "NSE"
    timeframe: str = "15m"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    limit: int = 500


# ============================================
# CLICKHOUSE CLIENT
# ============================================

class ClickHouseClient:
    """ClickHouse database client for market data"""
    
    def __init__(self):
        self.connected = False
        self._client = None
    
    async def connect(self, host: str = "localhost", port: int = 9000):
        """Connect to ClickHouse"""
        try:
            from clickhouse_driver import Client
            self._client = Client(host=host, port=port)
            
            # Test connection
            result = self._client.execute("SELECT 1")
            self.connected = True
            logger.info("Connected to ClickHouse")
            return True
        except Exception as e:
            logger.warning(f"ClickHouse connection failed: {e}")
            self.connected = False
            return False
    
    async def insert_ticks(self, ticks: List[Tick]) -> int:
        """Insert ticks into ClickHouse"""
        if not self.connected:
            logger.warning("ClickHouse not connected, skipping insert")
            return 0
        
        try:
            data = [
                (
                    t.exchange_token,
                    t.exchange,
                    t.symbol,
                    t.timestamp,
                    float(t.ltp),
                    float(t.bid_price) if t.bid_price else None,
                    float(t.ask_price) if t.ask_price else None,
                    0, 0,  # bid_qty, ask_qty
                    t.volume,
                    None  # open_interest
                )
                for t in ticks
            ]
            
            self._client.execute(
                """
                INSERT INTO market_data.ticks 
                (exchange_token, exchange, symbol, timestamp, ltp, 
                 bid_price, ask_price, bid_qty, ask_qty, volume, open_interest)
                VALUES
                """,
                data
            )
            return len(ticks)
        except Exception as e:
            logger.error(f"Failed to insert ticks: {e}")
            return 0
    
    async def insert_candles(self, candles: List[Candle]) -> int:
        """Insert candles into ClickHouse"""
        if not self.connected:
            return 0
        
        try:
            data = [
                (
                    c.exchange_token,
                    c.exchange,
                    c.symbol,
                    c.timeframe,
                    c.timestamp,
                    float(c.open),
                    float(c.high),
                    float(c.low),
                    float(c.close),
                    c.volume,
                    None,  # open_interest
                    None,  # trade_count
                    float(c.vwap) if c.vwap else None
                )
                for c in candles
            ]
            
            self._client.execute(
                """
                INSERT INTO market_data.candles 
                (exchange_token, exchange, symbol, timeframe, timestamp,
                 open, high, low, close, volume, open_interest, trade_count, vwap)
                VALUES
                """,
                data
            )
            return len(candles)
        except Exception as e:
            logger.error(f"Failed to insert candles: {e}")
            return 0
    
    async def get_candles(
        self,
        symbol: str,
        timeframe: str = "15m",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 500
    ) -> List[dict]:
        """Get candles from ClickHouse"""
        if not self.connected:
            return []
        
        try:
            start = start_date or (date.today() - timedelta(days=30))
            end = end_date or date.today()
            
            query = f"""
                SELECT 
                    exchange_token, symbol, timeframe, timestamp,
                    open, high, low, close, volume, vwap
                FROM market_data.candles
                WHERE symbol = %(symbol)s
                  AND timeframe = %(timeframe)s
                  AND timestamp >= %(start)s
                  AND timestamp <= %(end)s
                ORDER BY timestamp DESC
                LIMIT %(limit)s
            """
            
            result = self._client.execute(
                query,
                {
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "start": datetime.combine(start, datetime.min.time()),
                    "end": datetime.combine(end, datetime.max.time()),
                    "limit": limit
                }
            )
            
            candles = []
            for row in result:
                candles.append({
                    "exchange_token": row[0],
                    "symbol": row[1],
                    "timeframe": row[2],
                    "timestamp": row[3].isoformat(),
                    "open": float(row[4]),
                    "high": float(row[5]),
                    "low": float(row[6]),
                    "close": float(row[7]),
                    "volume": int(row[8]),
                    "vwap": float(row[9]) if row[9] else None
                })
            
            return candles
        except Exception as e:
            logger.error(f"Failed to get candles: {e}")
            return []
    
    async def get_latest_tick(self, symbol: str) -> Optional[dict]:
        """Get latest tick for a symbol"""
        if not self.connected:
            return None
        
        try:
            result = self._client.execute(
                """
                SELECT exchange_token, symbol, timestamp, ltp, volume
                FROM market_data.ticks
                WHERE symbol = %(symbol)s
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                {"symbol": symbol.upper()}
            )
            
            if result:
                row = result[0]
                return {
                    "exchange_token": row[0],
                    "symbol": row[1],
                    "timestamp": row[2].isoformat(),
                    "ltp": float(row[3]),
                    "volume": int(row[4])
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get latest tick: {e}")
            return None


# Global client
db = ClickHouseClient()


@app.on_event("startup")
async def startup():
    """Connect to ClickHouse on startup"""
    import os
    host = os.getenv("CLICKHOUSE_HOST", "localhost")
    port = int(os.getenv("CLICKHOUSE_PORT", "9000"))
    await db.connect(host, port)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "clickhouse_connected": db.connected
    }


@app.post("/ticks")
async def store_ticks(ticks: List[Tick]):
    """Store ticks in ClickHouse"""
    count = await db.insert_ticks(ticks)
    return {"stored": count}


@app.post("/candles")
async def store_candles(candles: List[Candle]):
    """Store candles in ClickHouse"""
    count = await db.insert_candles(candles)
    return {"stored": count}


@app.get("/candles/{symbol}")
async def get_candles(
    symbol: str,
    timeframe: str = Query(default="15m"),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(default=500, le=5000)
):
    """Get historical candles"""
    candles = await db.get_candles(symbol, timeframe, start_date, end_date, limit)
    return {"candles": candles}


@app.get("/ticks/{symbol}/latest")
async def get_latest_tick(symbol: str):
    """Get latest tick for symbol"""
    tick = await db.get_latest_tick(symbol)
    if not tick:
        raise HTTPException(status_code=404, detail="No tick found")
    return tick


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008, reload=True)
