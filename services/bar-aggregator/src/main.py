"""
Bar Aggregator Service
Aggregates ticks into OHLCV candles at multiple timeframes
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
from decimal import Decimal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bar Aggregator Service",
    description="Aggregates ticks into OHLCV candles",
    version="1.0.0"
)


class Tick(BaseModel):
    exchange_token: str
    symbol: str
    timestamp: datetime
    ltp: float
    volume: int = 0


class Candle(BaseModel):
    exchange_token: str
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int = 0


class CandleBuffer:
    """Buffers ticks and aggregates into candles"""
    
    TIMEFRAMES = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
    }
    
    def __init__(self):
        # {symbol: {timeframe: {bar_timestamp: candle_data}}}
        self.buffers: Dict[str, Dict[str, Dict[datetime, dict]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.completed_candles: List[Candle] = []
    
    def _get_bar_timestamp(self, tick_time: datetime, timeframe: str) -> datetime:
        """Get the starting timestamp for a bar"""
        delta = self.TIMEFRAMES[timeframe]
        
        if timeframe == "1d":
            # Daily bars start at market open (9:15 IST)
            return tick_time.replace(hour=9, minute=15, second=0, microsecond=0)
        elif timeframe == "1h":
            return tick_time.replace(minute=0, second=0, microsecond=0)
        elif timeframe == "15m":
            minute = (tick_time.minute // 15) * 15
            return tick_time.replace(minute=minute, second=0, microsecond=0)
        elif timeframe == "5m":
            minute = (tick_time.minute // 5) * 5
            return tick_time.replace(minute=minute, second=0, microsecond=0)
        else:  # 1m
            return tick_time.replace(second=0, microsecond=0)
    
    def process_tick(self, tick: Tick) -> List[Candle]:
        """Process a tick and return any completed candles"""
        completed = []
        
        for timeframe in self.TIMEFRAMES.keys():
            bar_ts = self._get_bar_timestamp(tick.timestamp, timeframe)
            symbol_buffer = self.buffers[tick.symbol][timeframe]
            
            # Check if we need to close previous bar
            if symbol_buffer:
                prev_ts = max(symbol_buffer.keys())
                if bar_ts > prev_ts:
                    # Close previous bar
                    prev_bar = symbol_buffer[prev_ts]
                    completed.append(Candle(
                        exchange_token=tick.exchange_token,
                        symbol=tick.symbol,
                        timeframe=timeframe,
                        timestamp=prev_ts,
                        open=prev_bar["open"],
                        high=prev_bar["high"],
                        low=prev_bar["low"],
                        close=prev_bar["close"],
                        volume=prev_bar["volume"],
                        trade_count=prev_bar["count"]
                    ))
                    # Clear old bars
                    del symbol_buffer[prev_ts]
            
            # Update current bar
            if bar_ts not in symbol_buffer:
                symbol_buffer[bar_ts] = {
                    "open": tick.ltp,
                    "high": tick.ltp,
                    "low": tick.ltp,
                    "close": tick.ltp,
                    "volume": tick.volume,
                    "count": 1
                }
            else:
                bar = symbol_buffer[bar_ts]
                bar["high"] = max(bar["high"], tick.ltp)
                bar["low"] = min(bar["low"], tick.ltp)
                bar["close"] = tick.ltp
                bar["volume"] += tick.volume
                bar["count"] += 1
        
        self.completed_candles.extend(completed)
        return completed
    
    def get_current_bar(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Get the current incomplete bar"""
        if symbol in self.buffers and timeframe in self.buffers[symbol]:
            bars = self.buffers[symbol][timeframe]
            if bars:
                latest_ts = max(bars.keys())
                bar = bars[latest_ts]
                return {
                    "timestamp": latest_ts,
                    **bar
                }
        return None
    
    def flush_all(self) -> List[Candle]:
        """Flush all current bars as completed (e.g., end of day)"""
        completed = []
        
        for symbol, timeframes in self.buffers.items():
            for timeframe, bars in timeframes.items():
                for ts, bar in bars.items():
                    completed.append(Candle(
                        exchange_token="",  # Would need to track this
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=ts,
                        open=bar["open"],
                        high=bar["high"],
                        low=bar["low"],
                        close=bar["close"],
                        volume=bar["volume"],
                        trade_count=bar["count"]
                    ))
        
        self.buffers.clear()
        return completed


# Global buffer
buffer = CandleBuffer()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_symbols": len(buffer.buffers),
        "completed_candles": len(buffer.completed_candles)
    }


@app.post("/ticks")
async def process_tick(tick: Tick):
    """Process incoming tick"""
    completed = buffer.process_tick(tick)
    
    # TODO: Publish completed candles to Kafka
    # TODO: Store in ClickHouse
    
    return {
        "processed": True,
        "completed_candles": len(completed)
    }


@app.post("/ticks/batch")
async def process_ticks_batch(ticks: List[Tick]):
    """Process batch of ticks"""
    total_completed = 0
    
    for tick in ticks:
        completed = buffer.process_tick(tick)
        total_completed += len(completed)
    
    return {
        "processed": len(ticks),
        "completed_candles": total_completed
    }


@app.get("/bars/current/{symbol}/{timeframe}")
async def get_current_bar(symbol: str, timeframe: str):
    """Get current incomplete bar"""
    if timeframe not in CandleBuffer.TIMEFRAMES:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe: {timeframe}")
    
    bar = buffer.get_current_bar(symbol.upper(), timeframe)
    if not bar:
        raise HTTPException(status_code=404, detail="No current bar")
    
    return bar


@app.get("/bars/completed")
async def get_completed_bars(limit: int = 100):
    """Get recently completed candles"""
    return {
        "candles": buffer.completed_candles[-limit:]
    }


@app.post("/flush")
async def flush_bars():
    """Flush all current bars (end of day)"""
    completed = buffer.flush_all()
    return {
        "flushed": len(completed)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=True)
