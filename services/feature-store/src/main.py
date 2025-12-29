"""
Feature Store Service
Centralized storage and retrieval of ML features with historical data
"""
import asyncio
import logging
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Feature Store Service",
    description="Centralized feature storage and retrieval for ML",
    version="2.0.0"
)


# ============================================
# MODELS
# ============================================

class FeatureSet(BaseModel):
    """A collection of features for a symbol at a point in time"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    exchange: str = "NSE"
    timestamp: datetime
    timeframe: str = "15m"
    
    # Price features
    price: float
    returns_1: float
    returns_5: float
    returns_10: float
    returns_20: float
    
    # Momentum indicators
    rsi_7: Optional[float] = None
    rsi_14: Optional[float] = None
    rsi_21: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None
    roc: Optional[float] = None
    
    # Trend indicators
    sma_10: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_55: Optional[float] = None
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    supertrend: Optional[float] = None
    supertrend_direction: Optional[int] = None
    
    # Volatility indicators
    atr_14: Optional[float] = None
    atr_percent: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    bb_percent_b: Optional[float] = None
    keltner_upper: Optional[float] = None
    keltner_lower: Optional[float] = None
    historical_volatility: Optional[float] = None
    
    # Volume indicators
    volume: Optional[int] = None
    volume_sma_20: Optional[float] = None
    volume_ratio: Optional[float] = None
    obv: Optional[float] = None
    obv_change: Optional[float] = None
    vwap: Optional[float] = None
    mfi: Optional[float] = None
    
    # Price patterns
    higher_high: Optional[bool] = None
    lower_low: Optional[bool] = None
    inside_bar: Optional[bool] = None
    outside_bar: Optional[bool] = None
    gap_up: Optional[bool] = None
    gap_down: Optional[bool] = None
    
    # Derived features
    price_vs_sma20: Optional[float] = None
    price_vs_sma50: Optional[float] = None
    price_vs_sma200: Optional[float] = None
    sma_cross_20_50: Optional[int] = None
    ema_cross_9_21: Optional[int] = None
    volatility_regime: Optional[str] = None  # LOW, NORMAL, HIGH
    trend_strength: Optional[float] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureQuery(BaseModel):
    symbols: List[str]
    start_date: date
    end_date: date
    timeframe: str = "15m"
    features: Optional[List[str]] = None  # None = all features


class FeatureStats(BaseModel):
    feature_name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    missing_pct: float


# ============================================
# FEATURE STORE ENGINE
# ============================================

class FeatureStore:
    """
    In-memory feature store with ClickHouse persistence
    Designed for fast feature retrieval during inference
    """
    
    def __init__(self):
        # In-memory cache: {symbol: {timestamp: FeatureSet}}
        self.cache: Dict[str, Dict[datetime, FeatureSet]] = defaultdict(dict)
        self.stats: Dict[str, FeatureStats] = {}
        self._clickhouse_client = None
    
    async def connect_clickhouse(self, host: str, port: int):
        """Connect to ClickHouse for persistence"""
        try:
            from clickhouse_driver import Client
            self._clickhouse_client = Client(host=host, port=port)
            await self._ensure_tables()
            logger.info("Connected to ClickHouse feature store")
        except Exception as e:
            logger.warning(f"ClickHouse not available: {e}")
    
    async def _ensure_tables(self):
        """Create feature store tables if not exist"""
        if not self._clickhouse_client:
            return
        
        self._clickhouse_client.execute("""
            CREATE TABLE IF NOT EXISTS features.feature_store (
                id String,
                symbol String,
                exchange String,
                timestamp DateTime64(3, 'Asia/Kolkata'),
                timeframe String,
                price Float64,
                returns_1 Float64,
                returns_5 Float64,
                returns_10 Float64,
                returns_20 Float64,
                rsi_14 Nullable(Float64),
                macd Nullable(Float64),
                atr_14 Nullable(Float64),
                volume Nullable(UInt64),
                features_json String,
                created_at DateTime64(3) DEFAULT now64(3)
            )
            ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (symbol, timestamp)
        """)
    
    async def store_features(self, feature_set: FeatureSet) -> str:
        """Store a feature set"""
        # Store in cache
        self.cache[feature_set.symbol][feature_set.timestamp] = feature_set
        
        # Persist to ClickHouse
        if self._clickhouse_client:
            import json
            self._clickhouse_client.execute(
                """INSERT INTO features.feature_store 
                   (id, symbol, exchange, timestamp, timeframe, price, 
                    returns_1, returns_5, returns_10, returns_20,
                    rsi_14, macd, atr_14, volume, features_json)
                   VALUES""",
                [(
                    feature_set.id,
                    feature_set.symbol,
                    feature_set.exchange,
                    feature_set.timestamp,
                    feature_set.timeframe,
                    feature_set.price,
                    feature_set.returns_1,
                    feature_set.returns_5,
                    feature_set.returns_10,
                    feature_set.returns_20,
                    feature_set.rsi_14,
                    feature_set.macd,
                    feature_set.atr_14,
                    feature_set.volume,
                    json.dumps(feature_set.dict())
                )]
            )
        
        return feature_set.id
    
    async def get_features(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
        lookback: int = 1
    ) -> List[FeatureSet]:
        """Get features for a symbol"""
        if symbol not in self.cache:
            return []
        
        features = list(self.cache[symbol].values())
        features.sort(key=lambda x: x.timestamp, reverse=True)
        
        if timestamp:
            features = [f for f in features if f.timestamp <= timestamp]
        
        return features[:lookback]
    
    async def get_latest_features(self, symbols: List[str]) -> Dict[str, FeatureSet]:
        """Get latest features for multiple symbols"""
        result = {}
        for symbol in symbols:
            features = await self.get_features(symbol, lookback=1)
            if features:
                result[symbol] = features[0]
        return result
    
    async def get_historical_features(
        self,
        query: FeatureQuery
    ) -> Dict[str, List[FeatureSet]]:
        """Get historical features for training"""
        result = {}
        
        for symbol in query.symbols:
            if symbol not in self.cache:
                continue
            
            features = [
                f for f in self.cache[symbol].values()
                if query.start_date <= f.timestamp.date() <= query.end_date
            ]
            features.sort(key=lambda x: x.timestamp)
            result[symbol] = features
        
        return result
    
    async def compute_and_store(
        self,
        symbol: str,
        candles: List[dict],
        timeframe: str = "15m"
    ) -> FeatureSet:
        """Compute features from candles and store"""
        if len(candles) < 200:
            raise ValueError(f"Need at least 200 candles, got {len(candles)}")
        
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        volumes = np.array([c.get("volume", 0) for c in candles])
        
        latest = candles[-1]
        price = closes[-1]
        
        # Compute all features
        feature_set = FeatureSet(
            symbol=symbol,
            timestamp=latest.get("timestamp", datetime.utcnow()),
            timeframe=timeframe,
            price=price,
            returns_1=(closes[-1] / closes[-2] - 1) * 100,
            returns_5=(closes[-1] / closes[-6] - 1) * 100,
            returns_10=(closes[-1] / closes[-11] - 1) * 100,
            returns_20=(closes[-1] / closes[-21] - 1) * 100,
            
            # RSI
            rsi_7=self._compute_rsi(closes, 7),
            rsi_14=self._compute_rsi(closes, 14),
            rsi_21=self._compute_rsi(closes, 21),
            
            # MACD
            **self._compute_macd(closes),
            
            # Moving Averages
            sma_10=np.mean(closes[-10:]),
            sma_20=np.mean(closes[-20:]),
            sma_50=np.mean(closes[-50:]),
            sma_200=np.mean(closes[-200:]),
            ema_9=self._compute_ema(closes, 9),
            ema_21=self._compute_ema(closes, 21),
            
            # ATR
            atr_14=self._compute_atr(highs, lows, closes, 14),
            
            # Volume
            volume=int(volumes[-1]),
            volume_sma_20=np.mean(volumes[-20:]),
            volume_ratio=volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1,
            
            # Bollinger Bands
            **self._compute_bollinger(closes, 20, 2),
            
            # Derived
            price_vs_sma20=(price / np.mean(closes[-20:]) - 1) * 100,
            price_vs_sma50=(price / np.mean(closes[-50:]) - 1) * 100,
        )
        
        # Determine volatility regime
        atr_pct = feature_set.atr_14 / price * 100 if feature_set.atr_14 else 0
        if atr_pct < 1:
            feature_set.volatility_regime = "LOW"
        elif atr_pct > 3:
            feature_set.volatility_regime = "HIGH"
        else:
            feature_set.volatility_regime = "NORMAL"
        
        # Store
        await self.store_features(feature_set)
        
        return feature_set
    
    def _compute_rsi(self, prices: np.ndarray, period: int) -> float:
        deltas = np.diff(prices[-period-1:])
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _compute_ema(self, prices: np.ndarray, period: int) -> float:
        multiplier = 2 / (period + 1)
        ema = prices[-period]
        for price in prices[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def _compute_macd(self, prices: np.ndarray) -> dict:
        ema12 = self._compute_ema(prices, 12)
        ema26 = self._compute_ema(prices, 26)
        macd = ema12 - ema26
        return {
            "macd": macd,
            "macd_signal": macd * 0.9,  # Simplified
            "macd_histogram": macd * 0.1
        }
    
    def _compute_atr(self, highs, lows, closes, period: int) -> float:
        tr = []
        for i in range(-period, 0):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        return np.mean(tr)
    
    def _compute_bollinger(self, prices: np.ndarray, period: int, std_dev: float) -> dict:
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return {
            "bb_upper": upper,
            "bb_middle": sma,
            "bb_lower": lower,
            "bb_width": (upper - lower) / sma * 100,
            "bb_percent_b": (prices[-1] - lower) / (upper - lower) if upper != lower else 0.5
        }


# Global store
store = FeatureStore()


@app.on_event("startup")
async def startup():
    host = os.getenv("CLICKHOUSE_HOST", "localhost")
    port = int(os.getenv("CLICKHOUSE_PORT", "9000"))
    await store.connect_clickhouse(host, port)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cached_symbols": len(store.cache),
        "total_features": sum(len(v) for v in store.cache.values())
    }


@app.post("/features/compute")
async def compute_features(
    symbol: str,
    candles: List[dict],
    timeframe: str = "15m"
):
    """Compute and store features from candles"""
    feature_set = await store.compute_and_store(symbol, candles, timeframe)
    return feature_set


@app.get("/features/{symbol}/latest")
async def get_latest(symbol: str):
    """Get latest features for a symbol"""
    features = await store.get_features(symbol.upper(), lookback=1)
    if not features:
        raise HTTPException(status_code=404, detail="No features found")
    return features[0]


@app.get("/features/{symbol}/history")
async def get_history(
    symbol: str,
    lookback: int = Query(default=100, le=1000)
):
    """Get historical features for a symbol"""
    features = await store.get_features(symbol.upper(), lookback=lookback)
    return {"symbol": symbol, "features": features}


@app.post("/features/batch")
async def get_batch(symbols: List[str]):
    """Get latest features for multiple symbols"""
    features = await store.get_latest_features([s.upper() for s in symbols])
    return {"features": features}


@app.post("/features/historical")
async def get_historical(query: FeatureQuery):
    """Get historical features for training"""
    features = await store.get_historical_features(query)
    return {"features": features}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013, reload=True)
