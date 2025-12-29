"""
Feature Service
Online feature computation for ML models
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import os
import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Feature Service",
    description="Online feature computation for ML models",
    version="1.0.0"
)


class FeatureRequest(BaseModel):
    instrument_id: str
    timeframe: str = "15m"


class FeatureResponse(BaseModel):
    instrument_id: str
    timeframe: str
    timestamp: datetime
    version: str
    features: Dict[str, Any]


class FeatureComputer:
    """
    Online Feature Computation Engine
    
    Computes technical indicators and derived features
    from market data for ML model scoring.
    """
    
    VERSION = "v1"
    
    def __init__(self):
        # Cache for computed features
        self.cache: Dict[str, FeatureResponse] = {}
    
    async def compute_features(
        self,
        instrument_id: str,
        timeframe: str = "15m"
    ) -> FeatureResponse:
        """
        Compute feature vector for an instrument
        
        Features computed:
        - Price returns (1, 5, 20 periods)
        - Volatility (20-period)
        - RSI (14-period)
        - MACD (12, 26, 9)
        - ATR (14-period)
        - Volume metrics
        - Trend regime
        """
        # Get candles from ClickHouse
        candles = await self._get_candles(instrument_id, timeframe, periods=50)
        
        if not candles:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for {instrument_id}"
            )
        
        # Compute features
        features = await self._compute_all_features(candles)
        
        response = FeatureResponse(
            instrument_id=instrument_id,
            timeframe=timeframe,
            timestamp=datetime.utcnow(),
            version=self.VERSION,
            features=features
        )
        
        # Cache
        cache_key = f"{instrument_id}:{timeframe}"
        self.cache[cache_key] = response
        
        return response
    
    async def _get_candles(
        self,
        instrument_id: str,
        timeframe: str,
        periods: int
    ) -> List[Dict]:
        """Get candles from market store"""
        symbol = instrument_id.split(":")[-1].upper()
        nse_url = os.getenv("NSE_SERVICE_URL", "http://localhost:8020").rstrip("/")

        if timeframe not in {"1d"}:
            raise HTTPException(
                status_code=400,
                detail="Only timeframe=1d is supported until ClickHouse ingestion is implemented"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{nse_url}/api/historical/{symbol}")
                response.raise_for_status()
                payload = response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to fetch historical data for {symbol}: {e}")

        candles_raw = payload.get("candles") or []
        if not candles_raw:
            return []

        candles = [
            {
                "open": float(c.get("open", 0) or 0),
                "high": float(c.get("high", 0) or 0),
                "low": float(c.get("low", 0) or 0),
                "close": float(c.get("close", 0) or 0),
                "volume": int(c.get("volume", 0) or 0),
            }
            for c in candles_raw
            if c.get("close") is not None
        ]

        return candles[-periods:]
    
    async def _compute_all_features(self, candles: List[Dict]) -> Dict[str, Any]:
        """Compute all features from candles"""
        import numpy as np
        
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        volumes = np.array([c["volume"] for c in candles])
        
        # Returns
        returns_1 = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
        returns_5 = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
        returns_20 = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
        
        # Volatility (20-period)
        if len(closes) >= 20:
            log_returns = np.log(closes[-20:] / np.roll(closes[-20:], 1))[1:]
            volatility_20 = np.std(log_returns) * np.sqrt(252) * 100
        else:
            volatility_20 = 0
        
        # RSI
        from libs.indicators.src.momentum import rsi
        rsi_values = rsi(closes, 14)
        rsi_14 = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50
        
        # MACD
        from libs.indicators.src.momentum import macd
        macd_line, signal_line, histogram = macd(closes)
        macd_val = macd_line[-1] if not np.isnan(macd_line[-1]) else 0
        macd_signal = signal_line[-1] if not np.isnan(signal_line[-1]) else 0
        macd_hist = histogram[-1] if not np.isnan(histogram[-1]) else 0
        
        # ATR
        from libs.indicators.src.volatility import atr
        atr_values = atr(highs, lows, closes, 14)
        atr_14 = atr_values[-1] if not np.isnan(atr_values[-1]) else 0
        
        # Volume
        vol_sma_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_spike = volumes[-1] / vol_sma_20 if vol_sma_20 > 0 else 1
        
        # Trend regime
        from libs.indicators.src.trend import sma
        sma_20 = sma(closes, 20)[-1]
        sma_50 = sma(closes, 50)[-1] if len(closes) >= 50 else sma_20
        
        if closes[-1] > sma_20 > sma_50:
            trend_regime = "UPTREND"
        elif closes[-1] < sma_20 < sma_50:
            trend_regime = "DOWNTREND"
        else:
            trend_regime = "SIDEWAYS"
        
        # Volatility regime
        if volatility_20 > 30:
            volatility_regime = "HIGH"
        elif volatility_20 < 15:
            volatility_regime = "LOW"
        else:
            volatility_regime = "MEDIUM"
        
        return {
            "price": float(closes[-1]),
            "returns_1": float(returns_1),
            "returns_5": float(returns_5),
            "returns_20": float(returns_20),
            "volatility_20": float(volatility_20),
            "rsi_14": float(rsi_14),
            "macd": float(macd_val),
            "macd_signal": float(macd_signal),
            "macd_histogram": float(macd_hist),
            "atr_14": float(atr_14),
            "volume_spike": float(volume_spike),
            "trend_regime": trend_regime,
            "volatility_regime": volatility_regime,
        }


# Global feature computer
feature_computer = FeatureComputer()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": FeatureComputer.VERSION
    }


@app.get("/features/latest")
async def get_latest_features(
    instrument_id: str = Query(...),
    timeframe: str = Query(default="15m")
):
    """Get latest feature vector for an instrument"""
    return await feature_computer.compute_features(instrument_id, timeframe)


@app.post("/features/batch")
async def get_batch_features(
    requests: List[FeatureRequest]
):
    """Get features for multiple instruments"""
    results = []
    for req in requests:
        try:
            features = await feature_computer.compute_features(
                req.instrument_id,
                req.timeframe
            )
            results.append(features)
        except Exception as e:
            logger.error(f"Error computing features for {req.instrument_id}: {e}")
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, reload=True)
