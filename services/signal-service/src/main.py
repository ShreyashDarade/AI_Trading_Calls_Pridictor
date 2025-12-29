"""
Signal Service
AI-powered trading signal generation using REAL data
"""
import asyncio
import logging
import os
import sys
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np

# Add libs to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Signal Service",
    description="AI-powered trading signal generation",
    version="1.0.0"
)


# ============================================
# CONFIGURATION
# ============================================

class Config:
    GROWW_API_KEY = os.getenv("GROWW_API_KEY", "")
    GROWW_ACCESS_TOKEN = os.getenv("GROWW_ACCESS_TOKEN", "")
    FEATURE_SERVICE_URL = os.getenv("FEATURE_SERVICE_URL", "http://localhost:8004")
    MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
    MIN_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.6"))


config = Config()


# ============================================
# MODELS
# ============================================

class SignalAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


class Signal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    instrument_id: str
    symbol: str
    exchange: str = "NSE"
    
    action: SignalAction
    confidence: float
    
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    
    timeframe: str = "15m"
    reason_codes: List[str] = []
    
    feature_snapshot: Optional[Dict[str, Any]] = None
    model_version: str = "v1"
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class GenerateSignalRequest(BaseModel):
    symbol: str
    exchange: str = "NSE"
    timeframe: str = "15m"


class SignalGeneratorConfig(BaseModel):
    min_confidence: float = 0.6
    stop_loss_pct: float = 2.0
    target_1_pct: float = 3.0
    target_2_pct: float = 5.0
    max_signals_per_day: int = 50


# ============================================
# SIGNAL GENERATOR ENGINE
# ============================================

class SignalGenerator:
    """
    AI Signal Generation Engine
    Uses trained ML models + technical analysis rules
    """
    
    def __init__(self):
        self.signals: Dict[str, Signal] = {}
        self.model = None
        self.model_version = "v1"
        self.config = SignalGeneratorConfig()
        self._groww_client = None
    
    async def initialize(self, api_key: str, access_token: str):
        """Initialize with Groww credentials and load model"""
        try:
            # Initialize Groww client
            from libs.groww_client.src.client import GrowwClient
            self._groww_client = GrowwClient(api_key, access_token)
            
            # Load trained model
            await self._load_model()
            
            logger.info("Signal generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize signal generator: {e}")
    
    async def _load_model(self):
        """Load the latest trained model"""
        try:
            model_files = list(config.MODELS_DIR.glob("*.pkl"))
            if model_files:
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                with open(latest_model, "rb") as f:
                    self.model = pickle.load(f)
                self.model_version = latest_model.stem
                logger.info(f"Loaded model: {self.model_version}")
            else:
                logger.warning("No trained model found, using rule-based signals")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    async def generate_signal(
        self,
        symbol: str,
        exchange: str = "NSE",
        timeframe: str = "15m"
    ) -> Signal:
        """Generate trading signal for a symbol using real data"""
        
        # Step 1: Get real market data
        candles = await self._get_candles(exchange, symbol, timeframe)
        
        if not candles or len(candles) < 50:
            return Signal(
                instrument_id=f"{exchange}:{symbol}",
                symbol=symbol,
                exchange=exchange,
                action=SignalAction.NO_TRADE,
                confidence=0.0,
                timeframe=timeframe,
                reason_codes=["INSUFFICIENT_DATA"]
            )
        
        # Step 2: Compute features
        features = await self._compute_features(candles)
        
        # Step 3: Get current price
        current_price = candles[-1]["close"]
        
        # Step 4: Generate prediction
        if self.model:
            prediction, confidence = await self._model_prediction(features)
        else:
            prediction, confidence = await self._rule_based_prediction(features)
        
        # Step 5: Determine action
        if prediction == 1 and confidence >= self.config.min_confidence:
            action = SignalAction.LONG
        elif prediction == -1 and confidence >= self.config.min_confidence:
            action = SignalAction.SHORT
        else:
            action = SignalAction.NO_TRADE
        
        # Step 6: Calculate levels
        entry_price = current_price
        stop_loss = None
        target_1 = None
        target_2 = None
        
        if action == SignalAction.LONG:
            stop_loss = entry_price * (1 - self.config.stop_loss_pct / 100)
            target_1 = entry_price * (1 + self.config.target_1_pct / 100)
            target_2 = entry_price * (1 + self.config.target_2_pct / 100)
        elif action == SignalAction.SHORT:
            stop_loss = entry_price * (1 + self.config.stop_loss_pct / 100)
            target_1 = entry_price * (1 - self.config.target_1_pct / 100)
            target_2 = entry_price * (1 - self.config.target_2_pct / 100)
        
        # Step 7: Generate reason codes
        reason_codes = self._generate_reason_codes(features, action)
        
        # Create signal
        signal = Signal(
            instrument_id=f"{exchange}:{symbol}",
            symbol=symbol,
            exchange=exchange,
            action=action,
            confidence=confidence,
            entry_price=round(entry_price, 2) if entry_price else None,
            stop_loss=round(stop_loss, 2) if stop_loss else None,
            target_1=round(target_1, 2) if target_1 else None,
            target_2=round(target_2, 2) if target_2 else None,
            timeframe=timeframe,
            reason_codes=reason_codes,
            feature_snapshot=features,
            model_version=self.model_version,
            expires_at=datetime.utcnow() + timedelta(hours=4)
        )
        
        # Store signal
        self.signals[signal.id] = signal
        
        return signal
    
    async def _get_candles(
        self,
        exchange: str,
        symbol: str,
        timeframe: str
    ) -> List[dict]:
        """Get historical candles from Groww API"""
        try:
            from libs.groww_client.src.historical import HistoricalDataClient
            
            client = HistoricalDataClient(
                config.GROWW_API_KEY,
                config.GROWW_ACCESS_TOKEN
            )
            
            # Map timeframe to Groww interval
            interval_map = {
                "1m": "minute",
                "5m": "5minute",
                "15m": "15minute",
                "1h": "60minute",
                "1d": "day"
            }
            
            interval = interval_map.get(timeframe, "15minute")
            
            candles = await client.get_candles(
                exchange=exchange,
                symbol=symbol,
                interval=interval,
                from_date=datetime.now().date() - timedelta(days=30),
                to_date=datetime.now().date()
            )
            
            return candles
            
        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {e}")
            return []
    
    async def _compute_features(self, candles: List[dict]) -> Dict[str, float]:
        """Compute technical indicators and features"""
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        volumes = np.array([c["volume"] for c in candles])
        
        try:
            from libs.indicators.src.momentum import rsi, macd
            from libs.indicators.src.trend import sma, ema
            from libs.indicators.src.volatility import atr, bollinger_bands
            
            # Compute indicators
            rsi_values = rsi(closes, 14)
            macd_line, signal_line, histogram = macd(closes)
            sma_20 = sma(closes, 20)
            sma_50 = sma(closes, 50)
            ema_9 = ema(closes, 9)
            ema_21 = ema(closes, 21)
            atr_values = atr(highs, lows, closes, 14)
            bb_upper, bb_middle, bb_lower = bollinger_bands(closes, 20, 2)
            
            # Volume analysis
            vol_sma = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            volume_spike = volumes[-1] / vol_sma if vol_sma > 0 else 1
            
            # Returns
            returns_1 = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
            returns_5 = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0
            returns_20 = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
            
            features = {
                "price": float(closes[-1]),
                "rsi_14": float(rsi_values[-1]) if not np.isnan(rsi_values[-1]) else 50,
                "macd": float(macd_line[-1]) if not np.isnan(macd_line[-1]) else 0,
                "macd_signal": float(signal_line[-1]) if not np.isnan(signal_line[-1]) else 0,
                "macd_histogram": float(histogram[-1]) if not np.isnan(histogram[-1]) else 0,
                "sma_20": float(sma_20[-1]),
                "sma_50": float(sma_50[-1]) if len(closes) >= 50 else float(sma_20[-1]),
                "ema_9": float(ema_9[-1]),
                "ema_21": float(ema_21[-1]),
                "atr_14": float(atr_values[-1]) if not np.isnan(atr_values[-1]) else 0,
                "bb_upper": float(bb_upper[-1]),
                "bb_lower": float(bb_lower[-1]),
                "bb_width": float((bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] * 100),
                "volume_spike": float(volume_spike),
                "returns_1": float(returns_1),
                "returns_5": float(returns_5),
                "returns_20": float(returns_20),
            }
            
            # Derived features
            features["price_vs_sma20"] = (closes[-1] / sma_20[-1] - 1) * 100
            features["sma_cross"] = 1 if sma_20[-1] > features["sma_50"] else -1
            features["ema_cross"] = 1 if ema_9[-1] > ema_21[-1] else -1
            
            return features
            
        except Exception as e:
            logger.error(f"Feature computation error: {e}")
            return {"price": float(closes[-1]), "error": str(e)}
    
    async def _model_prediction(
        self,
        features: Dict[str, float]
    ) -> tuple:
        """Get prediction from ML model"""
        try:
            # Prepare feature vector
            feature_names = [
                "returns_1", "returns_5", "returns_20",
                "rsi_14", "macd", "macd_signal", "macd_histogram",
                "atr_14", "volume_spike", "sma_cross", "ema_cross"
            ]
            
            X = np.array([[features.get(f, 0) for f in feature_names]])
            
            # Get prediction
            prediction = self.model.predict(X)[0]
            
            # Get probability if available
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(X)[0]
                confidence = max(probas)
            else:
                confidence = 0.7  # Default confidence
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return await self._rule_based_prediction(features)
    
    async def _rule_based_prediction(
        self,
        features: Dict[str, float]
    ) -> tuple:
        """Rule-based prediction when no ML model is available"""
        score = 0
        total_rules = 0
        
        rsi = features.get("rsi_14", 50)
        macd_hist = features.get("macd_histogram", 0)
        sma_cross = features.get("sma_cross", 0)
        ema_cross = features.get("ema_cross", 0)
        volume_spike = features.get("volume_spike", 1)
        price_vs_sma = features.get("price_vs_sma20", 0)
        
        # RSI rules
        if rsi < 30:
            score += 2  # Oversold - bullish
            total_rules += 2
        elif rsi > 70:
            score -= 2  # Overbought - bearish
            total_rules += 2
        else:
            total_rules += 1
        
        # MACD rules
        if macd_hist > 0:
            score += 1
        elif macd_hist < 0:
            score -= 1
        total_rules += 1
        
        # Moving average rules
        if sma_cross > 0:
            score += 1
        else:
            score -= 1
        total_rules += 1
        
        if ema_cross > 0:
            score += 1
        else:
            score -= 1
        total_rules += 1
        
        # Volume confirmation
        if volume_spike > 1.5:
            score += 1 if score > 0 else -1  # Confirms trend
            total_rules += 1
        
        # Price position
        if price_vs_sma > 2:
            score += 0.5  # Above SMA
        elif price_vs_sma < -2:
            score -= 0.5  # Below SMA
        total_rules += 1
        
        # Calculate prediction and confidence
        if score > 1:
            prediction = 1  # Long
        elif score < -1:
            prediction = -1  # Short
        else:
            prediction = 0  # No trade
        
        confidence = min(abs(score) / total_rules + 0.3, 0.95)
        
        return prediction, confidence
    
    def _generate_reason_codes(
        self,
        features: Dict[str, float],
        action: SignalAction
    ) -> List[str]:
        """Generate human-readable reason codes"""
        codes = []
        
        rsi = features.get("rsi_14", 50)
        macd_hist = features.get("macd_histogram", 0)
        volume_spike = features.get("volume_spike", 1)
        sma_cross = features.get("sma_cross", 0)
        
        if rsi < 30:
            codes.append("RSI_OVERSOLD")
        elif rsi > 70:
            codes.append("RSI_OVERBOUGHT")
        elif 40 <= rsi <= 60:
            codes.append("RSI_NEUTRAL")
        
        if macd_hist > 0:
            codes.append("MACD_BULLISH")
        elif macd_hist < 0:
            codes.append("MACD_BEARISH")
        
        if sma_cross > 0:
            codes.append("UPTREND")
        else:
            codes.append("DOWNTREND")
        
        if volume_spike > 1.5:
            codes.append("HIGH_VOLUME")
        elif volume_spike < 0.5:
            codes.append("LOW_VOLUME")
        
        if action == SignalAction.NO_TRADE:
            codes.append("LOW_CONFIDENCE")
        
        return codes[:5]  # Max 5 reason codes
    
    def get_signals(
        self,
        min_confidence: float = 0.5,
        action: Optional[SignalAction] = None,
        limit: int = 20
    ) -> List[Signal]:
        """Get signals with filtering"""
        signals = list(self.signals.values())
        
        # Filter
        signals = [s for s in signals if s.confidence >= min_confidence]
        if action:
            signals = [s for s in signals if s.action == action]
        
        # Sort by created_at descending
        signals.sort(key=lambda s: s.created_at, reverse=True)
        
        return signals[:limit]


# Global generator
generator = SignalGenerator()


# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup():
    await generator.initialize(
        config.GROWW_API_KEY,
        config.GROWW_ACCESS_TOKEN
    )


# ============================================
# ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": generator.model is not None,
        "model_version": generator.model_version,
        "total_signals": len(generator.signals)
    }


@app.post("/signals/generate")
async def generate_signal(request: GenerateSignalRequest):
    """Generate new signal for a symbol"""
    signal = await generator.generate_signal(
        symbol=request.symbol.upper(),
        exchange=request.exchange.upper(),
        timeframe=request.timeframe
    )
    return signal


@app.post("/signals/generate/batch")
async def generate_signals_batch(
    symbols: List[str],
    exchange: str = "NSE",
    timeframe: str = "15m"
):
    """Generate signals for multiple symbols"""
    signals = []
    
    for symbol in symbols:
        signal = await generator.generate_signal(
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            timeframe=timeframe
        )
        signals.append(signal)
    
    return {"signals": signals}


@app.get("/signals/latest")
async def get_latest_signals(
    limit: int = Query(default=20, le=50),
    min_confidence: float = Query(default=0.5, ge=0, le=1),
    action: Optional[str] = None
):
    """Get latest signals"""
    action_filter = None
    if action:
        action_filter = SignalAction(action.upper())
    
    signals = generator.get_signals(
        min_confidence=min_confidence,
        action=action_filter,
        limit=limit
    )
    
    return {"signals": signals}


@app.get("/signals/{signal_id}")
async def get_signal(signal_id: str):
    """Get specific signal"""
    if signal_id not in generator.signals:
        raise HTTPException(status_code=404, detail="Signal not found")
    return generator.signals[signal_id]


@app.post("/config")
async def update_config(config_update: SignalGeneratorConfig):
    """Update signal generator configuration"""
    generator.config = config_update
    return {"message": "Configuration updated", "config": config_update}


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return generator.config


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005, reload=True)
