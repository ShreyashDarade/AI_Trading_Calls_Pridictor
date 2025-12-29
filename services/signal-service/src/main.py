"""
Signal Service - Enhanced with GPT Integration
AI-powered trading signal generation using ML models OR GPT
"""
import asyncio
import logging
import os
import json
import pickle
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Signal Service",
    description="AI-powered trading signal generation with ML and GPT",
    version="2.0.0"
)


# ============================================
# CONFIGURATION
# ============================================

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    NSE_SERVICE_URL = os.getenv("NSE_SERVICE_URL", "http://localhost:8020")
    MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
    MIN_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.6"))
    PREDICTION_MODE = os.getenv("PREDICTION_MODE", "auto")  # auto, ml, gpt, rule


config = Config()


# ============================================
# MODELS
# ============================================

class SignalAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


class PredictionMode(str, Enum):
    AUTO = "auto"
    ML = "ml"
    GPT = "gpt"
    RULE = "rule"


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
    
    prediction_mode: str = "auto"
    model_version: str = "v1"
    gpt_analysis: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class GenerateSignalRequest(BaseModel):
    symbol: str
    exchange: str = "NSE"
    timeframe: str = "15m"
    mode: PredictionMode = PredictionMode.AUTO


class SignalGeneratorConfig(BaseModel):
    min_confidence: float = 0.6
    stop_loss_pct: float = 2.0
    target_1_pct: float = 3.0
    target_2_pct: float = 5.0
    prediction_mode: PredictionMode = PredictionMode.AUTO


# ============================================
# GPT PREDICTOR
# ============================================

class GPTPredictor:
    """Uses OpenAI GPT for market prediction"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.enabled = bool(api_key)
    
    async def predict(self, symbol: str, features: Dict[str, float]) -> Dict:
        """Get GPT prediction for trading signal"""
        if not self.enabled:
            return {"action": "NO_TRADE", "confidence": 0, "analysis": "GPT not configured"}
        
        prompt = self._build_prompt(symbol, features)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": self._system_prompt()},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return self._parse_response(content)
                else:
                    logger.error(f"GPT API error: {response.status_code}")
                    return {"action": "NO_TRADE", "confidence": 0, "analysis": "GPT API error"}
                    
        except Exception as e:
            logger.error(f"GPT prediction error: {e}")
            return {"action": "NO_TRADE", "confidence": 0, "analysis": str(e)}
    
    def _system_prompt(self) -> str:
        return """You are an expert Indian stock market analyst. Analyze the given technical indicators and provide a trading recommendation.

Your response MUST be in this exact JSON format:
{
    "action": "LONG" or "SHORT" or "NO_TRADE",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "key_factors": ["factor1", "factor2"]
}

Rules:
- LONG: Buy signal when bullish conditions are strong
- SHORT: Sell signal when bearish conditions are strong  
- NO_TRADE: When conditions are unclear or neutral
- Confidence should reflect the strength of your conviction
- Consider RSI, MACD, trends, and volume together"""
    
    def _build_prompt(self, symbol: str, features: Dict[str, float]) -> str:
        return f"""Analyze {symbol} (NSE) with these technical indicators:

- Current Price: ₹{features.get('price', 0):.2f}
- RSI (14): {features.get('rsi_14', 50):.1f}
- MACD: {features.get('macd', 0):.2f}
- MACD Signal: {features.get('macd_signal', 0):.2f}
- MACD Histogram: {features.get('macd_histogram', 0):.2f}
- Price vs SMA20: {features.get('price_vs_sma20', 0):.2f}%
- SMA20/SMA50 Cross: {"Bullish" if features.get('sma_cross', 0) > 0 else "Bearish"}
- EMA9/EMA21 Cross: {"Bullish" if features.get('ema_cross', 0) > 0 else "Bearish"}
- Volume Spike: {features.get('volume_spike', 1):.2f}x average
- 1-Day Return: {features.get('returns_1', 0):.2f}%
- 5-Day Return: {features.get('returns_5', 0):.2f}%
- ATR (14): {features.get('atr_14', 0):.2f}

Provide your trading recommendation in JSON format."""
    
    def _parse_response(self, content: str) -> Dict:
        try:
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)
                
                return {
                    "action": data.get("action", "NO_TRADE").upper(),
                    "confidence": float(data.get("confidence", 0.5)),
                    "analysis": data.get("reasoning", ""),
                    "reason_codes": data.get("key_factors", [])
                }
        except Exception as e:
            logger.error(f"Failed to parse GPT response: {e}")
        
        return {"action": "NO_TRADE", "confidence": 0, "analysis": "Failed to parse"}


# ============================================
# SIGNAL GENERATOR ENGINE
# ============================================

class SignalGenerator:
    """AI Signal Generation Engine with ML and GPT options"""
    
    def __init__(self):
        self.signals: Dict[str, Signal] = {}
        self.model = None
        self.model_version = "v1"
        self.config = SignalGeneratorConfig()
        self.gpt_predictor = GPTPredictor(config.OPENAI_API_KEY)
    
    async def initialize(self):
        """Initialize the generator"""
        await self._load_model()
        logger.info("Signal generator initialized")
    
    async def _load_model(self):
        """Load the latest trained model"""
        try:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_files = list(config.MODELS_DIR.glob("*.pkl"))
            if model_files:
                latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                with open(latest_model, "rb") as f:
                    self.model = pickle.load(f)
                self.model_version = latest_model.stem
                logger.info(f"Loaded model: {self.model_version}")
        except Exception as e:
            logger.warning(f"No model loaded: {e}")
    
    async def generate_signal(
        self,
        symbol: str,
        exchange: str = "NSE",
        timeframe: str = "15m",
        mode: PredictionMode = PredictionMode.AUTO
    ) -> Signal:
        """Generate trading signal using specified mode"""
        
        # Get market data
        features = await self._get_features(symbol)
        current_price = features.get("price", 0)
        
        if not current_price:
            return Signal(
                instrument_id=f"{exchange}:{symbol}",
                symbol=symbol,
                exchange=exchange,
                action=SignalAction.NO_TRADE,
                confidence=0.0,
                reason_codes=["NO_DATA"]
            )
        
        # Determine prediction mode
        if mode == PredictionMode.AUTO:
            if self.model:
                mode = PredictionMode.ML
            elif self.gpt_predictor.enabled:
                mode = PredictionMode.GPT
            else:
                mode = PredictionMode.RULE
        
        # Get prediction based on mode
        if mode == PredictionMode.GPT and self.gpt_predictor.enabled:
            result = await self.gpt_predictor.predict(symbol, features)
            action = SignalAction(result["action"])
            confidence = result["confidence"]
            reason_codes = result.get("reason_codes", [])
            gpt_analysis = result.get("analysis", "")
        elif mode == PredictionMode.ML and self.model:
            prediction, confidence = await self._model_prediction(features)
            action = self._prediction_to_action(prediction, confidence)
            reason_codes = self._generate_reason_codes(features, action)
            gpt_analysis = None
        else:
            prediction, confidence = await self._rule_based_prediction(features)
            action = self._prediction_to_action(prediction, confidence)
            reason_codes = self._generate_reason_codes(features, action)
            gpt_analysis = None
        
        # Calculate levels
        entry_price = current_price
        stop_loss, target_1, target_2 = self._calculate_levels(action, entry_price)
        
        signal = Signal(
            instrument_id=f"{exchange}:{symbol}",
            symbol=symbol,
            exchange=exchange,
            action=action,
            confidence=confidence,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2) if stop_loss else None,
            target_1=round(target_1, 2) if target_1 else None,
            target_2=round(target_2, 2) if target_2 else None,
            timeframe=timeframe,
            reason_codes=reason_codes[:5],
            prediction_mode=mode.value,
            gpt_analysis=gpt_analysis,
            model_version=self.model_version,
            expires_at=datetime.utcnow() + timedelta(hours=4)
        )
        
        self.signals[signal.id] = signal
        return signal
    
    def _prediction_to_action(self, prediction: int, confidence: float) -> SignalAction:
        if prediction == 1 and confidence >= self.config.min_confidence:
            return SignalAction.LONG
        elif prediction == -1 and confidence >= self.config.min_confidence:
            return SignalAction.SHORT
        return SignalAction.NO_TRADE
    
    def _calculate_levels(self, action: SignalAction, price: float):
        if action == SignalAction.LONG:
            return (
                price * (1 - self.config.stop_loss_pct / 100),
                price * (1 + self.config.target_1_pct / 100),
                price * (1 + self.config.target_2_pct / 100)
            )
        elif action == SignalAction.SHORT:
            return (
                price * (1 + self.config.stop_loss_pct / 100),
                price * (1 - self.config.target_1_pct / 100),
                price * (1 - self.config.target_2_pct / 100)
            )
        return None, None, None
    
    async def _get_features(self, symbol: str) -> Dict[str, float]:
        """Get features from NSE service or compute locally"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{config.NSE_SERVICE_URL}/api/quote/{symbol}",
                    timeout=10
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "price": data.get("ltp", 0),
                        "rsi_14": data.get("rsi", 50),
                        "macd": 0,
                        "macd_signal": 0,
                        "macd_histogram": 0,
                        "price_vs_sma20": data.get("change_percent", 0),
                        "sma_cross": 1 if data.get("change", 0) > 0 else -1,
                        "ema_cross": 1 if data.get("change", 0) > 0 else -1,
                        "volume_spike": 1.0,
                        "returns_1": data.get("change_percent", 0),
                        "returns_5": data.get("change_percent", 0) * 2,
                        "atr_14": abs(data.get("change", 0)),
                    }
        except Exception as e:
            logger.error(f"Feature fetch error: {e}")
        
        return {}
    
    async def _model_prediction(self, features: Dict[str, float]) -> tuple:
        """Get prediction from ML model"""
        try:
            feature_names = [
                "returns_1", "returns_5", "rsi_14", "macd", 
                "macd_signal", "macd_histogram", "atr_14", 
                "volume_spike", "sma_cross", "ema_cross"
            ]
            X = np.array([[features.get(f, 0) for f in feature_names]])
            prediction = self.model.predict(X)[0]
            
            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(X)[0]
                confidence = max(probas)
            else:
                confidence = 0.7
            
            return int(prediction), float(confidence)
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return await self._rule_based_prediction(features)
    
    async def _rule_based_prediction(self, features: Dict[str, float]) -> tuple:
        """Rule-based prediction fallback"""
        score = 0
        total_rules = 0
        
        rsi = features.get("rsi_14", 50)
        macd_hist = features.get("macd_histogram", 0)
        sma_cross = features.get("sma_cross", 0)
        ema_cross = features.get("ema_cross", 0)
        volume_spike = features.get("volume_spike", 1)
        
        # RSI rules
        if rsi < 30:
            score += 2
        elif rsi > 70:
            score -= 2
        total_rules += 2
        
        # MACD
        if macd_hist > 0:
            score += 1
        elif macd_hist < 0:
            score -= 1
        total_rules += 1
        
        # Crossovers
        score += sma_cross
        score += ema_cross
        total_rules += 2
        
        # Volume
        if volume_spike > 1.5:
            score += 1 if score > 0 else -1
        total_rules += 1
        
        # Determine signal
        if score > 1:
            prediction = 1
        elif score < -1:
            prediction = -1
        else:
            prediction = 0
        
        confidence = min(abs(score) / total_rules + 0.3, 0.95)
        return prediction, confidence
    
    def _generate_reason_codes(self, features: Dict, action: SignalAction) -> List[str]:
        """Generate reason codes"""
        codes = []
        rsi = features.get("rsi_14", 50)
        macd_hist = features.get("macd_histogram", 0)
        sma_cross = features.get("sma_cross", 0)
        volume = features.get("volume_spike", 1)
        
        if rsi < 30:
            codes.append("RSI_OVERSOLD")
        elif rsi > 70:
            codes.append("RSI_OVERBOUGHT")
        
        if macd_hist > 0:
            codes.append("MACD_BULLISH")
        elif macd_hist < 0:
            codes.append("MACD_BEARISH")
        
        if sma_cross > 0:
            codes.append("UPTREND")
        else:
            codes.append("DOWNTREND")
        
        if volume > 1.5:
            codes.append("HIGH_VOLUME")
        
        if action == SignalAction.NO_TRADE:
            codes.append("LOW_CONFIDENCE")
        
        return codes[:5]
    
    def get_signals(
        self,
        min_confidence: float = 0.5,
        action: Optional[SignalAction] = None,
        limit: int = 20
    ) -> List[Signal]:
        """Get signals with filtering"""
        signals = list(self.signals.values())
        signals = [s for s in signals if s.confidence >= min_confidence]
        if action:
            signals = [s for s in signals if s.action == action]
        signals.sort(key=lambda s: s.created_at, reverse=True)
        return signals[:limit]


# Global generator
generator = SignalGenerator()
_auto_task: Optional[asyncio.Task] = None


# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup():
    await generator.initialize()
    _maybe_start_auto_signal_generation()


@app.on_event("shutdown")
async def shutdown():
    global _auto_task
    if _auto_task and not _auto_task.done():
        _auto_task.cancel()
        _auto_task = None


def _maybe_start_auto_signal_generation() -> None:
    """
    Optionally run continuous signal generation from real market data.

    Controlled by env vars:
      - AUTO_GENERATE_SIGNALS=true|false (default false)
      - SIGNAL_WATCHLIST=RELIANCE,TCS,... (default empty; will use NSEClient's list if empty)
      - SIGNAL_REFRESH_SECONDS=300
    """
    global _auto_task
    if _auto_task and not _auto_task.done():
        return

    enabled = os.getenv("AUTO_GENERATE_SIGNALS", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return

    refresh_s = float(os.getenv("SIGNAL_REFRESH_SECONDS", "300"))
    watchlist_raw = os.getenv("SIGNAL_WATCHLIST", "").strip()
    symbols = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()] if watchlist_raw else []

    async def _runner() -> None:
        nonlocal symbols, refresh_s
        while True:
            try:
                if not symbols:
                    # Keep it small by default; can be overridden via SIGNAL_WATCHLIST.
                    symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "ITC", "ICICIBANK", "KOTAKBANK"]

                for symbol in symbols[:20]:
                    try:
                        await generator.generate_signal(symbol=symbol, mode=PredictionMode.AUTO)
                    except Exception as e:
                        logger.warning(f"Auto signal generation failed for {symbol}: {e}")
                await asyncio.sleep(max(5.0, refresh_s))
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Auto signal generation loop error: {e}")
                await asyncio.sleep(10.0)

    _auto_task = asyncio.create_task(_runner())


# ============================================
# ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": generator.model is not None,
        "gpt_enabled": generator.gpt_predictor.enabled,
        "model_version": generator.model_version,
        "total_signals": len(generator.signals)
    }


@app.get("/api/signals/latest")
@app.get("/signals/latest")
async def get_latest_signals(
    limit: int = Query(default=20, le=50),
    min_confidence: float = Query(default=0.5, ge=0, le=1),
    action: Optional[str] = None
):
    """Get latest signals"""
    action_filter = None
    if action:
        try:
            action_filter = SignalAction(action.upper())
        except:
            pass
    
    signals = generator.get_signals(
        min_confidence=min_confidence,
        action=action_filter,
        limit=limit
    )
    
    return {"signals": [s.dict() for s in signals]}


@app.post("/signals/generate")
async def generate_signal(request: GenerateSignalRequest):
    """Generate new signal"""
    signal = await generator.generate_signal(
        symbol=request.symbol.upper(),
        exchange=request.exchange.upper(),
        timeframe=request.timeframe,
        mode=request.mode
    )
    return signal


@app.post("/signals/generate/batch")
async def generate_batch(
    symbols: List[str],
    mode: PredictionMode = PredictionMode.AUTO
):
    """Generate signals for multiple symbols"""
    signals = []
    for symbol in symbols[:20]:  # Limit to 20
        signal = await generator.generate_signal(
            symbol=symbol.upper(),
            mode=mode
        )
        signals.append(signal)
    return {"signals": signals}


@app.get("/signals/{signal_id}")
async def get_signal(signal_id: str):
    """Get specific signal"""
    if signal_id not in generator.signals:
        raise HTTPException(status_code=404, detail="Signal not found")
    return generator.signals[signal_id]


@app.get("/config")
async def get_config():
    """Get configuration"""
    return {
        "min_confidence": generator.config.min_confidence,
        "prediction_modes": ["auto", "ml", "gpt", "rule"],
        "gpt_available": generator.gpt_predictor.enabled,
        "ml_available": generator.model is not None
    }


@app.post("/config")
async def update_config(config_update: SignalGeneratorConfig):
    """Update configuration"""
    generator.config = config_update
    return {"message": "Configuration updated"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005, reload=True)
