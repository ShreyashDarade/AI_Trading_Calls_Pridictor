"""
Model Training Pipeline
Trains and deploys ML models for signal generation
"""
import asyncio
import logging
import os
import pickle
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Training Service",
    description="ML model training and deployment pipeline",
    version="1.0.0"
)

# Model storage path
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# MODELS
# ============================================

class ModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class TrainingStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrainingConfig(BaseModel):
    model_type: ModelType = ModelType.XGBOOST
    universe: str = "NIFTY50"
    symbols: Optional[List[str]] = None
    start_date: date
    end_date: date
    timeframe: str = "15m"
    target_horizon: int = 10  # Number of bars ahead to predict
    min_confidence: float = 0.6
    
    # Model hyperparameters
    hyperparameters: Dict[str, Any] = {}
    
    # Training settings
    train_test_split: float = 0.2
    cross_validation_folds: int = 5
    early_stopping_rounds: int = 50


class TrainingJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    progress_percent: int = 0
    
    # Results
    metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    model_version: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelInfo(BaseModel):
    model_id: str
    version: str
    model_type: ModelType
    created_at: datetime
    metrics: Dict[str, float]
    is_active: bool = False
    path: str


# ============================================
# FEATURE ENGINEERING
# ============================================

class FeatureEngineer:
    """Generates features for ML training"""
    
    FEATURE_NAMES = [
        "returns_1", "returns_5", "returns_10", "returns_20",
        "volatility_10", "volatility_20",
        "rsi_14", "rsi_7",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width",
        "atr_14",
        "volume_sma_20", "volume_spike",
        "sma_20", "sma_50", "sma_cross",
        "ema_9", "ema_21",
        "adx_14",
        "obv_change",
        "hour_of_day", "day_of_week"
    ]
    
    @staticmethod
    async def compute_features(candles: List[dict]) -> Tuple[np.ndarray, List[str]]:
        """Compute feature matrix from candles"""
        if len(candles) < 60:
            return np.array([]), []
        
        closes = np.array([c["close"] for c in candles])
        highs = np.array([c["high"] for c in candles])
        lows = np.array([c["low"] for c in candles])
        volumes = np.array([c["volume"] for c in candles])
        
        features = []
        
        for i in range(50, len(candles)):
            row = []
            
            # Returns
            row.append((closes[i] / closes[i-1] - 1) * 100)
            row.append((closes[i] / closes[i-5] - 1) * 100)
            row.append((closes[i] / closes[i-10] - 1) * 100)
            row.append((closes[i] / closes[i-20] - 1) * 100)
            
            # Volatility
            row.append(np.std(closes[i-10:i]) / closes[i] * 100)
            row.append(np.std(closes[i-20:i]) / closes[i] * 100)
            
            # RSI
            row.append(FeatureEngineer._compute_rsi(closes[:i+1], 14))
            row.append(FeatureEngineer._compute_rsi(closes[:i+1], 7))
            
            # MACD
            ema12 = FeatureEngineer._ema(closes[:i+1], 12)
            ema26 = FeatureEngineer._ema(closes[:i+1], 26)
            macd = ema12 - ema26
            signal = FeatureEngineer._ema(np.array([macd]), 9) if i > 35 else macd
            row.extend([macd, signal, macd - signal])
            
            # Bollinger Bands
            sma20 = np.mean(closes[i-20:i])
            std20 = np.std(closes[i-20:i])
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            row.extend([bb_upper, bb_lower, (bb_upper - bb_lower) / sma20])
            
            # ATR
            row.append(FeatureEngineer._compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14))
            
            # Volume
            vol_sma = np.mean(volumes[i-20:i])
            row.extend([vol_sma, volumes[i] / vol_sma if vol_sma > 0 else 1])
            
            # Moving averages
            sma50 = np.mean(closes[i-50:i]) if i >= 50 else closes[i]
            row.extend([sma20, sma50, 1 if sma20 > sma50 else 0])
            
            # EMA
            row.extend([
                FeatureEngineer._ema(closes[:i+1], 9),
                FeatureEngineer._ema(closes[:i+1], 21)
            ])
            
            # ADX (simplified)
            row.append(50)  # Placeholder
            
            # OBV change
            row.append(0)  # Placeholder
            
            # Time features
            timestamp = candles[i].get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            row.extend([timestamp.hour, timestamp.weekday()])
            
            features.append(row)
        
        return np.array(features), FeatureEngineer.FEATURE_NAMES
    
    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return prices[-1]
        
        multiplier = 2 / (period + 1)
        ema = prices[-period]
        
        for price in prices[-period+1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        if len(closes) < period + 1:
            return 0.0
        
        tr = []
        for i in range(-period, 0):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr.append(max(high_low, high_close, low_close))
        
        return np.mean(tr)


# ============================================
# MODEL TRAINER
# ============================================

class ModelTrainer:
    """Trains ML models for signal prediction"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.models: Dict[str, ModelInfo] = {}
        self.active_model: Optional[str] = None
    
    async def start_training(self, config: TrainingConfig) -> TrainingJob:
        """Start a training job"""
        job = TrainingJob(config=config)
        self.jobs[job.id] = job
        return job
    
    async def run_training(self, job_id: str):
        """Execute training job"""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.utcnow()
        
        try:
            # Step 1: Load data (30%)
            job.progress_percent = 10
            candles = await self._load_training_data(job.config)
            job.progress_percent = 30
            
            if len(candles) < 100:
                raise ValueError("Insufficient data for training")
            
            # Step 2: Feature engineering (50%)
            features, feature_names = await FeatureEngineer.compute_features(candles)
            job.progress_percent = 50
            
            # Step 3: Create labels
            labels = self._create_labels(candles, job.config.target_horizon)
            job.progress_percent = 60
            
            # Align features and labels
            labels = labels[50:len(features)+50]
            if len(features) != len(labels):
                min_len = min(len(features), len(labels))
                features = features[:min_len]
                labels = labels[:min_len]
            
            # Step 4: Train model (80%)
            model, metrics = await self._train_model(
                features, labels, feature_names, job.config
            )
            job.progress_percent = 80
            
            # Step 5: Save model (100%)
            model_version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            model_path = MODELS_DIR / f"{job.config.model_type.value}_{model_version}.pkl"
            
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            # Calculate feature importance
            feature_importance = {}
            if hasattr(model, "feature_importances_"):
                for name, imp in zip(feature_names, model.feature_importances_):
                    feature_importance[name] = float(imp)
            
            job.model_path = str(model_path)
            job.model_version = model_version
            job.metrics = metrics
            job.feature_importance = feature_importance
            job.status = TrainingStatus.COMPLETED
            job.progress_percent = 100
            job.completed_at = datetime.utcnow()
            
            # Register model
            model_info = ModelInfo(
                model_id=job.id,
                version=model_version,
                model_type=job.config.model_type,
                created_at=job.completed_at,
                metrics=metrics,
                is_active=False,
                path=str(model_path)
            )
            self.models[job.id] = model_info
            
            logger.info(f"Training completed: {model_version}, Accuracy: {metrics.get('accuracy', 0):.2%}")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Training failed: {e}")
    
    async def _load_training_data(self, config: TrainingConfig) -> List[dict]:
        """Load historical candle data"""
        raise HTTPException(
            status_code=501,
            detail="Training data loading is not implemented; configure ClickHouse/Groww ingestion to use real historical candles"
        )
    
    def _create_labels(self, candles: List[dict], horizon: int) -> np.ndarray:
        """Create labels: 1 if price goes up by >1%, 0 otherwise"""
        closes = np.array([c["close"] for c in candles])
        labels = []
        
        for i in range(len(closes) - horizon):
            future_return = (closes[i + horizon] / closes[i] - 1) * 100
            if future_return > 1:
                labels.append(1)  # Buy signal
            elif future_return < -1:
                labels.append(-1)  # Sell signal
            else:
                labels.append(0)  # No signal
        
        # Pad with zeros
        labels.extend([0] * horizon)
        
        return np.array(labels)
    
    async def _train_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        config: TrainingConfig
    ) -> Tuple[Any, Dict[str, float]]:
        """Train the ML model"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=config.train_test_split, shuffle=False
        )
        
        # Train model based on type
        if config.model_type == ModelType.XGBOOST:
            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=config.hyperparameters.get("n_estimators", 100),
                    max_depth=config.hyperparameters.get("max_depth", 6),
                    learning_rate=config.hyperparameters.get("learning_rate", 0.1),
                    early_stopping_rounds=config.early_stopping_rounds,
                    eval_metric="mlogloss"
                )
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(n_estimators=100)
                model.fit(X_train, y_train)
                
        elif config.model_type == ModelType.LIGHTGBM:
            try:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=config.hyperparameters.get("n_estimators", 100),
                    max_depth=config.hyperparameters.get("max_depth", 6),
                    learning_rate=config.hyperparameters.get("learning_rate", 0.1)
                )
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(n_estimators=100)
                model.fit(X_train, y_train)
                
        elif config.model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=config.hyperparameters.get("n_estimators", 100),
                max_depth=config.hyperparameters.get("max_depth", 10)
            )
            model.fit(X_train, y_train)
            
        else:  # Gradient Boosting
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=config.hyperparameters.get("n_estimators", 100),
                max_depth=config.hyperparameters.get("max_depth", 6)
            )
            model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        return model, metrics
    
    def activate_model(self, model_id: str) -> bool:
        """Set a model as the active production model"""
        if model_id not in self.models:
            return False
        
        # Deactivate current model
        if self.active_model and self.active_model in self.models:
            self.models[self.active_model].is_active = False
        
        # Activate new model
        self.models[model_id].is_active = True
        self.active_model = model_id
        
        return True


# Global trainer
trainer = ModelTrainer()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "total_jobs": len(trainer.jobs),
        "total_models": len(trainer.models),
        "active_model": trainer.active_model
    }


@app.post("/train")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks
):
    """Start a new training job"""
    job = await trainer.start_training(config)
    background_tasks.add_task(trainer.run_training, job.id)
    
    return {
        "job_id": job.id,
        "status": job.status,
        "message": "Training started"
    }


@app.get("/jobs")
async def list_jobs():
    """List all training jobs"""
    jobs = sorted(trainer.jobs.values(), key=lambda x: x.created_at, reverse=True)
    return {"jobs": jobs}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get training job details"""
    job = trainer.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/models")
async def list_models():
    """List all trained models"""
    return {"models": list(trainer.models.values())}


@app.post("/models/{model_id}/activate")
async def activate_model(model_id: str):
    """Set model as active for production"""
    if trainer.activate_model(model_id):
        return {"message": f"Model {model_id} activated"}
    raise HTTPException(status_code=404, detail="Model not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012, reload=True)
