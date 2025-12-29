"""
Reinforcement Learning Trading Service
RL-based trading strategies using DQN and PPO
"""
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
from enum import Enum
from collections import deque
import random

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RL Trading Service",
    description="Reinforcement Learning trading strategies",
    version="4.0.0"
)


class Action(int, Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class RLConfig(BaseModel):
    state_size: int = 20
    action_size: int = 3
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 64
    update_frequency: int = 4


class TrainingStatus(BaseModel):
    episode: int = 0
    total_episodes: int = 0
    epsilon: float = 1.0
    avg_reward: float = 0
    total_profit: float = 0
    win_rate: float = 0
    is_training: bool = False


class DQNAgent:
    """Deep Q-Network trading agent"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.memory = deque(maxlen=config.memory_size)
        self.epsilon = config.epsilon
        self.model = None
        self.target_model = None
        self._build_model()
    
    def _build_model(self):
        """Build neural network (simplified version without TensorFlow)"""
        # Simplified linear model weights
        self.weights = np.random.randn(self.config.state_size, self.config.action_size) * 0.01
        self.bias = np.zeros(self.config.action_size)
    
    def get_state(self, data: dict) -> np.ndarray:
        """Convert market data to state vector"""
        features = [
            data.get("returns_1", 0),
            data.get("returns_5", 0),
            data.get("returns_10", 0),
            data.get("rsi", 50) / 100 - 0.5,
            data.get("macd", 0) / 100,
            data.get("atr_pct", 1) / 5,
            data.get("volume_ratio", 1) - 1,
            data.get("bb_position", 0.5),
            data.get("sma_cross", 0),
            data.get("ema_cross", 0),
            data.get("position", 0),
            data.get("unrealized_pnl", 0) / 100,
            data.get("hour", 12) / 24,
            data.get("day_of_week", 2) / 4,
        ]
        
        # Pad to state size
        while len(features) < self.config.state_size:
            features.append(0)
        
        return np.array(features[:self.config.state_size])
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_size - 1)
        
        q_values = np.dot(state, self.weights) + self.bias
        return int(np.argmax(q_values))
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch from memory"""
        if len(self.memory) < self.config.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.config.batch_size)
        total_loss = 0
        
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_q = np.dot(next_state, self.weights) + self.bias
                target += self.config.gamma * np.max(next_q)
            
            # Simple gradient update
            q_values = np.dot(state, self.weights) + self.bias
            error = target - q_values[action]
            
            self.weights[:, action] += self.config.learning_rate * error * state
            self.bias[action] += self.config.learning_rate * error
            
            total_loss += error ** 2
        
        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay
        
        return total_loss / self.config.batch_size
    
    def save(self, path: str):
        np.savez(path, weights=self.weights, bias=self.bias, epsilon=self.epsilon)
    
    def load(self, path: str):
        data = np.load(path)
        self.weights = data['weights']
        self.bias = data['bias']
        self.epsilon = float(data['epsilon'])


class TradingEnvironment:
    """Simulated trading environment for RL"""
    
    def __init__(self, candles: List[dict], initial_capital: float = 100000):
        self.candles = candles
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self) -> dict:
        self.current_step = 50  # Need history for indicators
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        return self._get_observation()
    
    def _get_observation(self) -> dict:
        if self.current_step >= len(self.candles):
            return {}
        
        candle = self.candles[self.current_step]
        history = self.candles[max(0, self.current_step-20):self.current_step+1]
        closes = [c["close"] for c in history]
        
        current_price = candle["close"]
        unrealized_pnl = 0
        if self.position != 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
        
        return {
            "price": current_price,
            "returns_1": (closes[-1] / closes[-2] - 1) * 100 if len(closes) > 1 else 0,
            "returns_5": (closes[-1] / closes[-6] - 1) * 100 if len(closes) > 5 else 0,
            "returns_10": (closes[-1] / closes[-11] - 1) * 100 if len(closes) > 10 else 0,
            "rsi": self._compute_rsi(closes),
            "macd": 0,
            "atr_pct": 1,
            "volume_ratio": 1,
            "bb_position": 0.5,
            "sma_cross": 1 if np.mean(closes[-10:]) > np.mean(closes[-20:]) else -1,
            "ema_cross": 0,
            "position": self.position,
            "unrealized_pnl": unrealized_pnl,
            "hour": 12,
            "day_of_week": 2
        }
    
    def _compute_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices[-period-1:])
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def step(self, action: int) -> Tuple[dict, float, bool]:
        """Execute action and return (next_obs, reward, done)"""
        if self.current_step >= len(self.candles) - 1:
            return {}, 0, True
        
        current_price = self.candles[self.current_step]["close"]
        next_price = self.candles[self.current_step + 1]["close"]
        reward = 0
        
        if action == Action.BUY and self.position == 0:
            self.position = 1
            self.entry_price = current_price
        
        elif action == Action.SELL and self.position == 1:
            pnl = next_price - self.entry_price
            reward = pnl / self.entry_price * 100
            self.capital += pnl
            self.trades.append({"pnl": pnl, "return_pct": reward})
            self.position = 0
            self.entry_price = 0
        
        elif action == Action.HOLD and self.position == 1:
            # Small penalty for holding
            reward = (next_price - current_price) / current_price * 100 - 0.01
        
        self.current_step += 1
        done = self.current_step >= len(self.candles) - 1
        
        return self._get_observation(), reward, done
    
    def get_metrics(self) -> dict:
        if not self.trades:
            return {"total_trades": 0, "win_rate": 0, "total_return": 0}
        
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        return {
            "total_trades": len(self.trades),
            "win_rate": wins / len(self.trades) * 100,
            "total_return": (self.capital - self.initial_capital) / self.initial_capital * 100,
            "avg_trade_return": np.mean([t["return_pct"] for t in self.trades])
        }


class RLTradingService:
    def __init__(self):
        self.config = RLConfig()
        self.agent = DQNAgent(self.config)
        self.status = TrainingStatus()
        self.training_task = None
    
    async def train(self, candles: List[dict], episodes: int = 100):
        """Train the RL agent"""
        self.status.is_training = True
        self.status.total_episodes = episodes
        
        env = TradingEnvironment(candles)
        
        for episode in range(episodes):
            obs = env.reset()
            state = self.agent.get_state(obs)
            total_reward = 0
            
            while True:
                action = self.agent.act(state)
                next_obs, reward, done = env.step(action)
                
                if not next_obs:
                    break
                
                next_state = self.agent.get_state(next_obs)
                self.agent.remember(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                
                if len(self.agent.memory) > self.config.batch_size:
                    self.agent.replay()
                
                if done:
                    break
            
            metrics = env.get_metrics()
            self.status.episode = episode + 1
            self.status.epsilon = self.agent.epsilon
            self.status.avg_reward = total_reward
            self.status.total_profit = metrics.get("total_return", 0)
            self.status.win_rate = metrics.get("win_rate", 0)
        
        self.status.is_training = False
        return self.status
    
    def predict(self, market_data: dict) -> dict:
        """Get action prediction"""
        state = self.agent.get_state(market_data)
        action = self.agent.act(state, training=False)
        q_values = np.dot(state, self.agent.weights) + self.agent.bias
        
        return {
            "action": Action(action).name,
            "confidence": float(np.max(q_values) / (np.sum(np.abs(q_values)) + 0.001)),
            "q_values": {"HOLD": float(q_values[0]), "BUY": float(q_values[1]), "SELL": float(q_values[2])}
        }


service = RLTradingService()


@app.get("/health")
async def health():
    return {"status": "healthy", "is_training": service.status.is_training}


@app.get("/status")
async def get_status():
    return service.status


@app.post("/train")
async def train(candles: List[dict], episodes: int = 100, background_tasks: BackgroundTasks = None):
    if not candles:
        raise HTTPException(status_code=400, detail="candles is required (real OHLCV data)")
    result = await service.train(candles, episodes)
    return result


@app.post("/predict")
async def predict(market_data: dict):
    return service.predict(market_data)


@app.get("/config")
async def get_config():
    return service.config


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8018, reload=True)
