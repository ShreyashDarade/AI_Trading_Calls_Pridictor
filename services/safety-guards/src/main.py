"""
Safety Guards Service
Production trading safety controls and circuit breakers
"""
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from uuid import uuid4
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Safety Guards Service",
    description="Production trading safety controls",
    version="3.0.0"
)


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class TradingStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class SafetyConfig(BaseModel):
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_position_size_pct: float = 5.0
    max_positions: int = 10
    max_orders_per_minute: int = 10
    max_daily_trades: int = 50
    emergency_stop_loss_pct: float = 5.0
    auto_square_off_time: str = "15:15"
    disable_after_hours: bool = True


class SafetyAlert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    level: AlertLevel
    message: str
    rule_violated: str
    current_value: float
    threshold: float
    action_taken: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradingState(BaseModel):
    status: TradingStatus = TradingStatus.ACTIVE
    daily_pnl: float = 0
    daily_pnl_pct: float = 0
    current_drawdown_pct: float = 0
    open_positions: int = 0
    orders_this_minute: int = 0
    trades_today: int = 0
    last_order_time: Optional[datetime] = None
    alerts: List[SafetyAlert] = []


class SafetyGuardsEngine:
    """Core safety controls for live trading"""
    
    def __init__(self):
        self.config = SafetyConfig()
        self.state = TradingState()
        self.capital = 1000000.0
        self.peak_equity = 1000000.0
        self.alerts_history: List[SafetyAlert] = []
    
    def check_order(self, symbol: str, quantity: int, price: float, is_buy: bool) -> tuple:
        """Check if order passes all safety checks"""
        checks = []
        
        # Check 1: Trading status
        if self.state.status != TradingStatus.ACTIVE:
            return False, f"Trading is {self.state.status.value}", "STATUS_CHECK"
        
        # Check 2: Market hours
        if self.config.disable_after_hours and not self._is_market_hours():
            return False, "Trading disabled outside market hours", "MARKET_HOURS"
        
        # Check 3: Daily loss limit
        if self.state.daily_pnl_pct < -self.config.max_daily_loss_pct:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                f"Daily loss limit breached: {self.state.daily_pnl_pct:.2f}%",
                "MAX_DAILY_LOSS",
                abs(self.state.daily_pnl_pct),
                self.config.max_daily_loss_pct,
                "Trading paused"
            )
            self.state.status = TradingStatus.PAUSED
            return False, "Daily loss limit reached", "DAILY_LOSS"
        
        # Check 4: Drawdown limit
        if self.state.current_drawdown_pct > self.config.max_drawdown_pct:
            self._trigger_alert(
                AlertLevel.CRITICAL,
                f"Max drawdown breached: {self.state.current_drawdown_pct:.2f}%",
                "MAX_DRAWDOWN",
                self.state.current_drawdown_pct,
                self.config.max_drawdown_pct,
                "Trading stopped"
            )
            self.state.status = TradingStatus.STOPPED
            return False, "Maximum drawdown limit reached", "DRAWDOWN"
        
        # Check 5: Position count
        if is_buy and self.state.open_positions >= self.config.max_positions:
            return False, f"Max positions ({self.config.max_positions}) reached", "MAX_POSITIONS"
        
        # Check 6: Position size
        position_value = quantity * price
        position_pct = position_value / self.capital * 100
        if position_pct > self.config.max_position_size_pct:
            return False, f"Position size {position_pct:.1f}% exceeds limit {self.config.max_position_size_pct}%", "POSITION_SIZE"
        
        # Check 7: Order rate limit
        now = datetime.utcnow()
        if self.state.last_order_time:
            if (now - self.state.last_order_time).seconds < 60:
                if self.state.orders_this_minute >= self.config.max_orders_per_minute:
                    return False, "Order rate limit exceeded", "RATE_LIMIT"
            else:
                self.state.orders_this_minute = 0
        
        # Check 8: Daily trade limit
        if self.state.trades_today >= self.config.max_daily_trades:
            return False, "Daily trade limit reached", "DAILY_TRADES"
        
        return True, "All checks passed", "PASSED"
    
    def record_order(self, pnl: float = 0):
        """Record an executed order"""
        self.state.orders_this_minute += 1
        self.state.trades_today += 1
        self.state.last_order_time = datetime.utcnow()
        
        self.state.daily_pnl += pnl
        self.state.daily_pnl_pct = self.state.daily_pnl / self.capital * 100
        
        current_equity = self.capital + self.state.daily_pnl
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        self.state.current_drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity * 100
    
    def emergency_stop(self, reason: str = "Manual trigger"):
        """Trigger emergency stop"""
        self.state.status = TradingStatus.EMERGENCY_STOP
        self._trigger_alert(
            AlertLevel.EMERGENCY,
            f"Emergency stop triggered: {reason}",
            "EMERGENCY_STOP",
            0,
            0,
            "All trading halted, positions to be closed"
        )
    
    def resume_trading(self):
        """Resume trading after pause"""
        if self.state.status in [TradingStatus.PAUSED, TradingStatus.STOPPED]:
            self.state.status = TradingStatus.ACTIVE
            return True
        return False
    
    def reset_daily(self):
        """Reset daily counters"""
        self.state.daily_pnl = 0
        self.state.daily_pnl_pct = 0
        self.state.trades_today = 0
        self.state.orders_this_minute = 0
        if self.state.status == TradingStatus.PAUSED:
            self.state.status = TradingStatus.ACTIVE
    
    def _is_market_hours(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)
        return market_open <= now <= market_close
    
    def _trigger_alert(self, level: AlertLevel, message: str, rule: str, 
                       current: float, threshold: float, action: str):
        alert = SafetyAlert(
            level=level,
            message=message,
            rule_violated=rule,
            current_value=current,
            threshold=threshold,
            action_taken=action
        )
        self.state.alerts.append(alert)
        self.alerts_history.append(alert)
        logger.warning(f"[{level.value}] {message}")


engine = SafetyGuardsEngine()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "trading_status": engine.state.status.value,
        "alerts_count": len(engine.state.alerts)
    }


@app.get("/status")
async def get_status():
    return engine.state


@app.get("/config")
async def get_config():
    return engine.config


@app.post("/config")
async def update_config(config: SafetyConfig):
    engine.config = config
    return {"message": "Configuration updated"}


@app.post("/check-order")
async def check_order(symbol: str, quantity: int, price: float, is_buy: bool = True):
    passed, message, rule = engine.check_order(symbol, quantity, price, is_buy)
    return {"allowed": passed, "message": message, "rule": rule}


@app.post("/emergency-stop")
async def emergency_stop(reason: str = "Manual trigger"):
    engine.emergency_stop(reason)
    return {"message": "Emergency stop activated", "status": engine.state.status.value}


@app.post("/resume")
async def resume():
    if engine.resume_trading():
        return {"message": "Trading resumed", "status": engine.state.status.value}
    raise HTTPException(status_code=400, detail="Cannot resume from current state")


@app.post("/reset-daily")
async def reset_daily():
    engine.reset_daily()
    return {"message": "Daily counters reset"}


@app.get("/alerts")
async def get_alerts(limit: int = 50):
    return {"alerts": engine.alerts_history[-limit:]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8017, reload=True)
