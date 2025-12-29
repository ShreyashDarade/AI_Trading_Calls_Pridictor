"""
Backtest Service
Walk-forward backtesting with historical data
"""
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Backtest Service",
    description="Walk-forward backtesting engine",
    version="1.0.0"
)


# ============================================
# MODELS
# ============================================

class BacktestStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class StrategyConfig(BaseModel):
    name: str = "AI Signal Default"
    timeframe: str = "15m"
    min_confidence: float = 0.7
    stop_loss_pct: float = 2.0
    target_pct: float = 4.0
    max_positions: int = 5
    position_size_pct: float = 5.0
    
    # Custom parameters
    parameters: Dict[str, Any] = {}


class BacktestRequest(BaseModel):
    name: str
    description: Optional[str] = None
    universe: str = "NIFTY50"  # NIFTY50, BANKNIFTY, CUSTOM
    symbols: Optional[List[str]] = None
    start_date: date
    end_date: date
    initial_capital: float = 1000000.0
    strategy: StrategyConfig = StrategyConfig()


class Trade(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    direction: str  # LONG, SHORT
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_reason: Optional[str] = None  # TARGET, STOP_LOSS, TIMEOUT
    signal_confidence: float


class BacktestResult(BaseModel):
    backtest_id: str
    name: str
    status: BacktestStatus
    progress_percent: int = 0
    
    # Config
    start_date: date
    end_date: date
    initial_capital: float
    strategy: str
    
    # Results (populated when complete)
    final_value: Optional[float] = None
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    total_trades: Optional[int] = None
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    # Trades
    trades: List[Trade] = []
    equity_curve: List[Dict] = []


# ============================================
# BACKTEST ENGINE
# ============================================

class BacktestEngine:
    """Walk-forward backtesting engine"""
    
    def __init__(self):
        self.backtests: Dict[str, BacktestResult] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def run_backtest(self, request: BacktestRequest) -> str:
        """Start a new backtest"""
        backtest_id = str(uuid4())
        
        result = BacktestResult(
            backtest_id=backtest_id,
            name=request.name,
            status=BacktestStatus.PENDING,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            strategy=request.strategy.name
        )
        
        self.backtests[backtest_id] = result
        
        return backtest_id
    
    async def execute_backtest(self, backtest_id: str, request: BacktestRequest):
        """Execute backtest in background"""
        result = self.backtests[backtest_id]
        result.status = BacktestStatus.RUNNING
        result.started_at = datetime.utcnow()
        
        try:
            # Get symbols
            symbols = request.symbols or await self._get_universe_symbols(request.universe)
            
            # Get historical data
            # TODO: Implement actual data fetching from ClickHouse
            
            # Simulate backtest execution
            total_days = (request.end_date - request.start_date).days
            
            capital = request.initial_capital
            trades = []
            equity_curve = []
            
            import random
            
            current_date = request.start_date
            while current_date <= request.end_date:
                # Update progress
                days_processed = (current_date - request.start_date).days
                result.progress_percent = int((days_processed / total_days) * 100)
                
                # Simulate trading day
                for symbol in random.sample(symbols, min(3, len(symbols))):
                    if random.random() > 0.7:  # 30% chance of signal
                        # Generate mock signal
                        confidence = random.uniform(0.6, 0.95)
                        
                        if confidence >= request.strategy.min_confidence:
                            entry_price = random.uniform(500, 3000)
                            direction = random.choice(["LONG", "SHORT"])
                            
                            # Simulate trade outcome
                            outcome = random.random()
                            
                            if outcome < 0.6:  # 60% win rate
                                exit_price = entry_price * (1 + request.strategy.target_pct / 100)
                                exit_reason = "TARGET"
                            else:
                                exit_price = entry_price * (1 - request.strategy.stop_loss_pct / 100)
                                exit_reason = "STOP_LOSS"
                            
                            if direction == "SHORT":
                                exit_price = 2 * entry_price - exit_price
                            
                            qty = int((capital * request.strategy.position_size_pct / 100) / entry_price)
                            pnl = (exit_price - entry_price) * qty
                            if direction == "SHORT":
                                pnl = -pnl
                            
                            capital += pnl
                            
                            trade = Trade(
                                symbol=symbol,
                                direction=direction,
                                entry_time=datetime.combine(current_date, datetime.min.time()),
                                exit_time=datetime.combine(current_date, datetime.min.time()) + timedelta(hours=random.randint(1, 6)),
                                entry_price=entry_price,
                                exit_price=exit_price,
                                quantity=qty,
                                pnl=pnl,
                                pnl_percent=(pnl / (entry_price * qty)) * 100,
                                exit_reason=exit_reason,
                                signal_confidence=confidence
                            )
                            trades.append(trade)
                
                # Record equity
                equity_curve.append({
                    "date": current_date.isoformat(),
                    "equity": capital
                })
                
                current_date += timedelta(days=1)
                
                # Small delay to simulate processing
                await asyncio.sleep(0.01)
            
            # Calculate final metrics
            winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl and t.pnl <= 0]
            
            result.final_value = capital
            result.total_return = ((capital / request.initial_capital) - 1) * 100
            result.total_trades = len(trades)
            result.winning_trades = len(winning_trades)
            result.losing_trades = len(losing_trades)
            result.win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
            
            if winning_trades:
                result.avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
            if losing_trades:
                result.avg_loss = sum(abs(t.pnl) for t in losing_trades) / len(losing_trades)
            
            if result.avg_loss and result.avg_loss > 0:
                gross_profit = sum(t.pnl for t in winning_trades)
                gross_loss = abs(sum(t.pnl for t in losing_trades))
                result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calculate max drawdown
            peak = request.initial_capital
            max_dd = 0
            for point in equity_curve:
                equity = point["equity"]
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown = max_dd
            
            # Calculate Sharpe ratio (simplified)
            if equity_curve:
                returns = []
                for i in range(1, len(equity_curve)):
                    ret = (equity_curve[i]["equity"] / equity_curve[i-1]["equity"]) - 1
                    returns.append(ret)
                
                if returns:
                    import numpy as np
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    result.sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
            result.trades = trades[-100:]  # Store last 100 trades
            result.equity_curve = equity_curve
            result.status = BacktestStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            result.progress_percent = 100
            
            if result.started_at:
                result.duration_seconds = int((result.completed_at - result.started_at).total_seconds())
            
            logger.info(f"Backtest {backtest_id} completed: {result.total_return:.2f}% return")
            
        except Exception as e:
            logger.error(f"Backtest {backtest_id} failed: {e}")
            result.status = BacktestStatus.FAILED
            result.completed_at = datetime.utcnow()
    
    async def _get_universe_symbols(self, universe: str) -> List[str]:
        """Get symbols for a universe"""
        universes = {
            "NIFTY50": [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
                "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
                "TITAN", "BAJFINANCE", "WIPRO", "HCLTECH", "ULTRACEMCO"
            ],
            "BANKNIFTY": [
                "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
                "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB"
            ],
        }
        return universes.get(universe, universes["NIFTY50"])
    
    def get_backtest(self, backtest_id: str) -> Optional[BacktestResult]:
        """Get backtest status and results"""
        return self.backtests.get(backtest_id)
    
    def list_backtests(self, limit: int = 20) -> List[BacktestResult]:
        """List all backtests"""
        results = list(self.backtests.values())
        results.sort(key=lambda x: x.started_at or datetime.min, reverse=True)
        return results[:limit]
    
    def cancel_backtest(self, backtest_id: str) -> bool:
        """Cancel a running backtest"""
        if backtest_id in self.running_tasks:
            self.running_tasks[backtest_id].cancel()
            if backtest_id in self.backtests:
                self.backtests[backtest_id].status = BacktestStatus.CANCELLED
            return True
        return False


# Global engine
engine = BacktestEngine()


# ============================================
# ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/backtests")
async def create_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
):
    """Create and start a new backtest"""
    backtest_id = await engine.run_backtest(request)
    
    # Run in background
    background_tasks.add_task(engine.execute_backtest, backtest_id, request)
    
    return {
        "backtest_id": backtest_id,
        "message": "Backtest started",
        "status": "PENDING"
    }


@app.get("/backtests")
async def list_backtests(limit: int = 20):
    """List all backtests"""
    results = engine.list_backtests(limit)
    
    # Return summary without full trade list
    return {
        "backtests": [
            {
                "backtest_id": r.backtest_id,
                "name": r.name,
                "status": r.status,
                "progress_percent": r.progress_percent,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "total_return": r.total_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "total_trades": r.total_trades,
                "completed_at": r.completed_at
            }
            for r in results
        ]
    }


@app.get("/backtests/{backtest_id}")
async def get_backtest(backtest_id: str):
    """Get backtest details"""
    result = engine.get_backtest(backtest_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return result


@app.get("/backtests/{backtest_id}/trades")
async def get_backtest_trades(backtest_id: str, limit: int = 100):
    """Get trades for a backtest"""
    result = engine.get_backtest(backtest_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return {"trades": result.trades[:limit]}


@app.get("/backtests/{backtest_id}/equity")
async def get_backtest_equity(backtest_id: str):
    """Get equity curve for a backtest"""
    result = engine.get_backtest(backtest_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return {"equity_curve": result.equity_curve}


@app.delete("/backtests/{backtest_id}")
async def cancel_backtest(backtest_id: str):
    """Cancel a running backtest"""
    if engine.cancel_backtest(backtest_id):
        return {"message": "Backtest cancelled"}
    raise HTTPException(status_code=400, detail="Cannot cancel backtest")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006, reload=True)
