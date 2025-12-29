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
            raise HTTPException(
                status_code=501,
                detail="Backtesting requires real historical candles from ClickHouse; synthetic/simulated backtests are disabled"
            )
            
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
