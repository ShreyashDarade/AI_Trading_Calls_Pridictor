"""
Portfolio Service
Manages positions, P&L tracking, and portfolio analytics
"""
import logging
from datetime import datetime, date
from typing import Dict, List, Optional
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Portfolio Service",
    description="Portfolio management and P&L tracking",
    version="1.0.0"
)


# ============================================
# MODELS
# ============================================

class Position(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    instrument_id: str
    symbol: str
    exchange: str = "NSE"
    quantity: int
    average_price: float
    current_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    mode: str = "PAPER"  # PAPER or LIVE
    opened_at: datetime = Field(default_factory=datetime.utcnow)


class Order(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    instrument_id: str
    symbol: str
    order_type: str  # MARKET, LIMIT, SL, SL-M
    transaction_type: str  # BUY, SELL
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    status: str = "PENDING"
    mode: str = "PAPER"
    signal_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PortfolioSummary(BaseModel):
    capital: float
    invested: float
    available: float
    total_pnl: float
    total_pnl_percent: float
    positions_count: int
    orders_today: int


class CreateOrderRequest(BaseModel):
    symbol: str
    exchange: str = "NSE"
    order_type: str = "MARKET"
    transaction_type: str  # BUY, SELL
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    signal_id: Optional[str] = None


# ============================================
# IN-MEMORY STORAGE (Replace with PostgreSQL)
# ============================================

class PortfolioStore:
    def __init__(self):
        self.capital = 1000000.0  # 10 lakh starting capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.equity_history: List[Dict] = []
    
    def _update_pnl(self):
        """Update P&L for all positions"""
        for pos in self.positions.values():
            if pos.current_price:
                pos.pnl = (pos.current_price - pos.average_price) * pos.quantity
                pos.pnl_percent = ((pos.current_price / pos.average_price) - 1) * 100
    
    def get_summary(self) -> PortfolioSummary:
        """Get portfolio summary"""
        invested = sum(
            pos.average_price * pos.quantity
            for pos in self.positions.values()
        )
        
        current_value = sum(
            (pos.current_price or pos.average_price) * pos.quantity
            for pos in self.positions.values()
        )
        
        total_pnl = current_value - invested
        total_pnl_percent = (total_pnl / invested * 100) if invested > 0 else 0
        
        available = self.capital - invested
        
        today = date.today()
        orders_today = len([
            o for o in self.orders
            if o.created_at.date() == today
        ])
        
        return PortfolioSummary(
            capital=self.capital,
            invested=invested,
            available=available,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            positions_count=len(self.positions),
            orders_today=orders_today
        )
    
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        self._update_pnl()
        return list(self.positions.values())
    
    def update_price(self, symbol: str, price: float):
        """Update current price for a position"""
        if symbol in self.positions:
            self.positions[symbol].current_price = price
            self._update_pnl()
    
    def create_order(self, request: CreateOrderRequest) -> Order:
        """Create a new order (paper trading)"""
        order = Order(
            instrument_id=f"{request.exchange}:{request.symbol}",
            symbol=request.symbol,
            order_type=request.order_type,
            transaction_type=request.transaction_type,
            quantity=request.quantity,
            price=request.price,
            trigger_price=request.trigger_price,
            signal_id=request.signal_id,
            status="PENDING"
        )
        
        self.orders.append(order)
        
        # In paper trading, execute immediately at market price
        if request.order_type == "MARKET":
            self._execute_order(order)
        
        return order
    
    def _execute_order(self, order: Order):
        """Execute order (paper trading)"""
        if order.price is None:
            raise HTTPException(
                status_code=400,
                detail="Market execution requires a real price; pass price explicitly or integrate live quotes"
            )
        current_price = float(order.price)
        
        order.price = current_price
        order.status = "COMPLETED"
        
        # Update position
        if order.transaction_type == "BUY":
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_qty = pos.quantity + order.quantity
                total_value = (pos.average_price * pos.quantity) + (current_price * order.quantity)
                pos.average_price = total_value / total_qty
                pos.quantity = total_qty
            else:
                self.positions[order.symbol] = Position(
                    instrument_id=order.instrument_id,
                    symbol=order.symbol,
                    quantity=order.quantity,
                    average_price=current_price,
                    current_price=current_price
                )
        else:  # SELL
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos.quantity -= order.quantity
                if pos.quantity <= 0:
                    del self.positions[order.symbol]
        
        self._update_pnl()
        logger.info(f"Order executed: {order.transaction_type} {order.quantity} {order.symbol} @ {current_price}")


# Global store
store = PortfolioStore()


# ============================================
# ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/portfolio", response_model=PortfolioSummary)
async def get_portfolio():
    """Get portfolio summary"""
    return store.get_summary()


@app.get("/positions", response_model=List[Position])
async def get_positions():
    """Get all open positions"""
    return store.get_positions()


@app.get("/positions/{symbol}")
async def get_position(symbol: str):
    """Get specific position"""
    symbol = symbol.upper()
    if symbol not in store.positions:
        raise HTTPException(status_code=404, detail=f"No position for {symbol}")
    return store.positions[symbol]


@app.post("/orders", response_model=Order)
async def create_order(request: CreateOrderRequest):
    """Create a new order"""
    return store.create_order(request)


@app.get("/orders")
async def get_orders(
    status: Optional[str] = None,
    limit: int = Query(default=50, le=100)
):
    """Get orders"""
    orders = store.orders
    
    if status:
        orders = [o for o in orders if o.status == status.upper()]
    
    return {"orders": orders[-limit:]}


@app.post("/prices/update")
async def update_prices(prices: Dict[str, float]):
    """Update current prices for positions"""
    for symbol, price in prices.items():
        store.update_price(symbol.upper(), price)
    return {"updated": len(prices)}


@app.get("/equity-curve")
async def get_equity_curve(days: int = Query(default=30, le=365)):
    """Get equity curve for charting"""
    raise HTTPException(
        status_code=501,
        detail="Equity curve history is not implemented; requires persistence of portfolio snapshots"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007, reload=True)
