"""
Order Service
Handles order placement with Groww API (Paper + Live trading)
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Order Service",
    description="Order management and execution via Groww API",
    version="1.0.0"
)


# ============================================
# MODELS
# ============================================

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ProductType(str, Enum):
    CNC = "CNC"  # Delivery
    MIS = "MIS"  # Intraday


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TRIGGER_PENDING = "TRIGGER_PENDING"


class TradingMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class PlaceOrderRequest(BaseModel):
    symbol: str
    exchange: str = "NSE"
    transaction_type: TransactionType
    order_type: OrderType = OrderType.MARKET
    product_type: ProductType = ProductType.MIS
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    disclosed_quantity: Optional[int] = None
    validity: str = "DAY"
    signal_id: Optional[str] = None
    mode: TradingMode = TradingMode.PAPER


class Order(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    exchange: str
    exchange_token: Optional[str] = None
    transaction_type: TransactionType
    order_type: OrderType
    product_type: ProductType
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    mode: TradingMode
    signal_id: Optional[str] = None
    
    # Execution details
    broker_order_id: Optional[str] = None
    filled_quantity: int = 0
    average_price: Optional[float] = None
    exchange_timestamp: Optional[datetime] = None
    
    # Audit
    rejection_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ModifyOrderRequest(BaseModel):
    quantity: Optional[int] = None
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    order_type: Optional[OrderType] = None


# ============================================
# ORDER EXECUTOR
# ============================================

class OrderExecutor:
    """Handles order execution in paper and live modes"""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.groww_client = None
    
    async def initialize(self, api_key: str, access_token: str):
        """Initialize Groww client for live trading"""
        try:
            from libs.groww_client.src.client import GrowwClient
            self.groww_client = GrowwClient(api_key, access_token)
            logger.info("Groww client initialized for live trading")
        except Exception as e:
            logger.error(f"Failed to initialize Groww client: {e}")
    
    async def place_order(self, request: PlaceOrderRequest) -> Order:
        """Place a new order"""
        order = Order(
            symbol=request.symbol.upper(),
            exchange=request.exchange.upper(),
            transaction_type=request.transaction_type,
            order_type=request.order_type,
            product_type=request.product_type,
            quantity=request.quantity,
            price=request.price,
            trigger_price=request.trigger_price,
            mode=request.mode,
            signal_id=request.signal_id
        )
        
        if request.mode == TradingMode.PAPER:
            order = await self._execute_paper_order(order)
        else:
            order = await self._execute_live_order(order)
        
        self.orders[order.id] = order
        return order
    
    async def _execute_paper_order(self, order: Order) -> Order:
        """Execute order in paper trading mode (simulated)"""
        try:
            # Get current market price from Groww or cache
            current_price = await self._get_current_price(order.symbol)
            
            if order.order_type == OrderType.MARKET:
                # Execute immediately at current price
                order.average_price = current_price
                order.filled_quantity = order.quantity
                order.status = OrderStatus.COMPLETED
                order.exchange_timestamp = datetime.utcnow()
                logger.info(f"Paper order executed: {order.transaction_type} {order.quantity} {order.symbol} @ {current_price}")
                
            elif order.order_type == OrderType.LIMIT:
                # Check if limit can be filled
                if order.transaction_type == TransactionType.BUY:
                    if current_price <= order.price:
                        order.average_price = order.price
                        order.filled_quantity = order.quantity
                        order.status = OrderStatus.COMPLETED
                    else:
                        order.status = OrderStatus.OPEN
                else:
                    if current_price >= order.price:
                        order.average_price = order.price
                        order.filled_quantity = order.quantity
                        order.status = OrderStatus.COMPLETED
                    else:
                        order.status = OrderStatus.OPEN
                        
            elif order.order_type in [OrderType.SL, OrderType.SL_M]:
                order.status = OrderStatus.TRIGGER_PENDING
            
            order.updated_at = datetime.utcnow()
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = str(e)
            logger.error(f"Paper order failed: {e}")
        
        return order
    
    async def _execute_live_order(self, order: Order) -> Order:
        """Execute order via Groww API (live trading)"""
        if not self.groww_client:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "Groww client not initialized"
            return order
        
        try:
            # Place order via Groww API
            response = await self.groww_client.place_order(
                exchange=order.exchange,
                symbol=order.symbol,
                transaction_type=order.transaction_type.value,
                order_type=order.order_type.value,
                product_type=order.product_type.value,
                quantity=order.quantity,
                price=order.price,
                trigger_price=order.trigger_price
            )
            
            if response.get("status") == "success":
                order.broker_order_id = response.get("order_id")
                order.status = OrderStatus.OPEN
                logger.info(f"Live order placed: {order.broker_order_id}")
            else:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = response.get("message", "Order rejected")
                
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = str(e)
            logger.error(f"Live order failed: {e}")
        
        order.updated_at = datetime.utcnow()
        return order
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol"""
        if self.groww_client:
            try:
                from libs.groww_client.src.live_data import LiveDataClient
                live_client = LiveDataClient(
                    self.groww_client.api_key,
                    self.groww_client.access_token
                )
                quote = await live_client.get_ltp("NSE", symbol)
                if quote:
                    return quote.get("ltp", 0)
            except Exception as e:
                logger.warning(f"Failed to get price from Groww: {e}")
        
        # Fallback: return 0 (should not happen in production)
        raise HTTPException(status_code=503, detail="Unable to get market price")
    
    async def modify_order(self, order_id: str, request: ModifyOrderRequest) -> Order:
        """Modify an existing order"""
        if order_id not in self.orders:
            raise HTTPException(status_code=404, detail="Order not found")
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.OPEN, OrderStatus.TRIGGER_PENDING]:
            raise HTTPException(status_code=400, detail="Order cannot be modified")
        
        if order.mode == TradingMode.LIVE and self.groww_client:
            # Modify via Groww API
            await self.groww_client.modify_order(
                order.broker_order_id,
                quantity=request.quantity,
                price=request.price,
                trigger_price=request.trigger_price
            )
        
        # Update local order
        if request.quantity:
            order.quantity = request.quantity
        if request.price:
            order.price = request.price
        if request.trigger_price:
            order.trigger_price = request.trigger_price
        if request.order_type:
            order.order_type = request.order_type
        
        order.updated_at = datetime.utcnow()
        return order
    
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an order"""
        if order_id not in self.orders:
            raise HTTPException(status_code=404, detail="Order not found")
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.OPEN, OrderStatus.TRIGGER_PENDING, OrderStatus.PENDING]:
            raise HTTPException(status_code=400, detail="Order cannot be cancelled")
        
        if order.mode == TradingMode.LIVE and self.groww_client:
            # Cancel via Groww API
            await self.groww_client.cancel_order(order.broker_order_id)
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)
    
    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        mode: Optional[TradingMode] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        orders = list(self.orders.values())
        
        if status:
            orders = [o for o in orders if o.status == status]
        if mode:
            orders = [o for o in orders if o.mode == mode]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        
        orders.sort(key=lambda x: x.created_at, reverse=True)
        return orders[:limit]


# Global executor
executor = OrderExecutor()


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    api_key = os.getenv("GROWW_API_KEY")
    access_token = os.getenv("GROWW_ACCESS_TOKEN")
    
    if api_key and access_token:
        await executor.initialize(api_key, access_token)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "groww_connected": executor.groww_client is not None,
        "total_orders": len(executor.orders)
    }


@app.post("/orders", response_model=Order)
async def place_order(request: PlaceOrderRequest):
    """Place a new order"""
    return await executor.place_order(request)


@app.get("/orders")
async def get_orders(
    status: Optional[OrderStatus] = None,
    mode: Optional[TradingMode] = None,
    symbol: Optional[str] = None,
    limit: int = 100
):
    """Get orders with filters"""
    orders = executor.get_orders(status, mode, symbol, limit)
    return {"orders": orders}


@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get specific order"""
    order = executor.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@app.put("/orders/{order_id}")
async def modify_order(order_id: str, request: ModifyOrderRequest):
    """Modify an order"""
    return await executor.modify_order(order_id, request)


@app.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an order"""
    return await executor.cancel_order(order_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009, reload=True)
