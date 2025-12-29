"""
Shared Schemas/Models for Indian AI Trader
Pydantic models used across all microservices
"""
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import uuid


# ============================================
# ENUMS
# ============================================

class Exchange(str, Enum):
    """Supported exchanges"""
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"  # NSE F&O
    BFO = "BFO"  # BSE F&O
    MCX = "MCX"  # Commodity


class Segment(str, Enum):
    """Market segments"""
    CASH = "CASH"
    FNO = "FNO"
    CURRENCY = "CUR"
    COMMODITY = "COM"


class InstrumentType(str, Enum):
    """Instrument types"""
    EQUITY = "EQ"
    FUTURE = "FUT"
    CALL_OPTION = "CE"
    PUT_OPTION = "PE"
    INDEX = "INDEX"


class SignalAction(str, Enum):
    """Trading signal actions"""
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


class OrderSide(str, Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"


class OrderStatus(str, Enum):
    """Order status"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Timeframe(str, Enum):
    """Candle timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


# ============================================
# INSTRUMENT MODELS
# ============================================

class Instrument(BaseModel):
    """Instrument/Symbol definition from Groww CSV"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exchange: Exchange
    segment: Segment
    instrument_type: InstrumentType
    trading_symbol: str
    groww_symbol: str
    exchange_token: str  # Required for streaming subscriptions
    name: str
    isin: Optional[str] = None
    lot_size: int = 1
    tick_size: float = 0.05
    expiry_date: Optional[datetime] = None
    strike_price: Optional[float] = None
    is_tradeable: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class InstrumentSearch(BaseModel):
    """Instrument search request"""
    query: str
    exchange: Optional[Exchange] = None
    segment: Optional[Segment] = None
    limit: int = 20


# ============================================
# MARKET DATA MODELS
# ============================================

class Tick(BaseModel):
    """Real-time tick data"""
    instrument_id: str
    exchange: Exchange
    trading_symbol: str
    timestamp: datetime
    ltp: float  # Last traded price
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    bid_price: Optional[float] = None
    bid_qty: Optional[int] = None
    ask_price: Optional[float] = None
    ask_qty: Optional[int] = None
    oi: Optional[int] = None  # Open interest (for F&O)
    change: Optional[float] = None
    change_percent: Optional[float] = None
    
    class Config:
        use_enum_values = True


class MarketDepth(BaseModel):
    """Market depth (order book)"""
    price: float
    quantity: int
    orders: int


class Quote(BaseModel):
    """Full quote with depth"""
    instrument_id: str
    exchange: Exchange
    trading_symbol: str
    timestamp: datetime
    ltp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    avg_price: Optional[float] = None
    oi: Optional[int] = None
    oi_change: Optional[int] = None
    bid_depth: List[MarketDepth] = []
    ask_depth: List[MarketDepth] = []
    lower_circuit: Optional[float] = None
    upper_circuit: Optional[float] = None
    
    class Config:
        use_enum_values = True


class Candle(BaseModel):
    """OHLCV Candle"""
    instrument_id: str
    exchange: Exchange
    trading_symbol: str
    timeframe: Timeframe
    timestamp: datetime  # Candle open time
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: Optional[int] = None
    vwap: Optional[float] = None
    trades: Optional[int] = None
    
    class Config:
        use_enum_values = True


# ============================================
# FEATURE MODELS
# ============================================

class FeatureVector(BaseModel):
    """Computed feature vector for ML"""
    instrument_id: str
    timestamp: datetime
    timeframe: Timeframe
    version: str = "v1"
    
    # Price features
    returns_1: Optional[float] = None
    returns_5: Optional[float] = None
    returns_20: Optional[float] = None
    
    # Volatility
    volatility_20: Optional[float] = None
    atr_14: Optional[float] = None
    
    # Momentum
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Trend
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Volume
    volume_sma_20: Optional[float] = None
    volume_spike: Optional[float] = None
    
    # Regime
    trend_regime: Optional[str] = None  # UPTREND, DOWNTREND, SIDEWAYS
    volatility_regime: Optional[str] = None  # HIGH, MEDIUM, LOW
    
    class Config:
        use_enum_values = True


# ============================================
# SIGNAL MODELS
# ============================================

class Signal(BaseModel):
    """AI-generated trading signal"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instrument_id: str
    trading_symbol: str
    exchange: Exchange
    timestamp: datetime
    
    # Signal details
    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
    expected_return: Optional[float] = None
    expected_risk: Optional[float] = None
    
    # Risk parameters
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    position_size_percent: Optional[float] = None
    
    # Metadata
    model_version: str
    feature_version: str
    data_snapshot_id: Optional[str] = None
    timeframe: Timeframe
    
    # Explanations
    top_features: List[Dict[str, Any]] = []
    reason_codes: List[str] = []
    
    # Gating
    passed_liquidity_check: bool = True
    passed_volatility_filter: bool = True
    passed_risk_limits: bool = True
    is_valid: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


# ============================================
# ORDER & POSITION MODELS
# ============================================

class Order(BaseModel):
    """Order model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: Optional[str] = None
    instrument_id: str
    trading_symbol: str
    exchange: Exchange
    
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: Optional[float] = None
    
    is_paper: bool = True
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class Position(BaseModel):
    """Open position"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instrument_id: str
    trading_symbol: str
    exchange: Exchange
    
    side: OrderSide
    quantity: int
    avg_entry_price: float
    current_price: Optional[float] = None
    
    # PnL
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    
    # Risk
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    
    is_paper: bool = True
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


# ============================================
# AUDIT MODELS
# ============================================

class AuditPacket(BaseModel):
    """Immutable audit record for every signal/decision"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # What
    event_type: str  # SIGNAL_GENERATED, ORDER_PLACED, POSITION_OPENED, etc.
    
    # Context
    signal_id: Optional[str] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    instrument_id: Optional[str] = None
    
    # Model info
    model_version: Optional[str] = None
    feature_version: Optional[str] = None
    data_snapshot_id: Optional[str] = None
    
    # Decision data
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    reason_codes: List[str] = []
    
    # Metadata
    service_name: str
    service_version: str
    environment: str
    
    class Config:
        use_enum_values = True


# ============================================
# RESPONSE MODELS
# ============================================

class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel):
    """Paginated response"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


# ============================================
# WATCHLIST MODELS
# ============================================

class WatchlistItem(BaseModel):
    """Single watchlist item"""
    instrument_id: str
    trading_symbol: str
    exchange: Exchange
    added_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


class Watchlist(BaseModel):
    """User watchlist"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    user_id: str
    items: List[WatchlistItem] = []
    is_default: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
