"""
Utility Functions for Indian AI Trader
Common helper functions used across all services
"""
from datetime import datetime, time, timedelta
from typing import Optional, List, Tuple
import pytz

# Indian Standard Time
IST = pytz.timezone("Asia/Kolkata")


def get_ist_now() -> datetime:
    """Get current datetime in IST"""
    return datetime.now(IST)


def to_ist(dt: datetime) -> datetime:
    """Convert datetime to IST"""
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(IST)


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC"""
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.astimezone(pytz.UTC)


# NSE Market Hours
NSE_OPEN = time(9, 15)  # 9:15 AM IST
NSE_CLOSE = time(15, 30)  # 3:30 PM IST
NSE_PRE_OPEN_START = time(9, 0)  # 9:00 AM IST
NSE_PRE_OPEN_END = time(9, 8)  # 9:08 AM IST


def is_market_hours(exchange: str = "NSE", dt: Optional[datetime] = None) -> bool:
    """
    Check if market is open for given exchange
    
    Args:
        exchange: Exchange code (NSE, BSE)
        dt: Datetime to check (defaults to now)
    
    Returns:
        True if market is open
    """
    if dt is None:
        dt = get_ist_now()
    else:
        dt = to_ist(dt)
    
    # Check if weekend
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    current_time = dt.time()
    
    # NSE/BSE regular market hours
    if exchange.upper() in ["NSE", "BSE"]:
        return NSE_OPEN <= current_time <= NSE_CLOSE
    
    # Add more exchanges as needed
    return False


def get_market_status(exchange: str = "NSE", dt: Optional[datetime] = None) -> str:
    """
    Get detailed market status
    
    Returns:
        Status string: OPEN, CLOSED, PRE_OPEN, POST_CLOSE
    """
    if dt is None:
        dt = get_ist_now()
    else:
        dt = to_ist(dt)
    
    # Weekend
    if dt.weekday() >= 5:
        return "CLOSED"
    
    current_time = dt.time()
    
    if current_time < NSE_PRE_OPEN_START:
        return "CLOSED"
    elif NSE_PRE_OPEN_START <= current_time < NSE_OPEN:
        return "PRE_OPEN"
    elif NSE_OPEN <= current_time <= NSE_CLOSE:
        return "OPEN"
    else:
        return "CLOSED"


def get_next_market_open(exchange: str = "NSE", dt: Optional[datetime] = None) -> datetime:
    """Get next market open datetime"""
    if dt is None:
        dt = get_ist_now()
    else:
        dt = to_ist(dt)
    
    # If before market open today, return today's open
    if dt.time() < NSE_OPEN and dt.weekday() < 5:
        return dt.replace(hour=9, minute=15, second=0, microsecond=0)
    
    # Otherwise, find next trading day
    next_day = dt + timedelta(days=1)
    while next_day.weekday() >= 5:  # Skip weekend
        next_day += timedelta(days=1)
    
    return next_day.replace(hour=9, minute=15, second=0, microsecond=0)


def format_currency(amount: float, currency: str = "INR") -> str:
    """
    Format amount as Indian currency
    
    Args:
        amount: Amount to format
        currency: Currency code
    
    Returns:
        Formatted string (e.g., "₹1,23,456.78")
    """
    if currency == "INR":
        # Indian numbering system
        is_negative = amount < 0
        amount = abs(amount)
        
        s = f"{amount:,.2f}"
        # Convert to Indian format (lakhs, crores)
        parts = s.split(".")
        integer_part = parts[0].replace(",", "")
        decimal_part = parts[1] if len(parts) > 1 else "00"
        
        if len(integer_part) > 3:
            # Last 3 digits
            result = integer_part[-3:]
            remaining = integer_part[:-3]
            
            # Add commas every 2 digits
            while remaining:
                result = remaining[-2:] + "," + result
                remaining = remaining[:-2]
            
            formatted = f"₹{'-' if is_negative else ''}{result}.{decimal_part}"
        else:
            formatted = f"₹{'-' if is_negative else ''}{integer_part}.{decimal_part}"
        
        return formatted
    
    # Default formatting
    return f"{currency} {amount:,.2f}"


def calculate_pnl_percent(entry_price: float, current_price: float, side: str = "BUY") -> float:
    """
    Calculate PnL percentage
    
    Args:
        entry_price: Entry price
        current_price: Current market price
        side: BUY or SELL
    
    Returns:
        PnL as percentage
    """
    if entry_price == 0:
        return 0.0
    
    if side.upper() == "BUY":
        return ((current_price - entry_price) / entry_price) * 100
    else:  # SELL (short)
        return ((entry_price - current_price) / entry_price) * 100


def calculate_position_value(quantity: int, price: float) -> float:
    """Calculate total position value"""
    return quantity * price


def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """Round price to nearest tick size"""
    return round(price / tick_size) * tick_size


def get_lot_size_multiple(quantity: int, lot_size: int) -> Tuple[int, int]:
    """
    Get quantity in lot multiples
    
    Returns:
        Tuple of (lots, remainder)
    """
    lots = quantity // lot_size
    remainder = quantity % lot_size
    return lots, remainder


def parse_trading_symbol(symbol: str) -> dict:
    """
    Parse trading symbol to extract components
    
    E.g., "NIFTY24DEC23500CE" -> {
        "underlying": "NIFTY",
        "expiry": "24DEC",
        "strike": 23500,
        "option_type": "CE"
    }
    """
    # Basic implementation - extend as needed
    result = {"raw": symbol}
    
    # Try to detect if it's an option
    if symbol.endswith("CE") or symbol.endswith("PE"):
        result["option_type"] = symbol[-2:]
        symbol = symbol[:-2]
        
        # Try to extract strike (last numeric part)
        strike_str = ""
        for i in range(len(symbol) - 1, -1, -1):
            if symbol[i].isdigit():
                strike_str = symbol[i] + strike_str
            else:
                break
        
        if strike_str:
            result["strike"] = int(strike_str)
            symbol = symbol[:-len(strike_str)]
        
        result["underlying"] = symbol
    else:
        result["underlying"] = symbol
    
    return result


def generate_snapshot_id(timestamp: Optional[datetime] = None) -> str:
    """Generate a unique snapshot ID for data versioning"""
    if timestamp is None:
        timestamp = get_ist_now()
    
    return timestamp.strftime("%Y%m%d_%H%M%S")


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator"""
    if denominator == 0:
        return default
    return numerator / denominator
