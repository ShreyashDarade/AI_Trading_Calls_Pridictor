"""
Position Sizing Algorithms
Different methods for calculating optimal position sizes
"""
from typing import Optional
import numpy as np


def fixed_fractional(
    capital: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float
) -> int:
    """
    Fixed Fractional Position Sizing
    
    Risks a fixed percentage of capital per trade.
    
    Args:
        capital: Available trading capital
        risk_percent: Percentage of capital to risk (e.g., 1.0 for 1%)
        entry_price: Expected entry price
        stop_loss: Stop loss price
    
    Returns:
        Number of shares to buy
    
    Example:
        capital = 100000
        risk_percent = 1.0  # Risk 1% = ₹1000
        entry_price = 500
        stop_loss = 490
        # Risk per share = 500 - 490 = 10
        # Shares = 1000 / 10 = 100
    """
    if entry_price <= 0 or stop_loss <= 0:
        return 0
    
    risk_per_share = abs(entry_price - stop_loss)
    
    if risk_per_share == 0:
        return 0
    
    risk_amount = capital * (risk_percent / 100)
    shares = int(risk_amount / risk_per_share)
    
    return max(0, shares)


def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    conservative_factor: float = 0.5
) -> float:
    """
    Kelly Criterion Position Sizing
    
    Mathematically optimal fraction of capital to bet.
    
    Args:
        win_rate: Historical win rate (0 to 1)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive number)
        conservative_factor: Fraction of Kelly to use (0.5 = half Kelly)
    
    Returns:
        Fraction of capital to allocate (0 to 1)
    
    Formula:
        Kelly% = W - [(1-W) / R]
        Where: W = win rate, R = win/loss ratio
    """
    if avg_loss == 0 or win_rate < 0 or win_rate > 1:
        return 0
    
    win_loss_ratio = avg_win / avg_loss
    
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Apply conservative factor
    kelly = kelly * conservative_factor
    
    # Clamp to reasonable range
    return max(0, min(kelly, 0.25))  # Never more than 25%


def volatility_adjusted_sizing(
    capital: float,
    target_volatility: float,
    instrument_volatility: float,
    price: float
) -> int:
    """
    Volatility-Adjusted Position Sizing
    
    Adjusts position size based on instrument volatility.
    More volatile instruments get smaller positions.
    
    Args:
        capital: Available trading capital
        target_volatility: Target portfolio volatility (annualized, e.g., 0.15 for 15%)
        instrument_volatility: Instrument's volatility (annualized)
        price: Current price
    
    Returns:
        Number of shares
    """
    if instrument_volatility <= 0 or price <= 0:
        return 0
    
    # Calculate position value for target volatility
    position_value = capital * (target_volatility / instrument_volatility)
    
    # Convert to shares
    shares = int(position_value / price)
    
    return max(0, shares)


def max_position_size(
    capital: float,
    price: float,
    max_position_percent: float = 5.0,
    lot_size: int = 1
) -> int:
    """
    Maximum Position Size
    
    Limits position to a maximum percentage of capital.
    
    Args:
        capital: Available trading capital
        price: Current price
        max_position_percent: Maximum position as % of capital (default 5%)
        lot_size: Minimum lot size (for F&O)
    
    Returns:
        Maximum number of shares (rounded to lot size)
    """
    if price <= 0:
        return 0
    
    max_value = capital * (max_position_percent / 100)
    max_shares = int(max_value / price)
    
    # Round to lot size
    if lot_size > 1:
        max_shares = (max_shares // lot_size) * lot_size
    
    return max(0, max_shares)


def calculate_position_value(quantity: int, price: float) -> float:
    """Calculate position value"""
    return quantity * price


def calculate_risk_amount(
    quantity: int,
    entry_price: float,
    stop_loss: float
) -> float:
    """Calculate amount at risk"""
    risk_per_share = abs(entry_price - stop_loss)
    return quantity * risk_per_share


def suggested_stop_loss(
    entry_price: float,
    atr: float,
    atr_multiplier: float = 2.0,
    side: str = "BUY"
) -> float:
    """
    Calculate suggested stop loss based on ATR
    
    Args:
        entry_price: Entry price
        atr: Average True Range value
        atr_multiplier: How many ATRs for stop (default 2)
        side: BUY or SELL
    
    Returns:
        Suggested stop loss price
    """
    stop_distance = atr * atr_multiplier
    
    if side.upper() == "BUY":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance


def risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    target: float
) -> float:
    """
    Calculate Risk/Reward Ratio
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        target: Target price
    
    Returns:
        Risk/Reward ratio (higher is better)
    """
    risk = abs(entry_price - stop_loss)
    reward = abs(target - entry_price)
    
    if risk == 0:
        return 0
    
    return reward / risk
