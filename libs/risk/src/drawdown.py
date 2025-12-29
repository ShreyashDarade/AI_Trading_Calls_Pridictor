"""
Drawdown Calculations and Limits
"""
from typing import List, Union, Dict, Any, Optional
import numpy as np


def calculate_drawdown(
    equity_curve: Union[List[float], np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Calculate drawdown series from equity curve
    
    Args:
        equity_curve: List of portfolio values over time
    
    Returns:
        Dict with 'drawdown' (absolute) and 'drawdown_percent' arrays
    """
    equity = np.array(equity_curve)
    n = len(equity)
    
    # Running maximum
    running_max = np.maximum.accumulate(equity)
    
    # Absolute drawdown
    drawdown = running_max - equity
    
    # Percentage drawdown
    drawdown_percent = np.where(
        running_max > 0,
        drawdown / running_max * 100,
        0
    )
    
    return {
        "drawdown": drawdown,
        "drawdown_percent": drawdown_percent,
        "running_max": running_max
    }


def max_drawdown(
    equity_curve: Union[List[float], np.ndarray]
) -> Dict[str, float]:
    """
    Calculate maximum drawdown
    
    Args:
        equity_curve: List of portfolio values over time
    
    Returns:
        Dict with max drawdown details
    """
    dd = calculate_drawdown(equity_curve)
    
    max_dd = np.max(dd["drawdown"])
    max_dd_percent = np.max(dd["drawdown_percent"])
    max_dd_idx = np.argmax(dd["drawdown_percent"])
    
    # Find drawdown start (peak before max drawdown)
    peak_idx = 0
    for i in range(max_dd_idx, -1, -1):
        if dd["drawdown_percent"][i] == 0:
            peak_idx = i
            break
    
    return {
        "max_drawdown": max_dd,
        "max_drawdown_percent": max_dd_percent,
        "peak_index": peak_idx,
        "trough_index": max_dd_idx,
        "peak_value": equity_curve[peak_idx] if peak_idx < len(equity_curve) else 0,
        "trough_value": equity_curve[max_dd_idx] if max_dd_idx < len(equity_curve) else 0
    }


def current_drawdown(
    equity_curve: Union[List[float], np.ndarray]
) -> Dict[str, float]:
    """
    Calculate current drawdown from last value
    
    Args:
        equity_curve: List of portfolio values
    
    Returns:
        Current drawdown details
    """
    equity = np.array(equity_curve)
    
    if len(equity) == 0:
        return {
            "drawdown": 0,
            "drawdown_percent": 0,
            "peak_value": 0,
            "current_value": 0
        }
    
    peak = np.max(equity)
    current = equity[-1]
    dd = peak - current
    dd_percent = (dd / peak * 100) if peak > 0 else 0
    
    return {
        "drawdown": dd,
        "drawdown_percent": dd_percent,
        "peak_value": peak,
        "current_value": current
    }


def check_drawdown_limits(
    equity_curve: Union[List[float], np.ndarray],
    max_drawdown_limit: float = 20.0,
    daily_loss_limit: Optional[float] = 3.0
) -> Dict[str, Any]:
    """
    Check if drawdown limits are breached
    
    Args:
        equity_curve: Portfolio equity curve
        max_drawdown_limit: Maximum allowed drawdown percentage
        daily_loss_limit: Maximum daily loss percentage (optional)
    
    Returns:
        Dict with breach status and details
    """
    breaches = []
    
    # Check max drawdown
    current_dd = current_drawdown(equity_curve)
    
    if current_dd["drawdown_percent"] >= max_drawdown_limit:
        breaches.append({
            "type": "MAX_DRAWDOWN",
            "current": current_dd["drawdown_percent"],
            "limit": max_drawdown_limit,
            "severity": "CRITICAL"
        })
    elif current_dd["drawdown_percent"] >= max_drawdown_limit * 0.8:
        breaches.append({
            "type": "DRAWDOWN_WARNING",
            "current": current_dd["drawdown_percent"],
            "limit": max_drawdown_limit,
            "severity": "WARNING"
        })
    
    # Check daily loss
    if daily_loss_limit and len(equity_curve) >= 2:
        daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2] * 100
        
        if daily_return <= -daily_loss_limit:
            breaches.append({
                "type": "DAILY_LOSS",
                "current": daily_return,
                "limit": -daily_loss_limit,
                "severity": "CRITICAL"
            })
    
    return {
        "breached": len([b for b in breaches if b["severity"] == "CRITICAL"]) > 0,
        "warning": len([b for b in breaches if b["severity"] == "WARNING"]) > 0,
        "breaches": breaches,
        "current_drawdown": current_dd
    }


def drawdown_duration(
    equity_curve: Union[List[float], np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Calculate drawdown periods and their durations
    
    Args:
        equity_curve: Portfolio equity curve
    
    Returns:
        List of drawdown periods with start, end, duration, depth
    """
    dd = calculate_drawdown(equity_curve)
    dd_percent = dd["drawdown_percent"]
    
    periods = []
    in_drawdown = False
    start_idx = 0
    
    for i in range(len(dd_percent)):
        if dd_percent[i] > 0 and not in_drawdown:
            in_drawdown = True
            start_idx = i - 1 if i > 0 else 0
        elif dd_percent[i] == 0 and in_drawdown:
            in_drawdown = False
            max_depth = np.max(dd_percent[start_idx:i])
            periods.append({
                "start_index": start_idx,
                "end_index": i,
                "duration": i - start_idx,
                "max_depth_percent": max_depth
            })
    
    # Handle ongoing drawdown
    if in_drawdown:
        max_depth = np.max(dd_percent[start_idx:])
        periods.append({
            "start_index": start_idx,
            "end_index": len(dd_percent) - 1,
            "duration": len(dd_percent) - 1 - start_idx,
            "max_depth_percent": max_depth,
            "ongoing": True
        })
    
    return periods


def recovery_factor(
    equity_curve: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate recovery factor (total return / max drawdown)
    
    Higher is better, indicates ability to recover from drawdowns.
    
    Args:
        equity_curve: Portfolio equity curve
    
    Returns:
        Recovery factor
    """
    equity = np.array(equity_curve)
    
    if len(equity) < 2:
        return 0
    
    total_return = (equity[-1] - equity[0]) / equity[0] * 100
    mdd = max_drawdown(equity)
    
    if mdd["max_drawdown_percent"] == 0:
        return float('inf') if total_return > 0 else 0
    
    return total_return / mdd["max_drawdown_percent"]
