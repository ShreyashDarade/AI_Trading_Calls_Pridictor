"""
Exposure Management
Portfolio exposure calculations and limits
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ExposureLimits:
    """Configurable exposure limits"""
    max_single_position_percent: float = 5.0  # Max 5% in single position
    max_sector_percent: float = 25.0  # Max 25% in single sector
    max_long_exposure_percent: float = 100.0  # Max long exposure
    max_short_exposure_percent: float = 50.0  # Max short exposure
    max_gross_exposure_percent: float = 150.0  # Max gross (long + short)
    max_net_exposure_percent: float = 100.0  # Max net (long - short)
    max_open_positions: int = 20  # Max number of positions


def calculate_exposure(
    positions: List[Dict[str, Any]],
    capital: float
) -> Dict[str, float]:
    """
    Calculate current portfolio exposure
    
    Args:
        positions: List of position dicts with 'value', 'side' keys
        capital: Total portfolio capital
    
    Returns:
        Dict with various exposure metrics
    """
    long_value = 0.0
    short_value = 0.0
    
    for pos in positions:
        value = abs(pos.get("value", 0))
        side = pos.get("side", "BUY").upper()
        
        if side == "BUY":
            long_value += value
        else:
            short_value += value
    
    gross_exposure = long_value + short_value
    net_exposure = long_value - short_value
    
    return {
        "long_value": long_value,
        "short_value": short_value,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "long_percent": (long_value / capital * 100) if capital > 0 else 0,
        "short_percent": (short_value / capital * 100) if capital > 0 else 0,
        "gross_percent": (gross_exposure / capital * 100) if capital > 0 else 0,
        "net_percent": (net_exposure / capital * 100) if capital > 0 else 0,
        "num_positions": len(positions),
    }


def check_exposure_limits(
    exposure: Dict[str, float],
    new_position_value: float,
    new_position_side: str,
    limits: Optional[ExposureLimits] = None
) -> Dict[str, Any]:
    """
    Check if a new position would violate exposure limits
    
    Args:
        exposure: Current exposure from calculate_exposure()
        new_position_value: Value of new position
        new_position_side: BUY or SELL
        limits: ExposureLimits config
    
    Returns:
        Dict with 'allowed' bool and 'violations' list
    """
    if limits is None:
        limits = ExposureLimits()
    
    violations = []
    
    # Simulate new exposure
    new_long = exposure["long_percent"]
    new_short = exposure["short_percent"]
    
    if new_position_side.upper() == "BUY":
        new_long += new_position_value
    else:
        new_short += new_position_value
    
    new_gross = new_long + new_short
    new_net = new_long - new_short
    
    # Check limits
    if new_long > limits.max_long_exposure_percent:
        violations.append({
            "type": "LONG_EXPOSURE",
            "current": new_long,
            "limit": limits.max_long_exposure_percent
        })
    
    if new_short > limits.max_short_exposure_percent:
        violations.append({
            "type": "SHORT_EXPOSURE",
            "current": new_short,
            "limit": limits.max_short_exposure_percent
        })
    
    if new_gross > limits.max_gross_exposure_percent:
        violations.append({
            "type": "GROSS_EXPOSURE",
            "current": new_gross,
            "limit": limits.max_gross_exposure_percent
        })
    
    if abs(new_net) > limits.max_net_exposure_percent:
        violations.append({
            "type": "NET_EXPOSURE",
            "current": new_net,
            "limit": limits.max_net_exposure_percent
        })
    
    if exposure["num_positions"] + 1 > limits.max_open_positions:
        violations.append({
            "type": "MAX_POSITIONS",
            "current": exposure["num_positions"] + 1,
            "limit": limits.max_open_positions
        })
    
    return {
        "allowed": len(violations) == 0,
        "violations": violations,
        "simulated_exposure": {
            "long_percent": new_long,
            "short_percent": new_short,
            "gross_percent": new_gross,
            "net_percent": new_net
        }
    }


def get_sector_exposure(
    positions: List[Dict[str, Any]],
    capital: float
) -> Dict[str, Dict[str, float]]:
    """
    Calculate exposure by sector
    
    Args:
        positions: List of positions with 'sector' and 'value' keys
        capital: Total portfolio capital
    
    Returns:
        Dict mapping sector to exposure details
    """
    sector_exposure = {}
    
    for pos in positions:
        sector = pos.get("sector", "UNKNOWN")
        value = abs(pos.get("value", 0))
        
        if sector not in sector_exposure:
            sector_exposure[sector] = {
                "value": 0,
                "percent": 0,
                "positions": 0
            }
        
        sector_exposure[sector]["value"] += value
        sector_exposure[sector]["positions"] += 1
    
    # Calculate percentages
    for sector in sector_exposure:
        sector_exposure[sector]["percent"] = (
            sector_exposure[sector]["value"] / capital * 100
            if capital > 0 else 0
        )
    
    return sector_exposure


def check_single_position_limit(
    position_value: float,
    capital: float,
    max_percent: float = 5.0
) -> bool:
    """
    Check if a single position is within limits
    
    Args:
        position_value: Value of the position
        capital: Total capital
        max_percent: Maximum allowed percentage
    
    Returns:
        True if within limits
    """
    if capital <= 0:
        return False
    
    position_percent = abs(position_value) / capital * 100
    return position_percent <= max_percent


def calculate_available_capital(
    capital: float,
    positions: List[Dict[str, Any]],
    max_exposure_percent: float = 100.0
) -> float:
    """
    Calculate remaining capital available for new positions
    
    Args:
        capital: Total capital
        positions: Current positions
        max_exposure_percent: Maximum exposure allowed
    
    Returns:
        Available capital for new positions
    """
    exposure = calculate_exposure(positions, capital)
    current_exposure = exposure["gross_exposure"]
    max_exposure = capital * (max_exposure_percent / 100)
    
    return max(0, max_exposure - current_exposure)
