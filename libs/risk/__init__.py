"""
Risk Management Library
Position sizing, exposure limits, drawdown rules
"""
from libs.risk.src.position_sizing import (
    fixed_fractional,
    kelly_criterion,
    volatility_adjusted_sizing,
    max_position_size,
)
from libs.risk.src.exposure import (
    calculate_exposure,
    check_exposure_limits,
    get_sector_exposure,
)
from libs.risk.src.drawdown import (
    calculate_drawdown,
    max_drawdown,
    check_drawdown_limits,
)
from libs.risk.src.risk_metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    value_at_risk,
    expected_shortfall,
)

__all__ = [
    # Position Sizing
    "fixed_fractional",
    "kelly_criterion",
    "volatility_adjusted_sizing",
    "max_position_size",
    # Exposure
    "calculate_exposure",
    "check_exposure_limits",
    "get_sector_exposure",
    # Drawdown
    "calculate_drawdown",
    "max_drawdown",
    "check_drawdown_limits",
    # Risk Metrics
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "value_at_risk",
    "expected_shortfall",
]
