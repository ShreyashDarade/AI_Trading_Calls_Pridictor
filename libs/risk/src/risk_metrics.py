"""
Risk Metrics
Sharpe, Sortino, Calmar, VaR, Expected Shortfall
"""
from typing import List, Union, Optional
import numpy as np


def sharpe_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.06,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio
    
    Measures risk-adjusted return.
    Higher is better, > 1 is good, > 2 is excellent.
    
    Args:
        returns: List of periodic returns (e.g., daily)
        risk_free_rate: Annual risk-free rate (default 6% for India)
        periods_per_year: Number of periods per year (252 for daily)
    
    Returns:
        Annualized Sharpe ratio
    """
    returns = np.array(returns)
    
    if len(returns) < 2:
        return 0
    
    # Convert annual risk-free to periodic
    rf_periodic = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_returns = returns - rf_periodic
    
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)
    
    if std_returns == 0:
        return 0
    
    # Annualize
    sharpe = (mean_excess / std_returns) * np.sqrt(periods_per_year)
    
    return sharpe


def sortino_ratio(
    returns: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.06,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio
    
    Like Sharpe, but only penalizes downside volatility.
    Higher is better.
    
    Args:
        returns: List of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year
    
    Returns:
        Annualized Sortino ratio
    """
    returns = np.array(returns)
    
    if len(returns) < 2:
        return 0
    
    # Convert annual risk-free to periodic
    rf_periodic = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_returns = returns - rf_periodic
    mean_excess = np.mean(excess_returns)
    
    # Downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) == 0:
        return float('inf') if mean_excess > 0 else 0
    
    downside_std = np.std(negative_returns, ddof=1)
    
    if downside_std == 0:
        return float('inf') if mean_excess > 0 else 0
    
    # Annualize
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    
    return sortino


def calmar_ratio(
    returns: Union[List[float], np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio
    
    Annual return divided by maximum drawdown.
    Higher is better.
    
    Args:
        returns: List of periodic returns
        periods_per_year: Periods per year
    
    Returns:
        Calmar ratio
    """
    returns = np.array(returns)
    
    if len(returns) < 2:
        return 0
    
    # Calculate equity curve
    equity = np.cumprod(1 + returns)
    
    # Annual return
    total_return = equity[-1] - 1
    num_years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_dd = np.max(drawdowns)
    
    if max_dd == 0:
        return float('inf') if annual_return > 0 else 0
    
    return annual_return / max_dd


def value_at_risk(
    returns: Union[List[float], np.ndarray],
    confidence_level: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Maximum expected loss at a given confidence level.
    
    Args:
        returns: List of periodic returns
        confidence_level: Confidence level (default 95%)
        method: "historical" or "parametric"
    
    Returns:
        VaR as a positive number (loss)
    """
    returns = np.array(returns)
    
    if len(returns) < 10:
        return 0
    
    if method == "historical":
        # Historical VaR: percentile of returns
        var = -np.percentile(returns, (1 - confidence_level) * 100)
    else:
        # Parametric VaR: assume normal distribution
        from scipy import stats
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(mean + z_score * std)
    
    return max(0, var)


def expected_shortfall(
    returns: Union[List[float], np.ndarray],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (CVaR)
    
    Average loss beyond VaR. More conservative than VaR.
    
    Args:
        returns: List of periodic returns
        confidence_level: Confidence level
    
    Returns:
        Expected Shortfall as positive number
    """
    returns = np.array(returns)
    
    if len(returns) < 10:
        return 0
    
    var = value_at_risk(returns, confidence_level, "historical")
    
    # Average of returns worse than VaR
    tail_returns = returns[returns <= -var]
    
    if len(tail_returns) == 0:
        return var
    
    return -np.mean(tail_returns)


def win_rate(trades: List[float]) -> float:
    """
    Calculate win rate
    
    Args:
        trades: List of trade returns/PnL
    
    Returns:
        Win rate as decimal (0 to 1)
    """
    trades = np.array(trades)
    
    if len(trades) == 0:
        return 0
    
    winners = np.sum(trades > 0)
    return winners / len(trades)


def profit_factor(trades: List[float]) -> float:
    """
    Calculate profit factor
    
    Gross profit / Gross loss. > 1 is profitable.
    
    Args:
        trades: List of trade returns/PnL
    
    Returns:
        Profit factor
    """
    trades = np.array(trades)
    
    gross_profit = np.sum(trades[trades > 0])
    gross_loss = abs(np.sum(trades[trades < 0]))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0
    
    return gross_profit / gross_loss


def average_win_loss_ratio(trades: List[float]) -> float:
    """
    Calculate average win / average loss
    
    Args:
        trades: List of trade returns/PnL
    
    Returns:
        Win/Loss ratio
    """
    trades = np.array(trades)
    
    winners = trades[trades > 0]
    losers = trades[trades < 0]
    
    if len(winners) == 0:
        avg_win = 0
    else:
        avg_win = np.mean(winners)
    
    if len(losers) == 0:
        avg_loss = 0
    else:
        avg_loss = abs(np.mean(losers))
    
    if avg_loss == 0:
        return float('inf') if avg_win > 0 else 0
    
    return avg_win / avg_loss


def expectancy(trades: List[float]) -> float:
    """
    Calculate trade expectancy
    
    Average expected return per trade.
    Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
    
    Args:
        trades: List of trade returns/PnL
    
    Returns:
        Expectancy per trade
    """
    trades = np.array(trades)
    
    if len(trades) == 0:
        return 0
    
    winners = trades[trades > 0]
    losers = trades[trades < 0]
    
    win_pct = len(winners) / len(trades)
    loss_pct = len(losers) / len(trades)
    
    avg_win = np.mean(winners) if len(winners) > 0 else 0
    avg_loss = abs(np.mean(losers)) if len(losers) > 0 else 0
    
    return (win_pct * avg_win) - (loss_pct * avg_loss)
