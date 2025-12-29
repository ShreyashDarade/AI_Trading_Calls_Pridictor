"""
Volatility Indicators
ATR, Bollinger Bands, Keltner Channels, Donchian Channels, Historical Volatility
"""
import numpy as np
from typing import Tuple, Union, List
import pandas as pd


def atr(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Average True Range (ATR)
    
    Measures market volatility by decomposing the entire range of an asset.
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        period: ATR period (default 14)
    
    Returns:
        ATR values
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    n = len(close)
    tr = np.zeros(n)
    
    # True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    
    tr[0] = high[0] - low[0]
    
    # ATR (smoothed average of TR)
    atr_values = np.zeros(n)
    atr_values[:period] = np.nan
    
    # First ATR is simple average
    atr_values[period - 1] = np.mean(tr[:period])
    
    # Subsequent ATRs use exponential smoothing
    for i in range(period, n):
        atr_values[i] = (atr_values[i - 1] * (period - 1) + tr[i]) / period
    
    return atr_values


def bollinger_bands(
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands
    
    Volatility bands placed above and below a moving average.
    
    Args:
        close: Closing prices
        period: SMA period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    close = np.array(close)
    n = len(close)
    
    # Middle band (SMA)
    middle = np.zeros(n)
    middle[:period - 1] = np.nan
    
    for i in range(period - 1, n):
        middle[i] = np.mean(close[i - period + 1:i + 1])
    
    # Standard deviation
    std = np.zeros(n)
    std[:period - 1] = np.nan
    
    for i in range(period - 1, n):
        std[i] = np.std(close[i - period + 1:i + 1], ddof=0)
    
    # Upper and lower bands
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return upper, middle, lower


def keltner_channels(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    ema_period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keltner Channels
    
    Volatility channels based on EMA and ATR.
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        ema_period: EMA period (default 20)
        atr_period: ATR period (default 10)
        atr_multiplier: ATR multiplier (default 2.0)
    
    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """
    close = np.array(close)
    
    # Middle line (EMA of close)
    from libs.indicators.src.trend import ema
    middle = ema(close, ema_period)
    
    # ATR
    atr_values = atr(high, low, close, atr_period)
    
    # Channels
    upper = middle + atr_multiplier * atr_values
    lower = middle - atr_multiplier * atr_values
    
    return upper, middle, lower


def donchian_channels(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Donchian Channels
    
    Price channels based on highest high and lowest low.
    
    Args:
        high: High prices
        low: Low prices
        period: Lookback period (default 20)
    
    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """
    high = np.array(high)
    low = np.array(low)
    n = len(high)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:period - 1] = np.nan
    lower[:period - 1] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def historical_volatility(
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> np.ndarray:
    """
    Historical Volatility (Standard Deviation of Returns)
    
    Measures volatility based on historical price changes.
    
    Args:
        close: Closing prices
        period: Lookback period (default 20)
        annualize: Whether to annualize volatility (default True)
        trading_days: Trading days per year (default 252)
    
    Returns:
        Historical volatility values (as decimal, not percentage)
    """
    close = np.array(close)
    n = len(close)
    
    # Log returns
    log_returns = np.zeros(n)
    for i in range(1, n):
        if close[i - 1] > 0:
            log_returns[i] = np.log(close[i] / close[i - 1])
    
    # Rolling standard deviation
    vol = np.zeros(n)
    vol[:period] = np.nan
    
    for i in range(period, n):
        vol[i] = np.std(log_returns[i - period + 1:i + 1], ddof=1)
    
    # Annualize
    if annualize:
        vol = vol * np.sqrt(trading_days)
    
    return vol


def volatility_regime(
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    low_threshold: float = 0.15,
    high_threshold: float = 0.30
) -> np.ndarray:
    """
    Determine volatility regime
    
    Args:
        close: Closing prices
        period: Volatility calculation period
        low_threshold: Below this = LOW volatility (annualized)
        high_threshold: Above this = HIGH volatility (annualized)
    
    Returns:
        Array of regime labels: "LOW", "MEDIUM", "HIGH"
    """
    vol = historical_volatility(close, period, annualize=True)
    
    n = len(close)
    regime = np.empty(n, dtype=object)
    regime[:] = "MEDIUM"
    
    for i in range(period, n):
        if np.isnan(vol[i]):
            continue
        
        if vol[i] < low_threshold:
            regime[i] = "LOW"
        elif vol[i] > high_threshold:
            regime[i] = "HIGH"
        else:
            regime[i] = "MEDIUM"
    
    return regime


def bollinger_bandwidth(
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0
) -> np.ndarray:
    """
    Bollinger Bandwidth
    
    Measures the width of Bollinger Bands as a percentage of the middle band.
    Useful for detecting volatility squeeze.
    
    Args:
        close: Closing prices
        period: SMA period
        std_dev: Standard deviation multiplier
    
    Returns:
        Bandwidth values as percentage
    """
    upper, middle, lower = bollinger_bands(close, period, std_dev)
    
    bandwidth = np.where(middle != 0, (upper - lower) / middle * 100, 0)
    
    return bandwidth


def bollinger_percent_b(
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0
) -> np.ndarray:
    """
    Bollinger %B
    
    Shows where price is relative to the bands.
    - %B > 1: Price above upper band
    - %B < 0: Price below lower band
    - %B = 0.5: Price at middle band
    
    Args:
        close: Closing prices
        period: SMA period
        std_dev: Standard deviation multiplier
    
    Returns:
        %B values
    """
    close = np.array(close)
    upper, middle, lower = bollinger_bands(close, period, std_dev)
    
    bandwidth = upper - lower
    percent_b = np.where(bandwidth != 0, (close - lower) / bandwidth, 0.5)
    
    return percent_b
