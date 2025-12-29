"""
Momentum Indicators
RSI, MACD, Stochastic, Williams %R, CCI, ROC
"""
import numpy as np
from typing import Tuple, Optional, List, Union
import pandas as pd


def rsi(
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Relative Strength Index (RSI)
    
    Measures speed and change of price movements.
    Values range from 0 to 100.
    - RSI > 70: Overbought
    - RSI < 30: Oversold
    
    Args:
        close: Closing prices
        period: RSI period (default 14)
    
    Returns:
        RSI values as numpy array
    """
    close = np.array(close)
    delta = np.diff(close)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Exponential moving average of gains/losses
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    
    # First average
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Subsequent averages (exponential smoothing)
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    # Calculate RS and RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_values = 100 - (100 / (1 + rs))
    
    # Set initial values to NaN
    rsi_values[:period] = np.nan
    
    return rsi_values


def macd(
    close: Union[List[float], np.ndarray, pd.Series],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence Divergence (MACD)
    
    Trend-following momentum indicator.
    
    Args:
        close: Closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    close = np.array(close)
    
    # Calculate EMAs
    fast_ema = _ema(close, fast_period)
    slow_ema = _ema(close, slow_period)
    
    # MACD line
    macd_line = fast_ema - slow_ema
    
    # Signal line (EMA of MACD)
    signal_line = _ema(macd_line, signal_period)
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def stochastic(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    k_period: int = 14,
    d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator
    
    Compares closing price to price range over a period.
    - %K > 80: Overbought
    - %K < 20: Oversold
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        k_period: %K period (default 14)
        d_period: %D smoothing period (default 3)
    
    Returns:
        Tuple of (%K, %D)
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    n = len(close)
    k = np.zeros(n)
    
    for i in range(k_period - 1, n):
        highest_high = np.max(high[i - k_period + 1:i + 1])
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        
        if highest_high != lowest_low:
            k[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
    
    k[:k_period - 1] = np.nan
    
    # %D is SMA of %K
    d = _sma(k, d_period)
    
    return k, d


def williams_r(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Williams %R
    
    Momentum indicator measuring overbought/oversold levels.
    Values range from -100 to 0.
    - %R > -20: Overbought
    - %R < -80: Oversold
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        period: Lookback period (default 14)
    
    Returns:
        Williams %R values
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    n = len(close)
    williams = np.zeros(n)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            williams[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    williams[:period - 1] = np.nan
    
    return williams


def cci(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 20
) -> np.ndarray:
    """
    Commodity Channel Index (CCI)
    
    Measures current price level relative to average.
    - CCI > 100: Potentially overbought
    - CCI < -100: Potentially oversold
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        period: Lookback period (default 20)
    
    Returns:
        CCI values
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    # Typical price
    tp = (high + low + close) / 3
    
    # SMA of typical price
    tp_sma = _sma(tp, period)
    
    # Mean deviation
    n = len(close)
    mean_dev = np.zeros(n)
    
    for i in range(period - 1, n):
        mean_dev[i] = np.mean(np.abs(tp[i - period + 1:i + 1] - tp_sma[i]))
    
    mean_dev[:period - 1] = np.nan
    
    # CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    cci_values = np.where(mean_dev != 0, (tp - tp_sma) / (0.015 * mean_dev), 0)
    cci_values[:period - 1] = np.nan
    
    return cci_values


def roc(
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 12
) -> np.ndarray:
    """
    Rate of Change (ROC)
    
    Percentage change in price over a period.
    
    Args:
        close: Closing prices
        period: Lookback period (default 12)
    
    Returns:
        ROC values as percentage
    """
    close = np.array(close)
    
    roc_values = np.zeros(len(close))
    roc_values[:period] = np.nan
    
    for i in range(period, len(close)):
        if close[i - period] != 0:
            roc_values[i] = ((close[i] - close[i - period]) / close[i - period]) * 100
    
    return roc_values


# ============================================
# HELPER FUNCTIONS
# ============================================

def _ema(
    data: np.ndarray,
    period: int
) -> np.ndarray:
    """Calculate Exponential Moving Average"""
    ema = np.zeros(len(data))
    ema[:period - 1] = np.nan
    
    # First EMA is SMA
    ema[period - 1] = np.mean(data[:period])
    
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def _sma(
    data: np.ndarray,
    period: int
) -> np.ndarray:
    """Calculate Simple Moving Average"""
    sma = np.zeros(len(data))
    sma[:period - 1] = np.nan
    
    for i in range(period - 1, len(data)):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    
    return sma
