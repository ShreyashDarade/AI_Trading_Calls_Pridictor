"""
Trend Indicators
SMA, EMA, WMA, DEMA, TEMA, SuperTrend, ADX
"""
import numpy as np
from typing import Tuple, Union, List
import pandas as pd


def sma(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int
) -> np.ndarray:
    """
    Simple Moving Average (SMA)
    
    Args:
        data: Price data
        period: SMA period
    
    Returns:
        SMA values
    """
    data = np.array(data)
    sma_values = np.zeros(len(data))
    sma_values[:period - 1] = np.nan
    
    for i in range(period - 1, len(data)):
        sma_values[i] = np.mean(data[i - period + 1:i + 1])
    
    return sma_values


def ema(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int
) -> np.ndarray:
    """
    Exponential Moving Average (EMA)
    
    Gives more weight to recent prices.
    
    Args:
        data: Price data
        period: EMA period
    
    Returns:
        EMA values
    """
    data = np.array(data)
    ema_values = np.zeros(len(data))
    ema_values[:period - 1] = np.nan
    
    # First EMA is SMA
    ema_values[period - 1] = np.mean(data[:period])
    
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(data)):
        ema_values[i] = (data[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]
    
    return ema_values


def wma(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int
) -> np.ndarray:
    """
    Weighted Moving Average (WMA)
    
    Linearly weighted average giving more weight to recent prices.
    
    Args:
        data: Price data
        period: WMA period
    
    Returns:
        WMA values
    """
    data = np.array(data)
    wma_values = np.zeros(len(data))
    wma_values[:period - 1] = np.nan
    
    weights = np.arange(1, period + 1)
    weight_sum = np.sum(weights)
    
    for i in range(period - 1, len(data)):
        wma_values[i] = np.sum(data[i - period + 1:i + 1] * weights) / weight_sum
    
    return wma_values


def dema(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int
) -> np.ndarray:
    """
    Double Exponential Moving Average (DEMA)
    
    Reduces lag compared to regular EMA.
    DEMA = 2 * EMA - EMA(EMA)
    
    Args:
        data: Price data
        period: DEMA period
    
    Returns:
        DEMA values
    """
    data = np.array(data)
    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    
    return 2 * ema1 - ema2


def tema(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int
) -> np.ndarray:
    """
    Triple Exponential Moving Average (TEMA)
    
    Further reduces lag compared to DEMA.
    TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    
    Args:
        data: Price data
        period: TEMA period
    
    Returns:
        TEMA values
    """
    data = np.array(data)
    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    
    return 3 * ema1 - 3 * ema2 + ema3


def supertrend(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 10,
    multiplier: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SuperTrend Indicator
    
    Trend-following indicator based on ATR.
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        period: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)
    
    Returns:
        Tuple of (supertrend_values, trend_direction)
        trend_direction: 1 = uptrend, -1 = downtrend
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    n = len(close)
    
    # Calculate ATR
    from libs.indicators.src.volatility import atr as calc_atr
    atr_values = calc_atr(high, low, close, period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_basic = hl2 + multiplier * atr_values
    lower_basic = hl2 - multiplier * atr_values
    
    # Final bands
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend_val = np.zeros(n)
    trend = np.zeros(n)
    
    upper_band[period - 1] = upper_basic[period - 1]
    lower_band[period - 1] = lower_basic[period - 1]
    
    for i in range(period, n):
        # Upper band
        if upper_basic[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i - 1]
        
        # Lower band
        if lower_basic[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i - 1]
        
        # SuperTrend
        if i == period:
            if close[i] <= upper_band[i]:
                trend[i] = 1
            else:
                trend[i] = -1
        else:
            if trend[i - 1] == -1 and close[i] <= upper_band[i]:
                trend[i] = 1
            elif trend[i - 1] == 1 and close[i] >= lower_band[i]:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]
        
        if trend[i] == 1:
            supertrend_val[i] = lower_band[i]
        else:
            supertrend_val[i] = upper_band[i]
    
    supertrend_val[:period] = np.nan
    trend[:period] = np.nan
    
    return supertrend_val, trend


def adx(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average Directional Index (ADX)
    
    Measures trend strength (not direction).
    - ADX > 25: Strong trend
    - ADX < 20: Weak trend / ranging
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        period: ADX period (default 14)
    
    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    n = len(close)
    
    # True Range
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
        
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    # Smoothed averages
    atr_smooth = np.zeros(n)
    dm_plus_smooth = np.zeros(n)
    dm_minus_smooth = np.zeros(n)
    
    atr_smooth[period] = np.sum(tr[1:period + 1])
    dm_plus_smooth[period] = np.sum(dm_plus[1:period + 1])
    dm_minus_smooth[period] = np.sum(dm_minus[1:period + 1])
    
    for i in range(period + 1, n):
        atr_smooth[i] = atr_smooth[i - 1] - (atr_smooth[i - 1] / period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i - 1] - (dm_plus_smooth[i - 1] / period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i - 1] - (dm_minus_smooth[i - 1] / period) + dm_minus[i]
    
    # +DI and -DI
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if atr_smooth[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr_smooth[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr_smooth[i]
    
    # DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum != 0:
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX (smoothed DX)
    adx_values = np.zeros(n)
    adx_values[2 * period - 1] = np.mean(dx[period:2 * period])
    
    for i in range(2 * period, n):
        adx_values[i] = (adx_values[i - 1] * (period - 1) + dx[i]) / period
    
    # Set NaN for warmup period
    adx_values[:2 * period - 1] = np.nan
    di_plus[:period] = np.nan
    di_minus[:period] = np.nan
    
    return adx_values, di_plus, di_minus


def trend_regime(
    close: Union[List[float], np.ndarray, pd.Series],
    short_period: int = 20,
    long_period: int = 50
) -> np.ndarray:
    """
    Determine trend regime based on moving average crossover
    
    Returns:
        Array of regime labels: "UPTREND", "DOWNTREND", "SIDEWAYS"
    """
    close = np.array(close)
    
    short_ma = sma(close, short_period)
    long_ma = sma(close, long_period)
    
    n = len(close)
    regime = np.empty(n, dtype=object)
    regime[:] = "SIDEWAYS"
    
    for i in range(long_period - 1, n):
        if np.isnan(short_ma[i]) or np.isnan(long_ma[i]):
            continue
        
        diff_pct = (short_ma[i] - long_ma[i]) / long_ma[i] * 100
        
        if diff_pct > 1:
            regime[i] = "UPTREND"
        elif diff_pct < -1:
            regime[i] = "DOWNTREND"
        else:
            regime[i] = "SIDEWAYS"
    
    return regime
