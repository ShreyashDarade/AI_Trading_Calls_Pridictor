"""
Volume Indicators
OBV, VWAP, Volume SMA, Volume Spike, MFI
"""
import numpy as np
from typing import Union, List
import pandas as pd


def obv(
    close: Union[List[float], np.ndarray, pd.Series],
    volume: Union[List[int], np.ndarray, pd.Series]
) -> np.ndarray:
    """
    On-Balance Volume (OBV)
    
    Cumulative indicator that adds/subtracts volume based on price direction.
    
    Args:
        close: Closing prices
        volume: Volume data
    
    Returns:
        OBV values
    """
    close = np.array(close)
    volume = np.array(volume)
    
    n = len(close)
    obv_values = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv_values[i] = obv_values[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv_values[i] = obv_values[i - 1] - volume[i]
        else:
            obv_values[i] = obv_values[i - 1]
    
    return obv_values


def vwap(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    volume: Union[List[int], np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Volume Weighted Average Price (VWAP)
    
    Average price weighted by volume. Commonly used intraday.
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Volume data
    
    Returns:
        VWAP values
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    volume = np.array(volume)
    
    # Typical price
    typical_price = (high + low + close) / 3
    
    # Cumulative sums
    cumulative_tp_volume = np.cumsum(typical_price * volume)
    cumulative_volume = np.cumsum(volume)
    
    # VWAP
    vwap_values = np.where(
        cumulative_volume != 0,
        cumulative_tp_volume / cumulative_volume,
        0
    )
    
    return vwap_values


def volume_sma(
    volume: Union[List[int], np.ndarray, pd.Series],
    period: int = 20
) -> np.ndarray:
    """
    Volume Simple Moving Average
    
    Args:
        volume: Volume data
        period: SMA period
    
    Returns:
        Volume SMA values
    """
    volume = np.array(volume, dtype=float)
    n = len(volume)
    
    vol_sma = np.zeros(n)
    vol_sma[:period - 1] = np.nan
    
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma


def volume_spike(
    volume: Union[List[int], np.ndarray, pd.Series],
    period: int = 20,
    threshold: float = 2.0
) -> np.ndarray:
    """
    Volume Spike Detection
    
    Identifies when volume is significantly above average.
    
    Args:
        volume: Volume data
        period: Lookback period for average
        threshold: Multiplier for spike detection
    
    Returns:
        Volume ratio (current volume / average volume)
    """
    volume = np.array(volume, dtype=float)
    vol_sma = volume_sma(volume, period)
    
    spike_ratio = np.where(vol_sma != 0, volume / vol_sma, 0)
    
    return spike_ratio


def mfi(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    volume: Union[List[int], np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Money Flow Index (MFI)
    
    Volume-weighted RSI. Ranges from 0 to 100.
    - MFI > 80: Overbought
    - MFI < 20: Oversold
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Volume data
        period: MFI period (default 14)
    
    Returns:
        MFI values
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    volume = np.array(volume, dtype=float)
    
    n = len(close)
    
    # Typical price
    typical_price = (high + low + close) / 3
    
    # Raw money flow
    raw_money_flow = typical_price * volume
    
    # Positive and negative money flow
    pos_mf = np.zeros(n)
    neg_mf = np.zeros(n)
    
    for i in range(1, n):
        if typical_price[i] > typical_price[i - 1]:
            pos_mf[i] = raw_money_flow[i]
        elif typical_price[i] < typical_price[i - 1]:
            neg_mf[i] = raw_money_flow[i]
    
    # Sum over period
    mfi_values = np.zeros(n)
    mfi_values[:period] = np.nan
    
    for i in range(period, n):
        pos_sum = np.sum(pos_mf[i - period + 1:i + 1])
        neg_sum = np.sum(neg_mf[i - period + 1:i + 1])
        
        if neg_sum != 0:
            money_ratio = pos_sum / neg_sum
            mfi_values[i] = 100 - (100 / (1 + money_ratio))
        else:
            mfi_values[i] = 100
    
    return mfi_values


def accumulation_distribution(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    volume: Union[List[int], np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Accumulation/Distribution Line
    
    Measures cumulative money flow.
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Volume data
    
    Returns:
        A/D line values
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    volume = np.array(volume, dtype=float)
    
    n = len(close)
    ad = np.zeros(n)
    
    for i in range(n):
        hl_range = high[i] - low[i]
        if hl_range != 0:
            clv = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
            ad[i] = clv * volume[i]
        if i > 0:
            ad[i] += ad[i - 1]
    
    return ad


def chaikin_money_flow(
    high: Union[List[float], np.ndarray, pd.Series],
    low: Union[List[float], np.ndarray, pd.Series],
    close: Union[List[float], np.ndarray, pd.Series],
    volume: Union[List[int], np.ndarray, pd.Series],
    period: int = 21
) -> np.ndarray:
    """
    Chaikin Money Flow (CMF)
    
    Measures buying and selling pressure over a period.
    Ranges from -1 to +1.
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        volume: Volume data
        period: CMF period (default 21)
    
    Returns:
        CMF values
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    volume = np.array(volume, dtype=float)
    
    n = len(close)
    
    # Money flow multiplier
    mf_mult = np.zeros(n)
    for i in range(n):
        hl_range = high[i] - low[i]
        if hl_range != 0:
            mf_mult[i] = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range
    
    # Money flow volume
    mf_volume = mf_mult * volume
    
    # CMF
    cmf = np.zeros(n)
    cmf[:period - 1] = np.nan
    
    for i in range(period - 1, n):
        vol_sum = np.sum(volume[i - period + 1:i + 1])
        if vol_sum != 0:
            cmf[i] = np.sum(mf_volume[i - period + 1:i + 1]) / vol_sum
    
    return cmf
