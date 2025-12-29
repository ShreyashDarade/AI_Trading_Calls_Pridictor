"""
Options Trading Service
Handles options chain, Greeks, and options strategies
"""
import logging
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from uuid import uuid4
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Options Trading Service",
    description="Options chain, Greeks calculation, and strategy analysis",
    version="2.0.0"
)


class OptionType(str, Enum):
    CALL = "CE"
    PUT = "PE"


class Option(BaseModel):
    symbol: str
    underlying: str
    expiry: date
    strike: float
    option_type: OptionType
    ltp: float = 0
    bid: float = 0
    ask: float = 0
    volume: int = 0
    oi: int = 0
    iv: float = 0
    delta: float = 0
    gamma: float = 0
    theta: float = 0
    vega: float = 0


class OptionsChain(BaseModel):
    underlying: str
    spot_price: float
    expiry: date
    options: List[Option]
    atm_strike: float
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    pcr: float = 0
    max_pain: float = 0


class OptionStrategy(BaseModel):
    name: str
    legs: List[dict]
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakeven: List[float] = []
    net_premium: float = 0


class BlackScholes:
    """Black-Scholes option pricing and Greeks"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        return BlackScholes.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    @staticmethod
    def delta(S, K, T, r, sigma, option_type: OptionType):
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == OptionType.CALL:
            return norm.cdf(d1)
        return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S, K, T, r, sigma, option_type: OptionType):
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r*T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
        return (term1 + term2) / 365
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1) / 100
    
    @staticmethod
    def implied_volatility(price, S, K, T, r, option_type: OptionType, precision=0.0001):
        sigma = 0.3
        for _ in range(100):
            if option_type == OptionType.CALL:
                calc_price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                calc_price = BlackScholes.put_price(S, K, T, r, sigma)
            
            vega = BlackScholes.vega(S, K, T, r, sigma) * 100
            if abs(vega) < 0.0001:
                break
            
            diff = price - calc_price
            if abs(diff) < precision:
                break
            
            sigma += diff / vega
            sigma = max(0.01, min(5.0, sigma))
        
        return sigma


class OptionsService:
    def __init__(self):
        self.chains_cache: Dict[str, OptionsChain] = {}
        self.risk_free_rate = 0.07  # 7% RBI rate
    
    def generate_chain(self, underlying: str, spot: float, expiry: date) -> OptionsChain:
        raise HTTPException(
            status_code=501,
            detail="Options chain generation is disabled (requires real market chain data source)"
        )
    
    def _calculate_max_pain(self, options: List[Option], strikes: List[float]) -> float:
        """Calculate max pain strike"""
        min_pain = float('inf')
        max_pain_strike = strikes[len(strikes)//2]
        
        for test_strike in strikes:
            pain = 0
            for opt in options:
                if opt.option_type == OptionType.CALL:
                    if test_strike > opt.strike:
                        pain += (test_strike - opt.strike) * opt.oi
                else:
                    if test_strike < opt.strike:
                        pain += (opt.strike - test_strike) * opt.oi
            
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = test_strike
        
        return max_pain_strike
    
    def build_strategy(self, name: str, spot: float, legs: List[dict]) -> OptionStrategy:
        """Build and analyze an options strategy"""
        net_premium = 0
        
        for leg in legs:
            premium = leg.get("premium", 0)
            qty = leg.get("quantity", 1)
            if leg.get("action") == "BUY":
                net_premium -= premium * qty
            else:
                net_premium += premium * qty
        
        return OptionStrategy(
            name=name,
            legs=legs,
            net_premium=round(net_premium, 2)
        )


service = OptionsService()


@app.get("/health")
async def health():
    return {"status": "healthy", "cached_chains": len(service.chains_cache)}


@app.get("/chain/{underlying}")
async def get_chain(
    underlying: str,
    spot: float = Query(...),
    expiry: Optional[date] = None
):
    """Get options chain for an underlying"""
    if not expiry:
        # Next Thursday
        today = date.today()
        days_until_thursday = (3 - today.weekday()) % 7
        if days_until_thursday == 0:
            days_until_thursday = 7
        expiry = today + timedelta(days=days_until_thursday)
    
    chain = service.generate_chain(underlying.upper(), spot, expiry)
    return chain


@app.get("/greeks")
async def calculate_greeks(
    spot: float,
    strike: float,
    expiry_days: int,
    iv: float,
    option_type: str = "CE"
):
    """Calculate option Greeks"""
    T = expiry_days / 365
    sigma = iv / 100
    opt_type = OptionType.CALL if option_type.upper() == "CE" else OptionType.PUT
    
    if opt_type == OptionType.CALL:
        price = BlackScholes.call_price(spot, strike, T, 0.07, sigma)
    else:
        price = BlackScholes.put_price(spot, strike, T, 0.07, sigma)
    
    return {
        "price": round(price, 2),
        "delta": round(BlackScholes.delta(spot, strike, T, 0.07, sigma, opt_type), 4),
        "gamma": round(BlackScholes.gamma(spot, strike, T, 0.07, sigma), 6),
        "theta": round(BlackScholes.theta(spot, strike, T, 0.07, sigma, opt_type), 2),
        "vega": round(BlackScholes.vega(spot, strike, T, 0.07, sigma), 2)
    }


@app.post("/strategy")
async def analyze_strategy(name: str, spot: float, legs: List[dict]):
    """Analyze an options strategy"""
    return service.build_strategy(name, spot, legs)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015, reload=True)
