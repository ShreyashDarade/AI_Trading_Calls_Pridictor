"""
Portfolio Optimization Service
Modern Portfolio Theory and optimization algorithms
"""
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Portfolio Optimization Service",
    description="MPT-based portfolio optimization",
    version="4.0.0"
)


class Asset(BaseModel):
    symbol: str
    expected_return: float  # Annual %
    volatility: float  # Annual %
    current_weight: float = 0
    min_weight: float = 0
    max_weight: float = 1


class OptimizationRequest(BaseModel):
    assets: List[Asset]
    correlation_matrix: Optional[List[List[float]]] = None
    risk_free_rate: float = 0.07  # 7%
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    optimization_type: str = "sharpe"  # sharpe, min_variance, max_return


class OptimizedPortfolio(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    diversification_ratio: float


class EfficientFrontier(BaseModel):
    portfolios: List[OptimizedPortfolio]
    optimal_sharpe: OptimizedPortfolio
    min_variance: OptimizedPortfolio


class PortfolioOptimizer:
    """Portfolio optimization using Mean-Variance Optimization"""
    
    def __init__(self):
        self.cache = {}
    
    def optimize(self, request: OptimizationRequest) -> OptimizedPortfolio:
        """Optimize portfolio based on objective"""
        n = len(request.assets)
        returns = np.array([a.expected_return / 100 for a in request.assets])
        volatilities = np.array([a.volatility / 100 for a in request.assets])
        
        # Build covariance matrix
        if request.correlation_matrix:
            corr = np.array(request.correlation_matrix)
        else:
            # Assume moderate correlation
            corr = np.eye(n) * 0.5 + 0.5 * np.ones((n, n))
            np.fill_diagonal(corr, 1)
        
        cov = np.outer(volatilities, volatilities) * corr
        
        # Optimization based on type
        if request.optimization_type == "min_variance":
            weights = self._min_variance(cov, request.assets)
        elif request.optimization_type == "max_return":
            weights = self._max_return(returns, request.assets)
        else:  # sharpe
            weights = self._max_sharpe(returns, cov, request.risk_free_rate, request.assets)
        
        # Calculate portfolio metrics
        port_return = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = (port_return - request.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        # Diversification ratio
        weighted_vol = np.dot(weights, volatilities)
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1
        
        return OptimizedPortfolio(
            weights={a.symbol: round(w, 4) for a, w in zip(request.assets, weights)},
            expected_return=round(port_return * 100, 2),
            volatility=round(port_vol * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            diversification_ratio=round(div_ratio, 2)
        )
    
    def _min_variance(self, cov: np.ndarray, assets: List[Asset]) -> np.ndarray:
        """Minimum variance portfolio"""
        n = len(assets)
        
        # Analytical solution for unconstrained case
        ones = np.ones(n)
        inv_cov = np.linalg.pinv(cov)
        weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
        
        # Apply constraints
        weights = self._apply_constraints(weights, assets)
        return weights
    
    def _max_return(self, returns: np.ndarray, assets: List[Asset]) -> np.ndarray:
        """Maximum return portfolio (concentrate in highest return)"""
        n = len(assets)
        weights = np.zeros(n)
        
        # Sort by return and allocate to highest
        sorted_idx = np.argsort(returns)[::-1]
        remaining = 1.0
        
        for idx in sorted_idx:
            max_w = assets[idx].max_weight
            min_w = assets[idx].min_weight
            allocation = min(remaining, max_w)
            weights[idx] = max(allocation, min_w)
            remaining -= weights[idx]
            if remaining <= 0:
                break
        
        return weights / weights.sum() if weights.sum() > 0 else weights
    
    def _max_sharpe(self, returns: np.ndarray, cov: np.ndarray, 
                   rf: float, assets: List[Asset]) -> np.ndarray:
        """Maximum Sharpe ratio portfolio using gradient ascent"""
        n = len(assets)
        weights = np.ones(n) / n  # Start equal weighted
        
        learning_rate = 0.01
        for _ in range(1000):
            port_return = np.dot(weights, returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
            
            if port_vol < 0.0001:
                break
            
            sharpe = (port_return - rf) / port_vol
            
            # Gradient of Sharpe ratio
            grad_return = returns / port_vol
            grad_vol = np.dot(cov, weights) / port_vol
            grad_sharpe = grad_return - sharpe * grad_vol
            
            # Update weights
            weights = weights + learning_rate * grad_sharpe
            weights = self._apply_constraints(weights, assets)
        
        return weights
    
    def _apply_constraints(self, weights: np.ndarray, assets: List[Asset]) -> np.ndarray:
        """Apply min/max weight constraints"""
        for i, asset in enumerate(assets):
            weights[i] = max(asset.min_weight, min(asset.max_weight, weights[i]))
        
        # Normalize to sum to 1
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(len(assets)) / len(assets)
        
        return weights
    
    def efficient_frontier(self, request: OptimizationRequest, 
                          points: int = 20) -> EfficientFrontier:
        """Generate efficient frontier"""
        returns = np.array([a.expected_return for a in request.assets])
        min_ret = returns.min()
        max_ret = returns.max()
        
        portfolios = []
        
        for target in np.linspace(min_ret, max_ret, points):
            # Find minimum variance portfolio for target return
            req_copy = request.copy()
            req_copy.target_return = target
            req_copy.optimization_type = "min_variance"
            
            try:
                portfolio = self.optimize(req_copy)
                portfolios.append(portfolio)
            except:
                continue
        
        # Find optimal portfolios
        if portfolios:
            optimal_sharpe = max(portfolios, key=lambda p: p.sharpe_ratio)
            min_var = min(portfolios, key=lambda p: p.volatility)
        else:
            optimal_sharpe = min_var = self.optimize(request)
        
        return EfficientFrontier(
            portfolios=portfolios,
            optimal_sharpe=optimal_sharpe,
            min_variance=min_var
        )
    
    def rebalance(self, current_weights: Dict[str, float], 
                 target_weights: Dict[str, float],
                 portfolio_value: float,
                 min_trade_value: float = 1000) -> List[dict]:
        """Calculate rebalancing trades"""
        trades = []
        
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            diff = target - current
            trade_value = abs(diff * portfolio_value)
            
            if trade_value >= min_trade_value:
                trades.append({
                    "symbol": symbol,
                    "action": "BUY" if diff > 0 else "SELL",
                    "weight_change": round(diff * 100, 2),
                    "trade_value": round(trade_value, 2)
                })
        
        return sorted(trades, key=lambda x: x["trade_value"], reverse=True)


optimizer = PortfolioOptimizer()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/optimize", response_model=OptimizedPortfolio)
async def optimize(request: OptimizationRequest):
    return optimizer.optimize(request)


@app.post("/efficient-frontier", response_model=EfficientFrontier)
async def efficient_frontier(request: OptimizationRequest, points: int = 20):
    return optimizer.efficient_frontier(request, points)


@app.post("/rebalance")
async def rebalance(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    portfolio_value: float,
    min_trade_value: float = 1000
):
    trades = optimizer.rebalance(current_weights, target_weights, portfolio_value, min_trade_value)
    return {"trades": trades}


@app.post("/risk-parity")
async def risk_parity(assets: List[Asset]):
    """Risk parity allocation"""
    volatilities = np.array([a.volatility for a in assets])
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()
    
    return {
        "weights": {a.symbol: round(w, 4) for a, w in zip(assets, weights)},
        "method": "risk_parity"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8019, reload=True)
