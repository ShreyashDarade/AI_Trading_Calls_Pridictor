"""
Groww Live Data Client
REST API client for live market data (Quote, LTP)
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class LiveDataClient:
    """
    Groww Live Data REST API Client
    
    Provides access to:
    - Full quotes with depth (GET /v1/live-data/quote)
    - LTP for multiple instruments (GET /v1/live-data/ltp) - up to 50 per call
    
    Usage:
        client = LiveDataClient(api_key="...", access_token="...")
        
        # Get full quote
        quote = await client.get_quote("NSE", "RELIANCE")
        
        # Get LTP for multiple
        ltps = await client.get_ltp_batch([
            {"exchange": "NSE", "trading_symbol": "RELIANCE"},
            {"exchange": "NSE", "trading_symbol": "TCS"}
        ])
    """
    
    BASE_URL = "https://api.groww.in"
    
    def __init__(
        self,
        api_key: str,
        access_token: str,
        base_url: Optional[str] = None,
        timeout: float = 10.0
    ):
        self.api_key = api_key
        self.access_token = access_token
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Api-Key": api_key,
            "Content-Type": "application/json"
        }
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout
            )
        return self._client
    
    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def get_quote(
        self,
        exchange: str,
        trading_symbol: str
    ) -> Dict[str, Any]:
        """
        Get full quote with market depth for a single instrument
        
        Endpoint: GET /v1/live-data/quote
        
        Returns:
            {
                "exchange": "NSE",
                "tradingSymbol": "RELIANCE",
                "ltp": 2450.50,
                "open": 2430.00,
                "high": 2465.00,
                "low": 2425.00,
                "close": 2428.00,
                "volume": 1234567,
                "bidDepth": [...],
                "askDepth": [...],
                "oi": 0,
                "change": 22.50,
                "changePercent": 0.93,
                ...
            }
        """
        client = await self._get_client()
        
        response = await client.get(
            "/v1/live-data/quote",
            params={
                "exchange": exchange.upper(),
                "tradingSymbol": trading_symbol
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Quote request failed: {response.status_code}")
            raise Exception(f"Failed to get quote: {response.status_code}")
        
        data = response.json()
        return self._normalize_quote(data)
    
    async def get_ltp(
        self,
        exchange: str,
        trading_symbol: str
    ) -> Dict[str, Any]:
        """Get LTP for a single instrument"""
        result = await self.get_ltp_batch([{
            "exchange": exchange,
            "trading_symbol": trading_symbol
        }])
        
        key = f"{exchange.upper()}:{trading_symbol}"
        return result.get(key, {})
    
    async def get_ltp_batch(
        self,
        instruments: List[Dict[str, str]],
        chunk_size: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get LTP for multiple instruments
        
        Endpoint: GET /v1/live-data/ltp
        Maximum: 50 instruments per request
        
        Args:
            instruments: List of {"exchange": "NSE", "trading_symbol": "..."}
            chunk_size: Instruments per request (max 50)
        
        Returns:
            Dict keyed by "EXCHANGE:SYMBOL" with LTP data
        """
        if chunk_size > 50:
            chunk_size = 50
        
        all_results = {}
        
        # Process in chunks
        for i in range(0, len(instruments), chunk_size):
            chunk = instruments[i:i + chunk_size]
            chunk_result = await self._get_ltp_chunk(chunk)
            all_results.update(chunk_result)
        
        return all_results
    
    async def _get_ltp_chunk(
        self,
        instruments: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Get LTP for a single chunk (max 50)"""
        client = await self._get_client()
        
        # Format: "NSE:RELIANCE,NSE:TCS,..."
        instrument_str = ",".join([
            f"{i.get('exchange', 'NSE').upper()}:{i['trading_symbol']}"
            for i in instruments
        ])
        
        response = await client.get(
            "/v1/live-data/ltp",
            params={"instruments": instrument_str}
        )
        
        if response.status_code != 200:
            logger.error(f"LTP request failed: {response.status_code}")
            raise Exception(f"Failed to get LTP: {response.status_code}")
        
        data = response.json()
        return self._normalize_ltp_response(data)
    
    def _normalize_quote(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize quote response to standard format"""
        return {
            "exchange": data.get("exchange", "").upper(),
            "trading_symbol": data.get("tradingSymbol", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "ltp": float(data.get("ltp", 0)),
            "open": float(data.get("open", 0)),
            "high": float(data.get("high", 0)),
            "low": float(data.get("low", 0)),
            "close": float(data.get("close", 0)),
            "volume": int(data.get("volume", 0)),
            "oi": int(data.get("oi", 0)),
            "oi_change": int(data.get("oiChange", 0)),
            "change": float(data.get("change", 0)),
            "change_percent": float(data.get("changePercent", 0)),
            "avg_price": float(data.get("avgPrice", 0)),
            "bid_depth": data.get("bidDepth", []),
            "ask_depth": data.get("askDepth", []),
            "lower_circuit": data.get("lowerCircuit"),
            "upper_circuit": data.get("upperCircuit"),
        }
    
    def _normalize_ltp_response(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Normalize LTP batch response"""
        result = {}
        
        # Handle different response formats
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    result[key] = {
                        "ltp": float(value.get("ltp", 0)),
                        "change": float(value.get("change", 0)),
                        "change_percent": float(value.get("changePercent", 0)),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                elif isinstance(value, (int, float)):
                    result[key] = {
                        "ltp": float(value),
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        return result
    
    async def poll_quotes(
        self,
        instruments: List[Dict[str, str]],
        interval_seconds: float = 1.0,
        callback=None
    ):
        """
        Continuously poll quotes at specified interval
        
        This is a fallback mechanism when streaming is not available.
        
        Args:
            instruments: Instruments to poll
            interval_seconds: Polling interval
            callback: Async function to call with each update
        """
        while True:
            try:
                ltps = await self.get_ltp_batch(instruments)
                
                if callback:
                    await callback(ltps)
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
            
            await asyncio.sleep(interval_seconds)
