"""
Groww Historical Data Client
REST API client for historical candle data (backtesting)
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import httpx

logger = logging.getLogger(__name__)


class HistoricalDataClient:
    """
    Groww Historical Data REST API Client
    
    Provides access to historical OHLCV candles for backtesting.
    Uses the official backtesting endpoint: GET /v1/historical/candles
    
    Note: Avoid deprecated /historical/candle/range endpoint.
    
    Usage:
        client = HistoricalDataClient(api_key="...", access_token="...")
        
        candles = await client.get_candles(
            exchange="NSE",
            trading_symbol="RELIANCE",
            interval="1d",
            from_date="2024-01-01",
            to_date="2024-12-01"
        )
    """
    
    BASE_URL = "https://api.groww.in"
    
    # Supported intervals
    VALID_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "1d", "1w", "1M"]
    
    def __init__(
        self,
        api_key: str,
        access_token: str,
        base_url: Optional[str] = None,
        timeout: float = 30.0
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
    
    async def get_candles(
        self,
        exchange: str,
        trading_symbol: str,
        interval: str,
        from_date: str,
        to_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get historical OHLCV candles
        
        Endpoint: GET /v1/historical/candles
        
        Args:
            exchange: Exchange code (NSE, BSE)
            trading_symbol: Trading symbol
            interval: Candle interval (1m, 5m, 15m, 30m, 1h, 1d, 1w, 1M)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        
        Returns:
            List of candle dictionaries:
            [
                {
                    "timestamp": "2024-01-01T09:15:00",
                    "open": 2430.00,
                    "high": 2445.00,
                    "low": 2425.00,
                    "close": 2440.00,
                    "volume": 123456
                },
                ...
            ]
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of: {self.VALID_INTERVALS}")
        
        client = await self._get_client()
        
        response = await client.get(
            "/v1/historical/candles",
            params={
                "exchange": exchange.upper(),
                "tradingSymbol": trading_symbol,
                "interval": interval,
                "fromDate": from_date,
                "toDate": to_date
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Historical data request failed: {response.status_code}")
            raise Exception(f"Failed to get historical data: {response.status_code}")
        
        data = response.json()
        return self._normalize_candles(data, exchange, trading_symbol, interval)
    
    async def get_candles_for_days(
        self,
        exchange: str,
        trading_symbol: str,
        interval: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """
        Get candles for the last N days
        
        Args:
            exchange: Exchange code
            trading_symbol: Trading symbol
            interval: Candle interval
            days: Number of days to fetch
        
        Returns:
            List of candles
        """
        to_date = date.today()
        from_date = to_date - timedelta(days=days)
        
        return await self.get_candles(
            exchange=exchange,
            trading_symbol=trading_symbol,
            interval=interval,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d")
        )
    
    async def get_daily_candles(
        self,
        exchange: str,
        trading_symbol: str,
        from_date: str,
        to_date: str
    ) -> List[Dict[str, Any]]:
        """Convenience method for daily candles"""
        return await self.get_candles(
            exchange=exchange,
            trading_symbol=trading_symbol,
            interval="1d",
            from_date=from_date,
            to_date=to_date
        )
    
    async def get_intraday_candles(
        self,
        exchange: str,
        trading_symbol: str,
        interval: str = "5m",
        trading_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get intraday candles for a single trading day
        
        Args:
            exchange: Exchange code
            trading_symbol: Trading symbol
            interval: Intraday interval (1m, 5m, 15m, 30m)
            trading_date: Date to fetch (defaults to today)
        
        Returns:
            List of intraday candles
        """
        if trading_date is None:
            trading_date = date.today().strftime("%Y-%m-%d")
        
        return await self.get_candles(
            exchange=exchange,
            trading_symbol=trading_symbol,
            interval=interval,
            from_date=trading_date,
            to_date=trading_date
        )
    
    def _normalize_candles(
        self,
        data: Any,
        exchange: str,
        trading_symbol: str,
        interval: str
    ) -> List[Dict[str, Any]]:
        """Normalize candle response to standard format"""
        candles = []
        
        # Handle different response formats
        raw_candles = data.get("candles", data) if isinstance(data, dict) else data
        
        if not isinstance(raw_candles, list):
            return candles
        
        for raw in raw_candles:
            try:
                if isinstance(raw, dict):
                    candle = {
                        "exchange": exchange.upper(),
                        "trading_symbol": trading_symbol,
                        "interval": interval,
                        "timestamp": raw.get("timestamp") or raw.get("date"),
                        "open": float(raw.get("open", 0)),
                        "high": float(raw.get("high", 0)),
                        "low": float(raw.get("low", 0)),
                        "close": float(raw.get("close", 0)),
                        "volume": int(raw.get("volume", 0)),
                        "oi": int(raw.get("oi", 0)) if raw.get("oi") else None
                    }
                elif isinstance(raw, list) and len(raw) >= 5:
                    # Array format: [timestamp, open, high, low, close, volume]
                    candle = {
                        "exchange": exchange.upper(),
                        "trading_symbol": trading_symbol,
                        "interval": interval,
                        "timestamp": raw[0],
                        "open": float(raw[1]),
                        "high": float(raw[2]),
                        "low": float(raw[3]),
                        "close": float(raw[4]),
                        "volume": int(raw[5]) if len(raw) > 5 else 0,
                        "oi": int(raw[6]) if len(raw) > 6 else None
                    }
                else:
                    continue
                
                candles.append(candle)
                
            except (ValueError, TypeError, IndexError) as e:
                logger.warning(f"Failed to parse candle: {e}")
                continue
        
        # Sort by timestamp
        candles.sort(key=lambda x: x["timestamp"])
        
        return candles
    
    async def download_for_backtest(
        self,
        exchange: str,
        trading_symbol: str,
        interval: str,
        from_date: str,
        to_date: str,
        output_path: str
    ):
        """
        Download and save historical data for backtesting
        
        Args:
            exchange: Exchange code
            trading_symbol: Trading symbol
            interval: Candle interval
            from_date: Start date
            to_date: End date
            output_path: Path to save CSV/JSON file
        """
        import json
        
        candles = await self.get_candles(
            exchange=exchange,
            trading_symbol=trading_symbol,
            interval=interval,
            from_date=from_date,
            to_date=to_date
        )
        
        with open(output_path, "w") as f:
            json.dump(candles, f, indent=2)
        
        logger.info(f"Saved {len(candles)} candles to {output_path}")
