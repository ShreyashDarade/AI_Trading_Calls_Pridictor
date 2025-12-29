"""
Groww API Client - Main Client Class
Unified interface for all Groww API interactions
"""
import logging
from typing import Optional, Dict, Any, List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class GrowwClient:
    """
    Main Groww API Client
    
    Provides unified access to all Groww API endpoints.
    Handles authentication, rate limiting, and error handling.
    
    Usage:
        client = GrowwClient(
            api_key="your_api_key",
            access_token="your_access_token"
        )
        
        # Get live quote
        quote = await client.get_quote(exchange="NSE", trading_symbol="RELIANCE")
    """
    
    BASE_URL = "https://api.groww.in"
    API_VERSION = "v1"
    
    def __init__(
        self,
        api_key: str,
        access_token: str,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize Groww API Client
        
        Args:
            api_key: Groww API key
            access_token: OAuth access token
            api_secret: API secret (optional, for some endpoints)
            base_url: Override base URL (optional)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.access_token = access_token
        self.api_secret = api_secret
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        
        # Build default headers
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Api-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # HTTP client (async)
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout
            )
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Groww API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON body data
        
        Returns:
            API response as dictionary
        
        Raises:
            GrowwAPIError: On API errors
        """
        client = await self._get_client()
        
        try:
            response = await client.request(
                method=method,
                url=f"/{self.API_VERSION}/{endpoint}",
                params=params,
                json=json_data
            )
            
            # Check for rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", 60)
                logger.warning(f"Rate limited. Retry after {retry_after}s")
                raise RateLimitError(retry_after=int(retry_after))
            
            # Check for auth errors
            if response.status_code == 401:
                logger.error("Authentication failed")
                raise AuthenticationError("Invalid or expired access token")
            
            # Check for other errors
            if response.status_code >= 400:
                error_data = response.json() if response.content else {}
                logger.error(f"API error: {response.status_code} - {error_data}")
                raise GrowwAPIError(
                    message=error_data.get("message", "Unknown error"),
                    status_code=response.status_code,
                    error_data=error_data
                )
            
            return response.json()
            
        except httpx.TimeoutException:
            logger.error(f"Request timeout for {endpoint}")
            raise GrowwAPIError(message="Request timeout", status_code=408)
        except httpx.RequestError as e:
            logger.error(f"Request error for {endpoint}: {e}")
            raise GrowwAPIError(message=str(e), status_code=500)
    
    # ========================================
    # LIVE DATA ENDPOINTS
    # ========================================
    
    async def get_quote(
        self,
        exchange: str,
        trading_symbol: str
    ) -> Dict[str, Any]:
        """
        Get full quote with depth for a single instrument
        
        Endpoint: GET /v1/live-data/quote
        
        Args:
            exchange: Exchange code (NSE, BSE)
            trading_symbol: Trading symbol
        
        Returns:
            Quote data including depth, OHLC, volumes
        """
        return await self._request(
            "GET",
            "live-data/quote",
            params={
                "exchange": exchange,
                "tradingSymbol": trading_symbol
            }
        )
    
    async def get_ltp(
        self,
        instruments: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Get LTP for multiple instruments (up to 50)
        
        Endpoint: GET /v1/live-data/ltp
        
        Args:
            instruments: List of {"exchange": "NSE", "tradingSymbol": "RELIANCE"}
        
        Returns:
            LTP data for all requested instruments
        """
        if len(instruments) > 50:
            raise ValueError("Maximum 50 instruments per LTP request")
        
        # Format instruments for API
        instrument_str = ",".join([
            f"{i['exchange']}:{i['tradingSymbol']}"
            for i in instruments
        ])
        
        return await self._request(
            "GET",
            "live-data/ltp",
            params={"instruments": instrument_str}
        )
    
    # ========================================
    # HISTORICAL DATA ENDPOINTS
    # ========================================
    
    async def get_historical_candles(
        self,
        exchange: str,
        trading_symbol: str,
        interval: str,
        from_date: str,
        to_date: str
    ) -> Dict[str, Any]:
        """
        Get historical OHLCV candles for backtesting
        
        Endpoint: GET /v1/historical/candles
        
        Args:
            exchange: Exchange code
            trading_symbol: Trading symbol
            interval: Candle interval (1m, 5m, 15m, 30m, 1h, 1d)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        
        Returns:
            Historical candle data
        """
        return await self._request(
            "GET",
            "historical/candles",
            params={
                "exchange": exchange,
                "tradingSymbol": trading_symbol,
                "interval": interval,
                "fromDate": from_date,
                "toDate": to_date
            }
        )
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def get_instruments_csv_url(self) -> str:
        """Get URL for instruments CSV download"""
        return "https://growwapi-assets.groww.in/instruments/instrument.csv"


class GrowwAPIError(Exception):
    """Exception for Groww API errors"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_data: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_data = error_data or {}
        super().__init__(self.message)


class RateLimitError(GrowwAPIError):
    """Rate limit exceeded"""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after}s",
            status_code=429
        )
        self.retry_after = retry_after


class AuthenticationError(GrowwAPIError):
    """Authentication failed"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message=message, status_code=401)
