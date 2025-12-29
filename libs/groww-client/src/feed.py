"""
Groww Feed Client
WebSocket/Streaming feed for real-time market data
"""
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Set
from datetime import datetime
import websockets

logger = logging.getLogger(__name__)


class FeedClient:
    """
    Groww Streaming Feed Client
    
    Provides real-time market data via WebSocket connection.
    Supports subscribing to up to 1000 instruments per connection.
    
    Requires exchange_token from the instrument master CSV for subscriptions.
    
    Usage:
        client = FeedClient(api_key="...", access_token="...")
        
        # Define callback
        async def on_tick(tick):
            print(f"{tick['symbol']}: {tick['ltp']}")
        
        # Subscribe and start
        await client.connect()
        await client.subscribe(
            exchange_tokens=["1234", "5678"],
            callback=on_tick
        )
        await client.start()
    """
    
    # Default WebSocket URL (update based on Groww documentation)
    WS_URL = "wss://feed.groww.in/socket"
    
    # Maximum instruments per subscription
    MAX_INSTRUMENTS = 1000
    
    def __init__(
        self,
        api_key: str,
        access_token: str,
        ws_url: Optional[str] = None,
        reconnect_delay: float = 5.0,
        ping_interval: float = 30.0
    ):
        """
        Initialize Feed Client
        
        Args:
            api_key: Groww API key
            access_token: OAuth access token
            ws_url: WebSocket URL (optional)
            reconnect_delay: Seconds to wait before reconnecting
            ping_interval: Seconds between ping messages
        """
        self.api_key = api_key
        self.access_token = access_token
        self.ws_url = ws_url or self.WS_URL
        self.reconnect_delay = reconnect_delay
        self.ping_interval = ping_interval
        
        self._websocket = None
        self._running = False
        self._subscribed_tokens: Set[str] = set()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._default_callback: Optional[Callable] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection
        
        Returns:
            True if connection successful
        """
        try:
            # Build auth headers
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "X-Api-Key": self.api_key
            }
            
            self._websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=None  # We handle pinging ourselves
            )
            
            logger.info("Connected to Groww feed")
            
            # Start ping task
            self._ping_task = asyncio.create_task(self._ping_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Close WebSocket connection"""
        self._running = False
        
        if self._ping_task:
            self._ping_task.cancel()
        
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        logger.info("Disconnected from Groww feed")
    
    async def subscribe(
        self,
        exchange_tokens: List[str],
        exchange: str = "NSE",
        mode: str = "full",
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Subscribe to instrument feeds
        
        Args:
            exchange_tokens: List of exchange tokens from instrument master
            exchange: Exchange code (NSE, BSE)
            mode: Subscription mode (full, ltp, quote)
            callback: Async function to call on tick updates
        
        Returns:
            True if subscription successful
        """
        if len(exchange_tokens) > self.MAX_INSTRUMENTS:
            logger.warning(f"Limiting to {self.MAX_INSTRUMENTS} instruments")
            exchange_tokens = exchange_tokens[:self.MAX_INSTRUMENTS]
        
        if not self._websocket:
            logger.error("Not connected. Call connect() first.")
            return False
        
        # Build subscription message
        subscribe_msg = {
            "action": "subscribe",
            "exchange": exchange.upper(),
            "tokens": exchange_tokens,
            "mode": mode
        }
        
        try:
            await self._websocket.send(json.dumps(subscribe_msg))
            
            # Track subscribed tokens
            for token in exchange_tokens:
                self._subscribed_tokens.add(token)
                if callback:
                    if token not in self._callbacks:
                        self._callbacks[token] = []
                    self._callbacks[token].append(callback)
            
            if callback and not self._default_callback:
                self._default_callback = callback
            
            logger.info(f"Subscribed to {len(exchange_tokens)} instruments")
            return True
            
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return False
    
    async def unsubscribe(
        self,
        exchange_tokens: List[str],
        exchange: str = "NSE"
    ) -> bool:
        """
        Unsubscribe from instrument feeds
        
        Args:
            exchange_tokens: List of exchange tokens to unsubscribe
            exchange: Exchange code
        
        Returns:
            True if unsubscription successful
        """
        if not self._websocket:
            return False
        
        unsubscribe_msg = {
            "action": "unsubscribe",
            "exchange": exchange.upper(),
            "tokens": exchange_tokens
        }
        
        try:
            await self._websocket.send(json.dumps(unsubscribe_msg))
            
            for token in exchange_tokens:
                self._subscribed_tokens.discard(token)
                self._callbacks.pop(token, None)
            
            logger.info(f"Unsubscribed from {len(exchange_tokens)} instruments")
            return True
            
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            return False
    
    async def start(self):
        """
        Start receiving and processing messages
        
        This runs indefinitely, processing incoming ticks.
        """
        self._running = True
        
        while self._running:
            try:
                if not self._websocket:
                    await self._reconnect()
                    continue
                
                message = await self._websocket.recv()
                await self._process_message(message)
                
            except websockets.ConnectionClosed:
                logger.warning("Connection closed. Reconnecting...")
                self._websocket = None
                await asyncio.sleep(self.reconnect_delay)
                
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            msg_type = data.get("type", "tick")
            
            if msg_type == "tick":
                tick = self._parse_tick(data)
                await self._dispatch_tick(tick)
                
            elif msg_type == "error":
                logger.error(f"Feed error: {data.get('message')}")
                
            elif msg_type == "connected":
                logger.info("Feed confirmed connected")
                
            elif msg_type == "subscribed":
                logger.debug(f"Subscription confirmed: {data}")
                
        except json.JSONDecodeError:
            # Binary message - parse as binary tick
            tick = self._parse_binary_tick(message)
            if tick:
                await self._dispatch_tick(tick)
    
    def _parse_tick(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON tick message"""
        return {
            "exchange_token": data.get("token"),
            "exchange": data.get("exchange", "NSE"),
            "trading_symbol": data.get("tradingSymbol", data.get("symbol")),
            "timestamp": datetime.utcnow(),
            "ltp": float(data.get("ltp", 0)),
            "open": float(data.get("open", 0)) if data.get("open") else None,
            "high": float(data.get("high", 0)) if data.get("high") else None,
            "low": float(data.get("low", 0)) if data.get("low") else None,
            "close": float(data.get("close", 0)) if data.get("close") else None,
            "volume": int(data.get("volume", 0)),
            "change": float(data.get("change", 0)),
            "change_percent": float(data.get("changePercent", 0)),
            "bid_price": float(data.get("bidPrice", 0)) if data.get("bidPrice") else None,
            "bid_qty": int(data.get("bidQty", 0)) if data.get("bidQty") else None,
            "ask_price": float(data.get("askPrice", 0)) if data.get("askPrice") else None,
            "ask_qty": int(data.get("askQty", 0)) if data.get("askQty") else None,
            "oi": int(data.get("oi", 0)) if data.get("oi") else None,
        }
    
    def _parse_binary_tick(self, message: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse binary tick message
        
        Binary format varies by provider - implement based on Groww specs.
        """
        # Placeholder - implement based on actual Groww binary format
        return None
    
    async def _dispatch_tick(self, tick: Dict[str, Any]):
        """Dispatch tick to registered callbacks"""
        token = tick.get("exchange_token")
        
        # Call token-specific callbacks
        if token in self._callbacks:
            for callback in self._callbacks[token]:
                try:
                    await callback(tick)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        # Call default callback
        if self._default_callback:
            try:
                await self._default_callback(tick)
            except Exception as e:
                logger.error(f"Default callback error: {e}")
    
    async def _reconnect(self):
        """Attempt to reconnect and resubscribe"""
        logger.info("Attempting to reconnect...")
        
        if await self.connect():
            # Resubscribe to all tokens
            if self._subscribed_tokens:
                await self.subscribe(
                    exchange_tokens=list(self._subscribed_tokens),
                    callback=self._default_callback
                )
        else:
            await asyncio.sleep(self.reconnect_delay)
    
    async def _ping_loop(self):
        """Send periodic pings to keep connection alive"""
        while self._running and self._websocket:
            try:
                await asyncio.sleep(self.ping_interval)
                if self._websocket:
                    await self._websocket.ping()
            except Exception as e:
                logger.warning(f"Ping failed: {e}")
                break
