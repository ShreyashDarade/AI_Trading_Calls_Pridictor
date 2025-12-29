"""
Market Ingestor Service
Ingests live market data from Groww APIs (Feed + REST fallback)
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from fastapi import FastAPI, BackgroundTasks

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Market Ingestor Service",
    description="Ingests live market data from Groww APIs",
    version="1.0.0"
)


class MarketIngestor:
    """
    Market Data Ingestor
    
    Responsibilities:
    1. Subscribe to Groww streaming feed
    2. Normalize incoming ticks
    3. Publish to Kafka topics
    4. Fallback to REST polling when feed is unavailable
    """
    
    def __init__(self):
        self.subscribed_tokens: Set[str] = set()
        self.is_connected = False
        self.last_tick_time: Dict[str, datetime] = {}
        self.feed_client = None
        self.rest_client = None
    
    async def start(self, exchange_tokens: List[str]):
        """Start ingesting data for given tokens"""
        logger.info(f"Starting ingestor for {len(exchange_tokens)} instruments")
        
        # Try streaming feed first
        try:
            await self._start_feed(exchange_tokens)
        except Exception as e:
            logger.warning(f"Feed connection failed, falling back to REST: {e}")
            await self._start_rest_polling(exchange_tokens)
    
    async def _start_feed(self, tokens: List[str]):
        """Start streaming feed connection"""
        from libs.groww_client.src.feed import FeedClient
        from libs.common.src.config import get_settings
        
        settings = get_settings()
        
        self.feed_client = FeedClient(
            api_key=settings.groww.api_key,
            access_token=settings.groww.access_token
        )
        
        if await self.feed_client.connect():
            self.is_connected = True
            await self.feed_client.subscribe(
                exchange_tokens=tokens,
                callback=self._on_tick
            )
            await self.feed_client.start()
    
    async def _start_rest_polling(self, tokens: List[str]):
        """Fallback to REST API polling"""
        from libs.groww_client.src.live_data import LiveDataClient
        from libs.common.src.config import get_settings
        
        settings = get_settings()
        
        self.rest_client = LiveDataClient(
            api_key=settings.groww.api_key,
            access_token=settings.groww.access_token
        )
        
        # Poll every 2 seconds
        while True:
            try:
                instruments = [{"exchange": "NSE", "trading_symbol": t} for t in tokens[:50]]
                quotes = await self.rest_client.get_ltp_batch(instruments)
                
                for key, data in quotes.items():
                    await self._on_tick({
                        "exchange_token": key,
                        "ltp": data.get("ltp"),
                        "timestamp": datetime.utcnow()
                    })
                    
            except Exception as e:
                logger.error(f"REST polling error: {e}")
            
            await asyncio.sleep(2)
    
    async def _on_tick(self, tick: Dict):
        """Process incoming tick"""
        # Normalize tick
        normalized = {
            "exchange_token": tick.get("exchange_token"),
            "timestamp": tick.get("timestamp", datetime.utcnow()),
            "ltp": tick.get("ltp"),
            "bid_price": tick.get("bid_price"),
            "ask_price": tick.get("ask_price"),
            "volume": tick.get("volume"),
        }
        
        # Publish to Kafka
        await self._publish_tick(normalized)
        
        # Update last tick time
        self.last_tick_time[normalized["exchange_token"]] = normalized["timestamp"]
    
    async def _publish_tick(self, tick: Dict):
        """Publish tick to Kafka"""
        # TODO: Implement Kafka producer
        logger.debug(f"Tick: {tick['exchange_token']} = {tick['ltp']}")
    
    async def stop(self):
        """Stop the ingestor"""
        if self.feed_client:
            await self.feed_client.disconnect()
        if self.rest_client:
            await self.rest_client.close()
        self.is_connected = False


# Global ingestor instance
ingestor = MarketIngestor()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "connected": ingestor.is_connected,
        "subscribed_count": len(ingestor.subscribed_tokens)
    }


@app.post("/start")
async def start_ingestor(
    tokens: List[str],
    background_tasks: BackgroundTasks
):
    """Start ingesting for given exchange tokens"""
    background_tasks.add_task(ingestor.start, tokens)
    return {"message": f"Started ingesting {len(tokens)} instruments"}


@app.post("/stop")
async def stop_ingestor():
    """Stop the ingestor"""
    await ingestor.stop()
    return {"message": "Ingestor stopped"}


@app.get("/status")
async def get_status():
    """Get ingestor status"""
    return {
        "connected": ingestor.is_connected,
        "subscribed_tokens": list(ingestor.subscribed_tokens),
        "last_ticks": {
            k: v.isoformat() for k, v in ingestor.last_tick_time.items()
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
