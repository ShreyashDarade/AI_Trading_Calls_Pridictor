"""
Instrument Service
Manages instrument master data from Groww API
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Instrument Service",
    description="Instrument master data management",
    version="1.0.0"
)


# ============================================
# MODELS
# ============================================

class InstrumentResponse(BaseModel):
    id: str
    exchange: str
    segment: str
    instrument_type: str
    trading_symbol: str
    groww_symbol: str
    exchange_token: str
    name: str
    isin: Optional[str]
    lot_size: int
    tick_size: float


class InstrumentSearchResult(BaseModel):
    total: int
    results: List[InstrumentResponse]


# ============================================
# IN-MEMORY STORAGE (Replace with PostgreSQL)
# ============================================

INSTRUMENTS: Dict[str, Dict[str, Any]] = {}
LAST_REFRESH: Optional[datetime] = None


# ============================================
# SERVICE LOGIC
# ============================================

async def refresh_instruments(force: bool = False):
    """
    Download and refresh instrument master from Groww
    """
    global INSTRUMENTS, LAST_REFRESH
    
    # Check if refresh is needed
    if not force and LAST_REFRESH:
        if datetime.utcnow() - LAST_REFRESH < timedelta(hours=12):
            logger.info("Instruments are fresh, skipping refresh")
            return
    
    logger.info("Refreshing instrument master from Groww...")
    
    try:
        # Import Groww client
        from libs.groww_client.src.instruments import InstrumentDownloader
        
        downloader = InstrumentDownloader()
        instruments = await downloader.download_and_parse()
        
        # Store in memory (in production, store in PostgreSQL)
        for inst in instruments:
            key = f"{inst['exchange']}:{inst['trading_symbol']}"
            INSTRUMENTS[key] = inst
        
        LAST_REFRESH = datetime.utcnow()
        logger.info(f"Loaded {len(instruments)} instruments")
        
    except Exception as e:
        logger.error(f"Failed to refresh instruments: {e}")
        raise


def search_instruments(
    query: str,
    exchange: Optional[str] = None,
    segment: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Search instruments by name or symbol"""
    query = query.upper()
    results = []
    
    for key, inst in INSTRUMENTS.items():
        # Filter by exchange
        if exchange and inst["exchange"] != exchange.upper():
            continue
        
        # Filter by segment
        if segment and inst["segment"] != segment.upper():
            continue
        
        # Match query
        if (
            query in inst["trading_symbol"].upper() or
            query in inst.get("name", "").upper() or
            query in inst.get("groww_symbol", "").upper()
        ):
            results.append(inst)
            if len(results) >= limit:
                break
    
    return results


def get_instrument_by_token(exchange: str, exchange_token: str) -> Optional[Dict[str, Any]]:
    """Get instrument by exchange token"""
    for inst in INSTRUMENTS.values():
        if inst["exchange"] == exchange.upper() and inst["exchange_token"] == exchange_token:
            return inst
    return None


def get_instrument(exchange: str, trading_symbol: str) -> Optional[Dict[str, Any]]:
    """Get instrument by exchange and trading symbol"""
    key = f"{exchange.upper()}:{trading_symbol.upper()}"
    return INSTRUMENTS.get(key)


# ============================================
# API ROUTES
# ============================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "instruments_loaded": len(INSTRUMENTS),
        "last_refresh": LAST_REFRESH.isoformat() if LAST_REFRESH else None
    }


@app.post("/refresh")
async def trigger_refresh(background_tasks: BackgroundTasks, force: bool = False):
    """Trigger instrument master refresh"""
    background_tasks.add_task(refresh_instruments, force)
    return {"message": "Refresh triggered", "force": force}


@app.get("/search")
async def api_search_instruments(
    q: str = Query(..., min_length=1),
    exchange: Optional[str] = None,
    segment: Optional[str] = None,
    limit: int = Query(default=20, le=100)
):
    """Search instruments"""
    results = search_instruments(q, exchange, segment, limit)
    return {"total": len(results), "results": results}


@app.get("/instrument/{exchange}/{trading_symbol}")
async def api_get_instrument(exchange: str, trading_symbol: str):
    """Get instrument by exchange and trading symbol"""
    inst = get_instrument(exchange, trading_symbol)
    if not inst:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument {exchange}:{trading_symbol} not found"
        )
    return inst


@app.get("/token/{exchange}/{exchange_token}")
async def api_get_by_token(exchange: str, exchange_token: str):
    """Get instrument by exchange token"""
    inst = get_instrument_by_token(exchange, exchange_token)
    if not inst:
        raise HTTPException(
            status_code=404,
            detail=f"Instrument with token {exchange_token} not found"
        )
    return inst


@app.get("/nifty50")
async def get_nifty50():
    """Get NIFTY 50 constituent symbols"""
    from libs.groww_client.src.instruments import InstrumentDownloader
    
    downloader = InstrumentDownloader()
    symbols = downloader.get_nifty50_symbols()
    
    instruments = []
    for symbol in symbols:
        inst = get_instrument("NSE", symbol)
        if inst:
            instruments.append(inst)
    
    return {"symbols": symbols, "instruments": instruments}


@app.get("/stats")
async def get_stats():
    """Get instrument statistics"""
    stats = {
        "total": len(INSTRUMENTS),
        "by_exchange": {},
        "by_segment": {},
        "by_type": {},
        "last_refresh": LAST_REFRESH.isoformat() if LAST_REFRESH else None
    }
    
    for inst in INSTRUMENTS.values():
        exchange = inst["exchange"]
        segment = inst["segment"]
        inst_type = inst["instrument_type"]
        
        stats["by_exchange"][exchange] = stats["by_exchange"].get(exchange, 0) + 1
        stats["by_segment"][segment] = stats["by_segment"].get(segment, 0) + 1
        stats["by_type"][inst_type] = stats["by_type"].get(inst_type, 0) + 1
    
    return stats


# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Instrument Service...")
    # Load instruments on startup
    try:
        await refresh_instruments()
    except Exception as e:
        logger.warning(f"Could not load instruments on startup: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
