"""
Audit Service
Immutable logging of all trading decisions and system events
"""
import asyncio
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from uuid import uuid4
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import json
import os

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audit Service",
    description="Immutable audit trail for all trading decisions",
    version="1.0.0"
)


# ============================================
# MODELS
# ============================================

class AuditEventType(str, Enum):
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_APPROVED = "SIGNAL_APPROVED"
    SIGNAL_REJECTED = "SIGNAL_REJECTED"
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_MODIFIED = "ORDER_MODIFIED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_EXECUTED = "ORDER_EXECUTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    RISK_CHECK_PASSED = "RISK_CHECK_PASSED"
    RISK_CHECK_FAILED = "RISK_CHECK_FAILED"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    ERROR = "ERROR"


class AuditEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    service_name: str
    
    # Entity references
    signal_id: Optional[str] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    instrument_id: Optional[str] = None
    symbol: Optional[str] = None
    
    # Event data
    action: Optional[str] = None
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    # Snapshot for reproducibility
    market_snapshot: Optional[Dict[str, Any]] = None
    feature_snapshot: Optional[Dict[str, Any]] = None
    
    # Hash for integrity verification
    prev_hash: Optional[str] = None
    hash: Optional[str] = None


class CreateAuditEventRequest(BaseModel):
    event_type: AuditEventType
    service_name: str
    user_id: Optional[str] = None
    signal_id: Optional[str] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    instrument_id: Optional[str] = None
    symbol: Optional[str] = None
    action: Optional[str] = None
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    market_snapshot: Optional[Dict[str, Any]] = None
    feature_snapshot: Optional[Dict[str, Any]] = None


# ============================================
# AUDIT STORE
# ============================================

class AuditStore:
    """Immutable audit event store"""
    
    def __init__(self):
        self.events: List[AuditEvent] = []
        self.last_hash: Optional[str] = None
        self._clickhouse_client = None
    
    async def connect_clickhouse(self, host: str = "localhost", port: int = 9000):
        """Connect to ClickHouse for persistent storage"""
        try:
            from clickhouse_driver import Client
            self._clickhouse_client = Client(host=host, port=port)
            logger.info("Connected to ClickHouse for audit storage")
            return True
        except Exception as e:
            logger.warning(f"ClickHouse connection failed: {e}")
            return False
    
    def _compute_hash(self, event: AuditEvent) -> str:
        """Compute hash for event integrity"""
        import hashlib
        
        data = f"{event.id}:{event.event_type}:{event.timestamp.isoformat()}:{event.service_name}"
        if event.details:
            data += f":{json.dumps(event.details, sort_keys=True)}"
        if self.last_hash:
            data += f":{self.last_hash}"
        
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    async def log_event(self, request: CreateAuditEventRequest) -> AuditEvent:
        """Log an audit event"""
        event = AuditEvent(
            event_type=request.event_type,
            service_name=request.service_name,
            user_id=request.user_id,
            signal_id=request.signal_id,
            order_id=request.order_id,
            position_id=request.position_id,
            instrument_id=request.instrument_id,
            symbol=request.symbol,
            action=request.action,
            reason=request.reason,
            details=request.details,
            market_snapshot=request.market_snapshot,
            feature_snapshot=request.feature_snapshot,
            prev_hash=self.last_hash
        )
        
        # Compute hash
        event.hash = self._compute_hash(event)
        self.last_hash = event.hash
        
        # Store in memory
        self.events.append(event)
        
        # Persist to ClickHouse
        if self._clickhouse_client:
            await self._persist_to_clickhouse(event)
        
        logger.debug(f"Audit event logged: {event.event_type} - {event.id}")
        return event
    
    async def _persist_to_clickhouse(self, event: AuditEvent):
        """Persist event to ClickHouse"""
        try:
            self._clickhouse_client.execute(
                """
                INSERT INTO market_data.signal_audit 
                (signal_id, instrument_id, symbol, timeframe, action, confidence,
                 entry_price, stop_loss, target_1, target_2, reason_codes,
                 feature_vector, model_version, snapshot_id, generated_at)
                VALUES
                """,
                [(
                    event.signal_id or event.id,
                    event.instrument_id or "",
                    event.symbol or "",
                    event.details.get("timeframe", "15m") if event.details else "15m",
                    event.action or "",
                    event.details.get("confidence", 0) if event.details else 0,
                    event.details.get("entry_price") if event.details else None,
                    event.details.get("stop_loss") if event.details else None,
                    event.details.get("target_1") if event.details else None,
                    event.details.get("target_2") if event.details else None,
                    event.details.get("reason_codes", []) if event.details else [],
                    json.dumps(event.feature_snapshot) if event.feature_snapshot else "",
                    event.details.get("model_version", "v1") if event.details else "v1",
                    event.id,
                    event.timestamp
                )]
            )
        except Exception as e:
            logger.error(f"Failed to persist audit event: {e}")
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        symbol: Optional[str] = None,
        signal_id: Optional[str] = None,
        order_id: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events"""
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if symbol:
            events = [e for e in events if e.symbol and e.symbol.upper() == symbol.upper()]
        if signal_id:
            events = [e for e in events if e.signal_id == signal_id]
        if order_id:
            events = [e for e in events if e.order_id == order_id]
        if start_date:
            events = [e for e in events if e.timestamp.date() >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp.date() <= end_date]
        
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events[:limit]
    
    def verify_chain(self) -> bool:
        """Verify integrity of audit chain"""
        if not self.events:
            return True
        
        prev_hash = None
        for event in self.events:
            if event.prev_hash != prev_hash:
                logger.error(f"Chain broken at event {event.id}")
                return False
            
            # Recompute hash
            computed = self._compute_hash_for_verify(event, prev_hash)
            if computed != event.hash:
                logger.error(f"Hash mismatch at event {event.id}")
                return False
            
            prev_hash = event.hash
        
        return True
    
    def _compute_hash_for_verify(self, event: AuditEvent, prev_hash: Optional[str]) -> str:
        """Compute hash for verification"""
        import hashlib
        
        data = f"{event.id}:{event.event_type}:{event.timestamp.isoformat()}:{event.service_name}"
        if event.details:
            data += f":{json.dumps(event.details, sort_keys=True)}"
        if prev_hash:
            data += f":{prev_hash}"
        
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    def get_signal_audit_trail(self, signal_id: str) -> List[AuditEvent]:
        """Get complete audit trail for a signal"""
        events = [e for e in self.events if e.signal_id == signal_id]
        events.sort(key=lambda x: x.timestamp)
        return events


# Global store
store = AuditStore()


@app.on_event("startup")
async def startup():
    """Connect to ClickHouse on startup"""
    host = os.getenv("CLICKHOUSE_HOST", "localhost")
    port = int(os.getenv("CLICKHOUSE_PORT", "9000"))
    await store.connect_clickhouse(host, port)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "total_events": len(store.events),
        "chain_valid": store.verify_chain()
    }


@app.post("/events", response_model=AuditEvent)
async def log_event(request: CreateAuditEventRequest):
    """Log an audit event"""
    return await store.log_event(request)


@app.get("/events")
async def get_events(
    event_type: Optional[AuditEventType] = None,
    symbol: Optional[str] = None,
    signal_id: Optional[str] = None,
    order_id: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(default=100, le=1000)
):
    """Query audit events"""
    events = store.get_events(
        event_type=event_type,
        symbol=symbol,
        signal_id=signal_id,
        order_id=order_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    return {"events": events}


@app.get("/events/{event_id}")
async def get_event(event_id: str):
    """Get specific audit event"""
    for event in store.events:
        if event.id == event_id:
            return event
    raise HTTPException(status_code=404, detail="Event not found")


@app.get("/signals/{signal_id}/trail")
async def get_signal_trail(signal_id: str):
    """Get complete audit trail for a signal"""
    trail = store.get_signal_audit_trail(signal_id)
    return {"signal_id": signal_id, "trail": trail}


@app.get("/verify")
async def verify_chain():
    """Verify integrity of audit chain"""
    is_valid = store.verify_chain()
    return {
        "valid": is_valid,
        "total_events": len(store.events),
        "last_hash": store.last_hash
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011, reload=True)
