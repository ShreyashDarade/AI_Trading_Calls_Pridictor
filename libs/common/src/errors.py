"""
Custom Error Classes for Indian AI Trader
Standardized exception handling across all services
"""
from typing import Optional, Dict, Any


class BaseAPIError(Exception):
    """Base exception for all API errors"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details
        }


class NotFoundError(BaseAPIError):
    """Resource not found error"""
    
    def __init__(
        self,
        resource: str,
        identifier: str,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"{resource} with identifier '{identifier}' not found"
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
            details=details or {"resource": resource, "identifier": identifier}
        )


class ValidationError(BaseAPIError):
    """Validation error"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(
            message=message,
            status_code=422,
            error_code="VALIDATION_ERROR",
            details=error_details
        )


class RateLimitError(BaseAPIError):
    """Rate limit exceeded error"""
    
    def __init__(
        self,
        limit: int,
        window: str = "minute",
        retry_after: Optional[int] = None
    ):
        message = f"Rate limit exceeded: {limit} requests per {window}"
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details={
                "limit": limit,
                "window": window,
                "retry_after_seconds": retry_after
            }
        )


class GrowwAPIError(BaseAPIError):
    """Error from Groww API"""
    
    def __init__(
        self,
        message: str,
        groww_error_code: Optional[str] = None,
        endpoint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if groww_error_code:
            error_details["groww_error_code"] = groww_error_code
        if endpoint:
            error_details["endpoint"] = endpoint
        super().__init__(
            message=f"Groww API Error: {message}",
            status_code=502,
            error_code="GROWW_API_ERROR",
            details=error_details
        )


class AuthenticationError(BaseAPIError):
    """Authentication failed error"""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_FAILED",
            details=details
        )


class AuthorizationError(BaseAPIError):
    """Authorization/permission error"""
    
    def __init__(
        self,
        message: str = "Permission denied",
        required_permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if required_permission:
            error_details["required_permission"] = required_permission
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_FAILED",
            details=error_details
        )


class InstrumentNotFoundError(NotFoundError):
    """Specific error for instrument not found"""
    
    def __init__(self, identifier: str, search_type: str = "id"):
        super().__init__(
            resource="Instrument",
            identifier=identifier,
            details={"search_type": search_type}
        )


class MarketClosedError(BaseAPIError):
    """Market is closed error"""
    
    def __init__(
        self,
        exchange: str,
        next_open: Optional[str] = None
    ):
        message = f"Market {exchange} is currently closed"
        super().__init__(
            message=message,
            status_code=503,
            error_code="MARKET_CLOSED",
            details={
                "exchange": exchange,
                "next_open": next_open
            }
        )


class TradingDisabledError(BaseAPIError):
    """Trading is disabled error"""
    
    def __init__(
        self,
        reason: str = "Trading is disabled",
        trading_type: str = "live"
    ):
        super().__init__(
            message=reason,
            status_code=403,
            error_code="TRADING_DISABLED",
            details={"trading_type": trading_type}
        )


class RiskLimitExceededError(BaseAPIError):
    """Risk limit exceeded error"""
    
    def __init__(
        self,
        limit_type: str,
        current_value: float,
        max_allowed: float,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"Risk limit exceeded for {limit_type}: {current_value} > {max_allowed}"
        error_details = details or {}
        error_details.update({
            "limit_type": limit_type,
            "current_value": current_value,
            "max_allowed": max_allowed
        })
        super().__init__(
            message=message,
            status_code=403,
            error_code="RISK_LIMIT_EXCEEDED",
            details=error_details
        )


class DataSnapshotError(BaseAPIError):
    """Error with data snapshot"""
    
    def __init__(
        self,
        snapshot_id: str,
        reason: str
    ):
        super().__init__(
            message=f"Data snapshot error for {snapshot_id}: {reason}",
            status_code=500,
            error_code="DATA_SNAPSHOT_ERROR",
            details={"snapshot_id": snapshot_id, "reason": reason}
        )


class KillSwitchActivatedError(BaseAPIError):
    """Kill switch has been activated"""
    
    def __init__(
        self,
        reason: str,
        activated_at: Optional[str] = None
    ):
        super().__init__(
            message=f"Kill switch activated: {reason}",
            status_code=503,
            error_code="KILL_SWITCH_ACTIVATED",
            details={"reason": reason, "activated_at": activated_at}
        )
