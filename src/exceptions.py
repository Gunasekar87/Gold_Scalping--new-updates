"""
Custom Exception Classes for AETHER Trading System.

This module defines all custom exceptions used throughout the trading bot,
following enterprise-grade error handling practices.

Author: AETHER Development Team
License: MIT
Version: 2.0.0
"""


class AETHERException(Exception):
    """Base exception class for all AETHER-related errors."""
    
    def __init__(self, message: str, error_code: str = "AETHER_000", details: dict = None):
        """
        Initialize AETHER exception.
        
        Args:
            message: Human-readable error description
            error_code: Machine-readable error code (e.g., "BROKER_001")
            details: Additional context as dictionary
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return formatted error string."""
        return self.message


# ============================================================================
# Broker & Connection Exceptions
# ============================================================================

class BrokerConnectionError(AETHERException):
    """Raised when broker connection fails or is lost."""
    
    def __init__(self, message: str, broker_name: str = None, **kwargs):
        details = {"broker": broker_name}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=message,
            error_code="BROKER_001",
            details=details
        )


class BrokerExecutionError(AETHERException):
    """Raised when order execution fails at broker level."""
    
    def __init__(self, message: str, symbol: str = None, order_type: str = None, **kwargs):
        details = {"symbol": symbol, "order_type": order_type}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=message,
            error_code="BROKER_002",
            details=details
        )


class InsufficientMarginError(AETHERException):
    """Raised when account has insufficient margin for trade."""
    
    def __init__(self, required: float, available: float, **kwargs):
        details = {"required": required, "available": available}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Insufficient margin: {required:.2f} required, {available:.2f} available",
            error_code="BROKER_003",
            details=details
        )


# ============================================================================
# AI & Model Exceptions
# ============================================================================

class ModelLoadError(AETHERException):
    """Raised when AI model fails to load."""
    
    def __init__(self, model_name: str, reason: str, **kwargs):
        details = {"model": model_name, "reason": reason}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Failed to load {model_name}: {reason}",
            error_code="AI_001",
            details=details
        )


class PredictionError(AETHERException):
    """Raised when AI prediction fails."""
    
    def __init__(self, model_name: str, reason: str, **kwargs):
        details = {"model": model_name, "reason": reason}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"{model_name} prediction failed: {reason}",
            error_code="AI_002",
            details=details
        )


class InvalidConfidenceError(AETHERException):
    """Raised when AI confidence is outside valid range [0, 1]."""
    
    def __init__(self, confidence: float, **kwargs):
        details = {"confidence": confidence}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Invalid confidence value: {confidence} (must be 0-1)",
            error_code="AI_003",
            details=details
        )


# ============================================================================
# Data & Validation Exceptions
# ============================================================================

class InsufficientDataError(AETHERException):
    """Raised when insufficient market data is available."""
    
    def __init__(self, required: int, available: int, data_type: str = "candles", **kwargs):
        details = {"required": required, "available": available, "type": data_type}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Insufficient {data_type}: {required} required, {available} available",
            error_code="DATA_001",
            details=details
        )


class InvalidDataError(AETHERException):
    """Raised when data validation fails."""
    
    def __init__(self, field: str, value: any, reason: str, **kwargs):
        details = {"field": field, "value": str(value), "reason": reason}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Invalid {field}: {value} ({reason})",
            error_code="DATA_002",
            details=details
        )


# ============================================================================
# Configuration Exceptions
# ============================================================================

class ConfigurationError(AETHERException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, reason: str, **kwargs):
        details = {"key": config_key, "reason": reason}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Configuration error for '{config_key}': {reason}",
            error_code="CONFIG_001",
            details=details
        )


# ============================================================================
# Risk Management Exceptions
# ============================================================================

class RiskLimitExceededError(AETHERException):
    """Raised when risk limits are exceeded."""
    
    def __init__(self, limit_type: str, current: float, maximum: float, **kwargs):
        details = {"type": limit_type, "current": current, "maximum": maximum}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"{limit_type} limit exceeded: {current:.2f} > {maximum:.2f}",
            error_code="RISK_001",
            details=details
        )


class DrawdownExceededError(AETHERException):
    """Raised when account drawdown exceeds threshold."""
    
    def __init__(self, current_dd: float, max_dd: float, **kwargs):
        details = {"current_drawdown": current_dd, "max_drawdown": max_dd}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Drawdown exceeded: {current_dd:.2%} > {max_dd:.2%}",
            error_code="RISK_002",
            details=details
        )


# ============================================================================
# Trading Logic Exceptions
# ============================================================================

class InvalidDecisionError(AETHERException):
    """Raised when trading decision is invalid."""
    
    def __init__(self, decision: str, valid_options: list, **kwargs):
        details = {"decision": decision, "valid_options": valid_options}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Invalid decision '{decision}' (valid: {valid_options})",
            error_code="LOGIC_001",
            details=details
        )


class PositionNotFoundError(AETHERException):
    """Raised when attempting to modify non-existent position."""
    
    def __init__(self, ticket: int, **kwargs):
        details = {"ticket": ticket}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Position not found: ticket {ticket}",
            error_code="LOGIC_002",
            details=details
        )


# ============================================================================
# Circuit Breaker Exceptions
# ============================================================================

class CircuitBreakerOpenError(AETHERException):
    """Raised when circuit breaker is OPEN and blocks operations."""
    
    def __init__(self, time_remaining: float, **kwargs):
        details = {"time_remaining": time_remaining}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Circuit breaker OPEN. Retry in {time_remaining:.1f}s",
            error_code="CIRCUIT_001",
            details=details
        )


# ============================================================================
# Market Condition Exceptions
# ============================================================================

class MarketPanicError(AETHERException):
    """Raised when market is in panic mode and trading should pause."""
    
    def __init__(self, volatility_pct: float, **kwargs):
        details = {"volatility": volatility_pct}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Market panic detected: {volatility_pct:.2%} volatility",
            error_code="MARKET_001",
            details=details
        )


class HighSpreadError(AETHERException):
    """Raised when spread is too high for safe execution."""
    
    def __init__(self, spread_pips: float, threshold: float, **kwargs):
        details = {"spread": spread_pips, "threshold": threshold}
        details.update(kwargs.get('details', {}))
        super().__init__(
            message=f"Spread too high: {spread_pips:.1f} > {threshold:.1f} pips",
            error_code="MARKET_002",
            details=details
        )
