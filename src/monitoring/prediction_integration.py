"""
Integration Module: Connect Prediction Tracker to Live Trading

This module integrates the PredictionTracker with the trading engine
to automatically log and validate all AI predictions in real-time.

Author: AETHER Development Team
"""

import logging
from typing import Dict, Optional
from src.monitoring.prediction_tracker import get_prediction_tracker

logger = logging.getLogger("PredictionIntegration")


def log_oracle_prediction(
    symbol: str,
    prediction: str,
    confidence: float,
    current_price: float,
    regime: Optional[str] = None,
    rsi: Optional[float] = None,
    atr: Optional[float] = None
) -> None:
    """
    Log an Oracle prediction for later validation.
    
    Call this immediately after Oracle.predict() in trading_engine.py
    
    Args:
        symbol: Trading symbol
        prediction: "UP", "DOWN", or "NEUTRAL"
        confidence: Prediction confidence
        current_price: Current market price
        regime: Market regime
        rsi: RSI value
        atr: ATR value
    """
    try:
        tracker = get_prediction_tracker()
        tracker.record_prediction(
            symbol=symbol,
            prediction=prediction,
            confidence=confidence,
            current_price=current_price,
            horizon_candles=5,
            model="Oracle",
            regime=regime,
            rsi=rsi,
            atr=atr
        )
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


def validate_predictions(
    symbol: str,
    current_price: float,
    current_time: float
) -> None:
    """
    Validate pending predictions.
    
    Call this in the main trading loop to check if predictions can be validated.
    
    Args:
        symbol: Trading symbol
        current_price: Current market price
        current_time: Current timestamp
    """
    try:
        tracker = get_prediction_tracker()
        tracker.validate_predictions(symbol, current_price, current_time)
    except Exception as e:
        logger.error(f"Failed to validate predictions: {e}")


def print_accuracy_report() -> None:
    """Print current prediction accuracy report."""
    try:
        tracker = get_prediction_tracker()
        tracker.print_report()
    except Exception as e:
        logger.error(f"Failed to print report: {e}")


def get_accuracy_metrics() -> Dict:
    """
    Get current accuracy metrics.
    
    Returns:
        Dictionary with accuracy statistics
    """
    try:
        tracker = get_prediction_tracker()
        return tracker.get_accuracy_report()
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {}
