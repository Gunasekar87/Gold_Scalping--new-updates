"""
Worker Agents - Specialized Trading Strategies.

This module implements the Worker Agents that execute specific strategies
assigned by the Supervisor.

1. RangeWorker: Mean Reversion (Buy Low, Sell High)
2. TrendWorker: Momentum (Buy High, Sell Higher)

Author: AETHER Development Team
License: MIT
Version: 5.0.0
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("Workers")

class BaseWorker:
    def get_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Returns (Action, Confidence, Reason)"""
        raise NotImplementedError

class RangeWorker(BaseWorker):
    """
    Specialist in Sideways Markets.
    Strategy: Aggressive Mean Reversion (Scalping).
    """
    def get_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float, str]:
        rsi = market_data.get('rsi', 50.0)
        pressure = market_data.get('pressure_metrics', {})
        
        # Base Logic
        action = "HOLD"
        confidence = 0.0
        reason = ""
        
        if rsi < 50:
            action = "BUY"
            confidence = (50 - rsi) / 50.0
            reason = f"Range Low (RSI {rsi:.1f})"
        elif rsi > 50:
            action = "SELL"
            confidence = (rsi - 50) / 50.0
            reason = f"Range High (RSI {rsi:.1f})"
            
        # [HIGHEST INTELLIGENCE] Holographic Filter
        if pressure and action != "HOLD":
            dom = pressure.get('dominance', 'NEUTRAL')
            intensity = pressure.get('intensity', 'NORMAL')
            
            if action == "BUY":
                if dom == "BUY":
                    confidence += 0.1
                    reason += " + Buy Pressure"
                elif dom == "SELL" and intensity == "HIGH":
                    confidence -= 0.2
                    reason += " - High Sell Pressure"
            elif action == "SELL":
                if dom == "SELL":
                    confidence += 0.1
                    reason += " + Sell Pressure"
                elif dom == "BUY" and intensity == "HIGH":
                    confidence -= 0.2
                    reason += " - High Buy Pressure"
                    
        # Boost confidence to ensure trade execution (unless penalized by High Intensity Pressure)
        if action != "HOLD":
            # [CRITICAL FIX] Do not forcefully boost confidence if there is a High Pressure Warning
            # "High Buy Pressure" means Breakout Risk -> Don't fade it blindly.
            if "High" in reason and "-" in reason and "Pressure" in reason:
                # We have a specific warning like "- High Buy Pressure"
                # Don't boost. Let it fail if confidence is low.
                pass 
            else:
                confidence = max(0.6, confidence) # Minimum 0.6 to trigger normal trades

        # Safety clamp
        confidence = max(0.0, min(1.0, confidence))
            
        return action, confidence, reason

class TrendWorker(BaseWorker):
    """
    Specialist in Trending Markets.
    Strategy: Aggressive Trend Following.
    """
    def get_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float, str]:
        trend_strength = market_data.get('trend_strength', 0.0)
        rsi = market_data.get('rsi', 50.0)
        pressure = market_data.get('pressure_metrics', {})
        
        # Base Logic
        action = "HOLD"
        confidence = 0.0
        reason = ""
        
        if rsi > 50:
            action = "BUY"
            confidence = trend_strength
            reason = f"Uptrend Momentum (RSI {rsi:.1f})"
        elif rsi < 50:
            action = "SELL"
            confidence = trend_strength
            reason = f"Downtrend Momentum (RSI {rsi:.1f})"
            
        # [HIGHEST INTELLIGENCE] Holographic Filter
        if pressure and action != "HOLD":
            dom = pressure.get('dominance', 'NEUTRAL')
            intensity = pressure.get('intensity', 'NORMAL')
            
            if action == "BUY":
                if dom == "BUY":
                    confidence += 0.15
                    reason += " + Buy Pressure"
                elif dom == "SELL":
                    confidence -= 0.1
                    reason += " - Sell Pressure"
            elif action == "SELL":
                if dom == "SELL":
                    confidence += 0.15
                    reason += " + Sell Pressure"
                elif dom == "BUY":
                    confidence -= 0.1
                    reason += " - Buy Pressure"

        if action != "HOLD":
            confidence = max(0.6, confidence)

        # Safety clamp
        confidence = max(0.0, min(1.0, confidence))

        return action, confidence, reason
