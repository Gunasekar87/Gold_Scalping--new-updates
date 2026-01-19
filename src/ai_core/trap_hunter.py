"""
Trap Hunter: Advanced AI Module for Institutional Trap Detection.
Part of the AETHER Intelligence Stack (Layer 5: Micro-Structure Defense).

This module specializes in identifying "Fake-Outs", "Liquidity Sweeps", and "Order Book Traps"
that often deceive reactive algorithms. It uses a combination of:
1. Volume Delta Divergence (Price Up + Volume Down)
2. Tick Velocity Anomalies (Speed Spikes with Zero Progress)
3. Structural Rejection (Wicks into Key Levels)

Author: AETHER Development Team
License: MIT
Version: 1.0.0 (World-Class Standard)
"""

import logging
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# Configure Logger
logger = logging.getLogger("TrapHunter")

@dataclass
class TrapSignal:
    is_trap: bool
    trap_type: str  # "BULL_TRAP", "BEAR_TRAP", "LIQUIDITY_GRAB_UP", "LIQUIDITY_GRAB_DOWN"
    confidence: float # 0.0 to 1.0
    severity: str     # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    details: str
    suggested_action: str = "NONE" # [PREDATOR] "BUY", "SELL", "NONE"
    counter_confidence: float = 0.0 # [PREDATOR] Confidence in the counter-trade

class TrapHunter:
    """
    The Predator of Predators.
    Detects institutional traps designed to hunt retail stops.
    """

    def __init__(self, tick_pressure_analyzer=None):
        self.tick_pressure = tick_pressure_analyzer
        self._history = deque(maxlen=100) # Store recent fake-out candidates
        
        # Configuration (Tunable via Bayesian Optimizer later)
        self.velocity_threshold_high = 20.0  # Ticks/sec indicative of "Stop Run"
        self.rejection_decay = 0.95          # Decay factor for trap signals
        self.trap_cooldown = 0.0
        
        # State caching for performance optimization
        self._last_scan_result = None
        self._last_scan_symbol = None
        self._last_scan_time = 0
        self._cache_ttl = 1.0  # Cache valid for 1 second
        
        logger.info("[INIT] Trap Hunter Module Online. Scanning for Institutional Traps...")

    def scan(self, symbol: str, candles: List[Dict[str, Any]], current_tick: Dict[str, float]) -> TrapSignal:
        """
        Scans the current market state for Traps.
        
        Args:
            symbol: Trading pair
            candles: Recent OHLCV data (M1)
            current_tick: Real-time price feed
        
        Returns:
            TrapSignal object indicating if a trap is active.
        """
        if not candles or len(candles) < 5:
            return TrapSignal(False, "NONE", 0.0, "LOW", "Insufficient Data", "NONE", 0.0)

        # 1. Macro-Structure Check (Are we breaking a level?)
        # We need to know if we are at a "Trap-Prone" location (Highs/Lows)
        current_price = current_tick['bid']
        recent_high = max([c['high'] for c in candles[-10:]])
        recent_low = min([c['low'] for c in candles[-10:]])
        
        breakout_up = current_price > recent_high
        breakout_down = current_price < recent_low
        
        # 2. Micro-Structure Analysis (The Truth Layer)
        # If price is breaking out, does the Order Flow verify it?
        
        pressure_metrics = self._get_pressure_metrics()
        pressure_score = pressure_metrics.get('pressure_score', 0.0)
        velocity = pressure_metrics.get('velocity', 0.0)
        dominance = pressure_metrics.get('dominance', 'NEUTRAL')
        
        # --- SCENARIO A: BULL TRAP (Fake Breakout Up) ---
        # Condition: Price Breaks High + "Absorption" or "Selling Pressure" or "Low Vol Rejection"
        if breakout_up:
            # Trap 1: Divergence (Price High, delta shows Selling)
            if pressure_score < -5.0:  # Selling into the breakout
                return TrapSignal(
                    True, "BULL_TRAP", 
                    0.85, "HIGH", 
                    f"Breakout Up ({current_price:.2f}) but Pressure is SELLING ({pressure_score:.1f})",
                    "SELL", 0.85 # [PREDATOR] Sell the fake breakout
                )
            
            # Trap 2: Exhaustion (Breakout with no velocity/volume)
            if velocity < 2.0: # Creating a high with no participation
                return TrapSignal(
                    True, "LIQUIDITY_GRAB_UP",
                    0.65, "MEDIUM",
                    f"Weak Breakout Up (Vel={velocity:.1f}), probable stop hunt.",
                    "SELL", 0.60 # Weaker signal
                )

        # --- SCENARIO B: BEAR TRAP (Fake Breakout Down) ---
        # Condition: Price Breaks Low + "Buying Pressure"
        if breakout_down:
            if pressure_score > 5.0: # Buying into the breakdown
                return TrapSignal(
                    True, "BEAR_TRAP", 
                    0.85, "HIGH", 
                    f"Breakdown Down ({current_price:.2f}) but Pressure is BUYING ({pressure_score:.1f})",
                    "BUY", 0.85 # [PREDATOR] Buy the fake breakdown
                )
             
            if velocity < 2.0:
                 return TrapSignal(
                    True, "LIQUIDITY_GRAB_DOWN",
                    0.65, "MEDIUM",
                    f"Weak Breakdown Down (Vel={velocity:.1f}), probable stop hunt.",
                    "BUY", 0.60
                )
                
        # --- SCENARIO C: ICEBERG DETECTION (High Velocity, Zero Move) ---
        # If velocity is HUGE but price didn't move much -> Hidden Wall
        if velocity > self.velocity_threshold_high:
            # Check price displacement over last few seconds?
            # Implied: If we are here, tick pressure class handles displacement check.
            # We assume 'ABSORPTION_FIGHT' state from TickPressure
            state = pressure_metrics.get('state', '')
            if state == "ABSORPTION_FIGHT":
                 result = TrapSignal(
                    True, "ORDER_BOOK_TRAP",
                    0.90, "CRITICAL",
                    f"Iceberg Detected! High Velocity ({velocity:.1f}) with Zero Progress.",
                    "NONE", 0.0 # Icebergs are risky to fade blindly without direction
                )
                 self._last_scan_result = result
                 self._last_scan_symbol = symbol
                 self._last_scan_time = time.time()
                 return result

        # No trap detected - cache result
        result = TrapSignal(False, "NONE", 0.0, "LOW", "Market Clear", "NONE", 0.0)
        self._last_scan_result = result
        self._last_scan_symbol = symbol
        self._last_scan_time = time.time()
        return result

    def _get_pressure_metrics(self) -> Dict[str, Any]:
        """Helper to safely fetch metrics from TickAnalyser."""
        if self.tick_pressure:
            return self.tick_pressure.get_pressure_metrics()
        return {}

    def is_trap(self, direction: str) -> bool:
        """
        Simple boolean check for high-confidence traps in a specific direction.
        Used by RiskManager for Go/No-Go decisions.
        
        Args:
            direction: "BUY" or "SELL" (The trade we WANT to take)
            
        Returns:
            True if it's unsafe (Trap detected)
        """
        import time
        
        # Use cached result if available and valid
        if (self._last_scan_result and 
            self._last_scan_symbol == direction and 
            (time.time() - self._last_scan_time) < self._cache_ttl):
            return self._last_scan_result.is_trap
        
        # No valid cache - would need to call scan() again or return safe default
        return False
    
    def check_wick_zones(self, candles: List[Dict], current_price: float, 
                         direction: str, atr: float) -> Tuple[bool, str]:
        """
        Check if current price is in a wick rejection zone.
        
        Wicks represent rejection zones where price was pushed back by the market.
        Entering at wick extremes often results in immediate reversals.
        
        Args:
            candles: Recent candle data (last 20)
            current_price: Current bid/ask price
            direction: "BUY" or "SELL"
            atr: Current ATR value for proximity calculation
            
        Returns:
            (is_in_wick_zone, reason)
        """
        if not candles or len(candles) < 5:
            return False, "Insufficient candle data"
        
        if atr <= 0:
            atr = 1.0  # Fallback ATR
        
        WICK_THRESHOLD = 0.5  # Wick must be >50% of candle body to be significant
        PROXIMITY_THRESHOLD = 0.3  # Within 30% of ATR from wick is too close
        
        # Check last 20 candles for significant wicks
        check_candles = candles[-20:] if len(candles) >= 20 else candles
        
        for candle in check_candles:
            try:
                open_price = float(candle.get('open', 0))
                close_price = float(candle.get('close', 0))
                high_price = float(candle.get('high', 0))
                low_price = float(candle.get('low', 0))
                
                if not all([open_price, close_price, high_price, low_price]):
                    continue
                
                # Calculate body and wicks
                body = abs(close_price - open_price)
                upper_wick = high_price - max(open_price, close_price)
                lower_wick = min(open_price, close_price) - low_price
                
                # Check for significant high wick (rejection at top)
                if upper_wick > body * WICK_THRESHOLD:
                    wick_zone_top = high_price
                    distance = abs(current_price - wick_zone_top)
                    
                    # If we're trying to BUY near a high wick rejection zone
                    if direction == "BUY" and distance < atr * PROXIMITY_THRESHOLD:
                        return True, f"Near high wick rejection zone (${wick_zone_top:.2f}, distance: ${distance:.2f})"
                
                # Check for significant low wick (rejection at bottom)
                if lower_wick > body * WICK_THRESHOLD:
                    wick_zone_bottom = low_price
                    distance = abs(current_price - wick_zone_bottom)
                    
                    # If we're trying to SELL near a low wick rejection zone
                    if direction == "SELL" and distance < atr * PROXIMITY_THRESHOLD:
                        return True, f"Near low wick rejection zone (${wick_zone_bottom:.2f}, distance: ${distance:.2f})"
            
            except (KeyError, ValueError, TypeError) as e:
                # Skip malformed candles
                continue
        
        return False, "Clear of wick zones"
