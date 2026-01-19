"""
Global Brain - Inter-Market Correlation Engine (Layer 9)

This module implements "God Mode" intelligence by monitoring correlated assets
(DXY, US10Y, SPX500) to predict Gold (XAUUSD) movements before they happen.

Concept:
- DXY (Dollar) UP -> Gold DOWN (Inverse Correlation)
- US10Y (Yields) UP -> Gold DOWN (Inverse Correlation)
- SPX500 (Risk On) UP -> Gold DOWN (Safe Haven Outflow)

The engine calculates a "Correlation Score" (-1.0 to +1.0) to bias the main trading engine.
"""

import logging
import time
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass

try:
    import MetaTrader5 as mt5
    from datetime import datetime
except Exception:  # pragma: no cover
    mt5 = None
    datetime = None

logger = logging.getLogger("GlobalBrain")

@dataclass
class CorrelationSignal:
    score: float  # -1.0 (Strong Bearish for Gold) to +1.0 (Strong Bullish for Gold)
    driver: str   # Which asset is driving the move (e.g., "DXY_SPIKE")
    confidence: float
    timestamp: float

class GlobalBrain:
    def __init__(self, market_data_manager):
        self.market_data = market_data_manager
        # Use proxies from config when available (MarketDataManager.CorrelationMonitor)
        usd_proxy = None
        risk_proxy = None
        try:
            if getattr(self.market_data, 'macro_eye', None):
                usd_proxy = getattr(self.market_data.macro_eye, 'usd_symbol', None)
                risk_proxy = getattr(self.market_data.macro_eye, 'risk_symbol', None)
        except Exception:
            usd_proxy = None
            risk_proxy = None

        # ENHANCEMENT 6: Expanded Correlation Matrix
        # Correlation signs relative to Gold (XAUUSD)
        # - USD strength up => Gold down (inverse)
        # - Risk-on up => Gold down (inverse)
        # - VIX up => Gold up (positive - fear drives gold)
        # - US10Y up => Gold down (inverse - yields compete with gold)
        self.correlations = {}
        if usd_proxy:
            self.correlations[usd_proxy] = -0.85  # Strong inverse
        if risk_proxy:
            self.correlations[risk_proxy] = -0.40  # Moderate inverse
        
        # ENHANCEMENT 6: Add VIX and US10Y
        self.correlations['VIX'] = 0.60   # Positive - fear index
        self.correlations['US10Y'] = -0.50  # Inverse - yield competition

        # ENHANCEMENT 6: Expanded Thresholds
        # Thresholds for "Significant Move" (percent change in last 1 min)
        self.thresholds = {}
        if usd_proxy:
            self.thresholds[usd_proxy] = 0.02  # 2% move in USD
        if risk_proxy:
            self.thresholds[risk_proxy] = 0.05  # 5% move in equities
        
        # ENHANCEMENT 6: Add VIX and US10Y thresholds
        self.thresholds['VIX'] = 0.10  # 10% move in VIX (volatile)
        self.thresholds['US10Y'] = 0.01  # 1% move in yields

        # Cache last computed signal to avoid overloading terminal on every tick
        self._cache_interval_s = 1.0
        self._last_signal: Optional[CorrelationSignal] = None
        self._last_fetch_ts = 0.0
        self.last_prices = {}
        self.last_update = 0
        
        logger.info(f"[GLOBAL BRAIN] Initialized with {len(self.correlations)} correlation pairs")
        
    def update_reference_prices(self, prices: Dict[str, float]):
        """Update the baseline prices for calculation."""
        for symbol, price in prices.items():
            self.last_prices[symbol] = price
        self.last_update = time.time()

    def analyze_impact(self, current_prices: Dict[str, float]) -> CorrelationSignal:
        """
        Analyze global markets to predict Gold's next move.
        Returns a signal bias for XAUUSD.
        """
        total_score = 0.0
        primary_driver = "NEUTRAL"
        max_impact = 0.0
        
        for symbol, correlation in self.correlations.items():
            if symbol not in current_prices or symbol not in self.last_prices:
                continue
                
            # Calculate % Change
            curr = current_prices[symbol]
            prev = self.last_prices[symbol]
            if prev == 0: continue
            
            pct_change = ((curr - prev) / prev) * 100.0
            
            # Check if move is significant (Noise Filter)
            threshold = self.thresholds.get(symbol, 0.05)
            if abs(pct_change) < threshold:
                continue
                
            # Calculate Impact Score
            # Example: DXY moves +0.1% (Bullish Dollar). Correlation is -0.85.
            # Impact = 0.1 * -0.85 = -0.085 (Bearish for Gold)
            impact = pct_change * correlation * 10.0 # Multiplier for sensitivity
            
            total_score += impact
            
            if abs(impact) > max_impact:
                max_impact = abs(impact)
                direction = "SURGE" if pct_change > 0 else "DUMP"
                primary_driver = f"{symbol}_{direction}"
                
        # Normalize Score to -1.0 to 1.0
        final_score = np.clip(total_score, -1.0, 1.0)
        
        # Confidence based on how many assets agree
        confidence = min(abs(total_score), 1.0)
        
        if abs(final_score) > 0.3:
            logger.info(f"[GLOBAL_BRAIN] Signal: {final_score:.2f} | Driver: {primary_driver} | DXY Impact: {total_score:.2f}")
            
        return CorrelationSignal(
            score=final_score,
            driver=primary_driver,
            confidence=confidence,
            timestamp=time.time()
        )

    def get_bias(self) -> float:
        """Get the current trading bias (-1.0 to 1.0) from live proxy symbols."""
        sig = self.get_bias_signal()
        return float(sig.score) if sig else 0.0

    def get_bias_signal(self) -> CorrelationSignal:
        """Fetch live proxy ticks and compute a correlation bias signal.

        Returns a CorrelationSignal with:
        - score in [-1, 1]
        - confidence in [0, 1]
        - driver describing primary proxy move
        """
        now = time.time()
        if self._last_signal is not None and (now - self._last_fetch_ts) < self._cache_interval_s:
            return self._last_signal

        if not self.correlations:
            self._last_fetch_ts = now
            self._last_signal = CorrelationSignal(score=0.0, driver="NO_PROXIES", confidence=0.0, timestamp=now)
            return self._last_signal

        if mt5 is None or datetime is None:
            self._last_fetch_ts = now
            self._last_signal = CorrelationSignal(score=0.0, driver="MT5_UNAVAILABLE", confidence=0.0, timestamp=now)
            return self._last_signal

        total_score = 0.0
        primary_driver = "NEUTRAL"
        max_impact = 0.0
        contributors = 0

        for sym, correlation in self.correlations.items():
            try:
                ticks = mt5.copy_ticks_from(sym, datetime.now(), 200, mt5.COPY_TICKS_ALL)
                if ticks is None or len(ticks) < 2:
                    continue
                current = float(ticks[-1][1])  # bid
                prev = float(ticks[0][1])
                if prev == 0:
                    continue

                pct_change = ((current - prev) / prev) * 100.0
                threshold = float(self.thresholds.get(sym, 0.0))
                if threshold > 0.0 and abs(pct_change) < threshold:
                    continue

                impact = pct_change * float(correlation) * 10.0
                total_score += impact
                contributors += 1

                if abs(impact) > max_impact:
                    max_impact = abs(impact)
                    direction = "SURGE" if pct_change > 0 else "DUMP"
                    primary_driver = f"{sym}_{direction}"
            except Exception:
                continue

        final_score = float(np.clip(total_score, -1.0, 1.0))
        confidence = 0.0
        if contributors > 0:
            # Confidence grows with magnitude and number of agreeing contributors
            confidence = float(min(abs(total_score), 1.0))

        self._last_fetch_ts = now
        self._last_signal = CorrelationSignal(
            score=final_score,
            driver=primary_driver,
            confidence=confidence,
            timestamp=now,
        )

        return self._last_signal
