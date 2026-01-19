"""
Supervisor Agent - The "Boss" of the Hierarchical AI System.

This module implements the Supervisor agent which is responsible for:
1. Detecting the current Market Regime (Range, Trend, Chaos)
2. Routing control to the appropriate Worker Agent
3. Managing global risk parameters based on regime

Author: AETHER Development Team
License: MIT
Version: 5.0.0
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.ai_core.regime_detector import RegimeDetector, MarketRegime

logger = logging.getLogger("Supervisor")

@dataclass
class Regime:
    name: str
    confidence: float
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict) # [NEW] Store Physics metrics

class Supervisor:
    """
    The Supervisor Agent determines the market regime and selects the active strategy.
    
    [UPGRADE] Integrated 'The Geometrician' (RegimeDetector) for Physics-based analysis.
    """
    
    def __init__(self):
        self.geometrician = RegimeDetector() # [AI LAYER 11]
        self.current_regime = "UNKNOWN"
        logger.info("[SUPERVISOR] Agent Initialized with Geometrician Engine")

    def detect_regime(self, market_data: Dict[str, Any], candles: Optional[List[Dict]] = None) -> Regime:
        """
        Analyze market data to classify the current regime.
        
        Args:
            market_data: Dictionary of indicators (ATR, ADX, etc.)
            candles: List of candlestick data (Required for Entropy/Hurst)
            
        Returns:
            Regime object with classification and metrics.
        """
        try:
            # 1. [GEOMETRICIAN] Advanced Regime Detection (Entropy/Hurst)
            geo_regime = None
            geo_metrics = {}
            if candles:
                atr_val = market_data.get('atr', 0.0)
                geo_signal = self.geometrician.detect(candles, current_atr=atr_val)
                geo_metrics = geo_signal.metrics
                
                # Check for CHAOS (Entropy > 0.85)
                # The Geometrician's detect() already handles this and returns MarketRegime.CHAOTIC
                if geo_signal.regime == MarketRegime.CHAOTIC:
                    return Regime(
                        name="CHAOS",
                        confidence=geo_signal.confidence,
                        description=f"High Entropy ({geo_metrics.get('entropy', 0):.2f}) - Random Walk",
                        metrics=geo_metrics
                    )

            # 2. [PHYSICIST] Pressure-Based Override (Reynolds Number)
            # If Reynolds Number is High -> TURBULENT (Breakout/Trend)
            # If Reynolds Number is Low -> LAMINAR (Range)
            pressure = market_data.get('pressure_metrics', {})
            physics = pressure.get('physics', {}) # From new tick_pressure.py
            
            re_number = physics.get('reynolds_number', 0.0)
            
            if re_number > 500: # Turbulent Flow
                return Regime(
                    name="TREND",
                    confidence=0.9,
                    description=f"Turbulent Flow (Re: {re_number:.0f}) - Momentum Breakout",
                    metrics={**geo_metrics, 'reynolds': re_number}
                )

            # 3. [LEGACY + HYBRID] Standard Classification
            # Fallback to ADX/Vol/TrendStrength if Physics doesn't force an override
            atr = market_data.get('atr', 0.0)
            trend_strength = market_data.get('trend_strength', 0.0)
            volatility_ratio = market_data.get('volatility_ratio', 1.0)
            if volatility_ratio is None or not isinstance(volatility_ratio, (int, float)):
                volatility_ratio = 1.0
            
            # Macro Context
            macro_context = market_data.get('macro_context', [0.0, 0.0])
            if not isinstance(macro_context, (list, tuple)) or len(macro_context) < 1:
                macro_context = [0.0, 0.0]
            usd_velocity = abs(float(macro_context[0] or 0.0))
            
            # Legacy Chaos Check
            if volatility_ratio > 2.5 or usd_velocity > 5.0:
                 return Regime(
                     name="CHAOS", 
                     confidence=1.0, 
                     description="Extreme Volatility / News Event",
                     metrics=geo_metrics
                 )
                
            if trend_strength > 0.3:
                return Regime(
                    name="TREND", 
                    confidence=trend_strength, 
                    description="Strong Directional Movement",
                    metrics=geo_metrics
                )
                
            return Regime(
                name="RANGE", 
                confidence=1.0 - trend_strength, 
                description="Sideways / Mean Reversion",
                metrics=geo_metrics
            )
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return Regime("RANGE", 0.0, "Fallback due to error")

    def get_active_worker(self, regime: Regime) -> str:
        """
        Select the appropriate worker for the detected regime.
        """
        if regime.name == "TREND":
            return "TREND_WORKER"
        elif regime.name == "RANGE":
            return "RANGE_WORKER"
        else:
            return "DEFENSIVE_WORKER" # For CHAOS or Unknown
