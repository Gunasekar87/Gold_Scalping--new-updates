"""
Regime Detector - AI Layer 11
==============================
Detects current market regime (trending, ranging, volatile, quiet).

This module provides regime classification to adapt trading strategies
based on current market conditions.

Author: AETHER Development Team
Version: 1.0
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("RegimeDetector")


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    BREAKOUT = "breakout"
    CHAOTIC = "chaotic"


@dataclass
class RegimeSignal:
    """Regime detection result."""
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    metrics: Dict[str, float]
    recommendation: str


class RegimeDetector:
    """
    Detects market regime using statistical analysis.
    
    Methods:
    1. ADX (Average Directional Index) for trend strength
    2. ATR ratio for volatility
    3. Price oscillation for ranging detection
    4. Breakout detection
    """
    
    def __init__(self):
        # Price history for regime analysis
        self._price_history = deque(maxlen=200)  # Last 200 candles
        
        # Regime history
        self._regime_history = deque(maxlen=50)
        
        # Configuration
        self.adx_trending_threshold = 20.0  # [FIX] Lowered from 25.0 to catch grinding trends
        self.adx_ranging_threshold = 20.0  # ADX < 20 = ranging
        self.atr_volatile_ratio = 1.5  # ATR > 1.5x average = volatile
        self.atr_quiet_ratio = 0.7  # ATR < 0.7x average = quiet
        
        logger.info("[REGIME DETECTOR] AI Layer 11 Online")
    
    def detect(self, candles: List[Dict], current_atr: float = 0.0) -> RegimeSignal:
        """
        Detect current market regime.
        
        Args:
            candles: Recent candle data (at least 50)
            current_atr: Current ATR value
            
        Returns:
            RegimeSignal with detection results
        """
        if not candles or len(candles) < 20:
            return RegimeSignal(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                metrics={},
                recommendation="Insufficient data"
            )
        
        # Update price history
        for candle in candles[-50:]:
            self._price_history.append(float(candle.get('close', 0)))
        
        # Calculate metrics
        metrics = {}
        
        # [GEOMETRICIAN] 1. Shannon Entropy (Chaos Detection)
        entropy = self._calculate_shannon_entropy(candles)
        metrics['entropy'] = entropy
        
        # [GEOMETRICIAN] 2. Hurst Exponent (Fractal Memory)
        hurst = self._calculate_hurst_exponent(candles)
        metrics['hurst'] = hurst

        # 1. Calculate ADX (trend strength)
        adx = self._calculate_adx(candles)
        metrics['adx'] = adx
        
        # 2. Calculate ATR ratio (volatility)
        atr_ratio = self._calculate_atr_ratio(candles, current_atr)
        metrics['atr_ratio'] = atr_ratio
        
        # 3. Calculate price oscillation (ranging behavior)
        oscillation = self._calculate_oscillation(candles)
        metrics['oscillation'] = oscillation
        
        # 4. Detect breakout
        is_breakout, breakout_direction = self._detect_breakout(candles)
        metrics['breakout'] = 1.0 if is_breakout else 0.0
        
        # Determine regime based on metrics
        regime, confidence = self._classify_regime(
            adx, atr_ratio, oscillation, is_breakout, breakout_direction, entropy, hurst
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(regime, metrics)
        
        # Store in history
        self._regime_history.append({
            'regime': regime,
            'confidence': confidence,
            'metrics': metrics.copy()
        })
        
        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            metrics=metrics,
            recommendation=recommendation
        )
    
    def _calculate_adx(self, candles: List[Dict], period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX).
        Simplified calculation for trend strength.
        
        Returns:
            ADX value (0-100, typically 0-50)
        """
        if len(candles) < period + 1:
            return 20.0  # Default neutral value
        
        try:
            # Calculate True Range and Directional Movement
            tr_list = []
            plus_dm_list = []
            minus_dm_list = []
            
            for i in range(1, min(len(candles), period + 10)):
                high = float(candles[-i].get('high', 0))
                low = float(candles[-i].get('low', 0))
                prev_high = float(candles[-i-1].get('high', 0))
                prev_low = float(candles[-i-1].get('low', 0))
                prev_close = float(candles[-i-1].get('close', 0))
                
                # True Range
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_list.append(tr)
                
                # Directional Movement
                plus_dm = max(high - prev_high, 0)
                minus_dm = max(prev_low - low, 0)
                
                if plus_dm > minus_dm:
                    plus_dm_list.append(plus_dm)
                    minus_dm_list.append(0)
                else:
                    plus_dm_list.append(0)
                    minus_dm_list.append(minus_dm)
            
            # Calculate smoothed averages
            if tr_list:
                atr = sum(tr_list[:period]) / period
                plus_di = (sum(plus_dm_list[:period]) / period) / atr * 100 if atr > 0 else 0
                minus_di = (sum(minus_dm_list[:period]) / period) / atr * 100 if atr > 0 else 0
                
                # Calculate DX and ADX
                dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
                
                return dx  # Simplified ADX (using DX as proxy)
            
            return 20.0
        except Exception as e:
            logger.debug(f"[REGIME] Error calculating ADX: {e}")
            return 20.0
    
    def _calculate_atr_ratio(self, candles: List[Dict], current_atr: float) -> float:
        """
        Calculate ATR ratio (current ATR vs average ATR).
        
        Returns:
            Ratio (1.0 = normal, >1.5 = volatile, <0.7 = quiet)
        """
        if current_atr <= 0 or len(candles) < 20:
            return 1.0
        
        try:
            # Calculate average ATR over last 50 candles
            atr_values = []
            for i in range(1, min(len(candles), 50)):
                high = float(candles[-i].get('high', 0))
                low = float(candles[-i].get('low', 0))
                atr_values.append(high - low)
            
            avg_atr = sum(atr_values) / len(atr_values) if atr_values else current_atr
            
            ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            return ratio
        except Exception as e:
            logger.debug(f"[REGIME] Error calculating ATR ratio: {e}")
            return 1.0
    
    def _calculate_oscillation(self, candles: List[Dict]) -> float:
        """
        Calculate price oscillation (ranging behavior).
        
        Returns:
            Oscillation score (0-1, higher = more ranging)
        """
        if len(candles) < 20:
            return 0.5
        
        try:
            # Get recent closes
            closes = [float(c.get('close', 0)) for c in candles[-20:]]
            
            # Calculate number of direction changes
            direction_changes = 0
            for i in range(1, len(closes) - 1):
                prev_dir = closes[i] - closes[i-1]
                next_dir = closes[i+1] - closes[i]
                
                if prev_dir * next_dir < 0:  # Direction changed
                    direction_changes += 1
            
            # Normalize to 0-1
            oscillation = direction_changes / (len(closes) - 2) if len(closes) > 2 else 0.5
            
            return oscillation
        except Exception as e:
            logger.debug(f"[REGIME] Error calculating oscillation: {e}")
            return 0.5
    
    def _calculate_shannon_entropy(self, candles: List[Dict], bins: int = 10) -> float:
        """
        Calculate Shannon Entropy to measure market disorder/randomness.
        H = -sum(p * log2(p))
        
        Returns:
            start_entropy: Normalized entropy (0.0=Ordered, 1.0=Max Chaos)
        """
        if len(candles) < 30:
            return 0.5
            
        try:
            # 1. Get returns
            closes = np.array([float(c.get('close', 0)) for c in candles])
            if len(closes) < 2:
                return 0.5
                
            returns = np.diff(closes) / closes[:-1]
            returns = returns[~np.isnan(returns)] # Filter NaNs
            
            if len(returns) == 0 or np.all(returns == 0):
                return 0.0
                
            # 2. Discretize into bins (Probability Mass Function)
            hist, _ = np.histogram(returns, bins=bins, density=True)
            
            # 3. Calculate Entopy
            # Convert density to probabilities (sum to 1) by multiplying by bin width
            # Approximate by just using normalized counts
            hist, _ = np.histogram(returns, bins=bins)
            probs = hist / np.sum(hist)
            probs = probs[probs > 0] # Avoid log(0)
            
            entropy = -np.sum(probs * np.log2(probs))
            
            # 4. Normalize (Max entropy is log2(bins))
            max_entropy = np.log2(bins)
            normalized_entropy = entropy / max_entropy
            
            return float(min(1.0, max(0.0, normalized_entropy)))
            
        except Exception as e:
            logger.debug(f"[GEOMETRICIAN] Error calculating Entropy: {e}")
            return 0.5

    def _calculate_hurst_exponent(self, candles: List[Dict]) -> float:
        """
        Calculate Hurst Exponent to measure long-term memory.
        H = 0.5 (Random), H > 0.5 (Trend), H < 0.5 (Mean Reversion)
        Using Rescaled Range (R/S) analysis.
        """
        if len(candles) < 50:
            return 0.5
            
        try:
            # Prepare data
            closes = np.array([float(c.get('close', 0)) for c in candles[-100:]]) # Use last 100 max
            
            # Simple R/S analysis approximation
            # (Full R/S is complex, we use a robust approximation for speed)
            
            # Calculate returns
            returns = np.diff(np.log(closes))
            
            # Mean return
            mean_return = np.mean(returns)
            
            # Cumulative deviation from mean
            deviations = returns - mean_return
            cum_deviations = np.cumsum(deviations)
            
            # Range
            R = np.max(cum_deviations) - np.min(cum_deviations)
            
            # Standard Deviation
            S = np.std(returns)
            
            if S == 0:
                return 0.5
                
            # R/S = c * (N)^H
            # log(R/S) = log(c) + H * log(N)
            # H ~ log(R/S) / log(N)
            
            RS = R / S
            N = len(returns)
            
            # Simplified point estimate
            H = np.log(RS) / np.log(N)
            
            return float(min(1.0, max(0.0, H)))
            
        except Exception as e:
            logger.debug(f"[GEOMETRICIAN] Error calculating Hurst: {e}")
            return 0.5

    def _detect_breakout(self, candles: List[Dict]) -> tuple:
        """
        Detect if price is breaking out of recent range.
        
        Returns:
            (is_breakout, direction)
        """
        if len(candles) < 20:
            return False, None
        
        try:
            # Get recent range (last 20 candles)
            recent = candles[-20:-1]
            recent_high = max(float(c.get('high', 0)) for c in recent)
            recent_low = min(float(c.get('low', 0)) for c in recent)
            
            # Current candle
            current = candles[-1]
            current_close = float(current.get('close', 0))
            
            # Check for breakout
            range_size = recent_high - recent_low
            breakout_threshold = range_size * 0.1  # 10% beyond range
            
            if current_close > recent_high + breakout_threshold:
                return True, "UP"
            elif current_close < recent_low - breakout_threshold:
                return True, "DOWN"
            
            return False, None
        except Exception as e:
            logger.debug(f"[REGIME] Error detecting breakout: {e}")
            return False, None
    
    def _classify_regime(self, adx: float, atr_ratio: float, oscillation: float,
                        is_breakout: bool, breakout_direction: Optional[str],
                        entropy: float = 0.5, hurst: float = 0.5) -> tuple:
        """
        Classify market regime based on metrics.
        
        Returns:
            (regime, confidence)
        """
        # [GEOMETRICIAN] Priority 0: CHAOS Detection
        # "Knowing when not to trade is as important as knowing when to trade."
        if entropy > 0.85:
            return MarketRegime.CHAOTIC, 0.95
            
        # Priority 1: Breakout
        if is_breakout:
            return MarketRegime.BREAKOUT, 0.9
        
        # Priority 2: Volatile/Quiet
        if atr_ratio > self.atr_volatile_ratio:
            return MarketRegime.VOLATILE, 0.85
        elif atr_ratio < self.atr_quiet_ratio:
            return MarketRegime.QUIET, 0.80
        
        # Priority 3: Trending vs Ranging (Refined by Hurst)
        is_trending_adx = adx > self.adx_trending_threshold
        is_trending_hurst = hurst > 0.55
        
        if is_trending_adx or is_trending_hurst:
            confidence = 0.75
            if is_trending_adx and is_trending_hurst:
                confidence = 0.90
                
            # Determine trend direction from recent price action
            if len(self._price_history) >= 20:
                recent_prices = list(self._price_history)[-20:]
                if recent_prices[-1] > recent_prices[0]:
                    return MarketRegime.TRENDING_UP, confidence
                else:
                    return MarketRegime.TRENDING_DOWN, confidence
            return MarketRegime.TRENDING_UP, 0.60
        
        elif adx < self.adx_ranging_threshold or oscillation > 0.5 or hurst < 0.45:
            confidence = 0.70
            return MarketRegime.RANGING, confidence
        
        # Default: Ranging with low confidence
        return MarketRegime.RANGING, 0.50
    
    def _generate_recommendation(self, regime: MarketRegime, 
                                metrics: Dict[str, float]) -> str:
        """Generate trading recommendation based on regime."""
        recommendations = {
            MarketRegime.TRENDING_UP: "Follow trend - wider stops, larger positions",
            MarketRegime.TRENDING_DOWN: "Follow trend - wider stops, larger positions",
            MarketRegime.RANGING: "Mean reversion - tighter stops, smaller positions",
            MarketRegime.VOLATILE: "Reduce position size, wider stops",
            MarketRegime.QUIET: "Normal operation, watch for breakout",
            MarketRegime.BREAKOUT: "Momentum trade - follow breakout direction"
        }
        
        return recommendations.get(regime, "Normal operation")
    
    def get_regime_parameters(self) -> Dict[str, float]:
        """
        Get recommended trading parameters based on current regime.
        
        Returns:
            Dict with parameter adjustments
        """
        if not self._regime_history:
            return {'stop_multiplier': 1.0, 'size_multiplier': 1.0}
        
        current = self._regime_history[-1]
        regime = current['regime']
        
        params = {
            MarketRegime.TRENDING_UP: {'stop_multiplier': 1.5, 'size_multiplier': 1.2},
            MarketRegime.TRENDING_DOWN: {'stop_multiplier': 1.5, 'size_multiplier': 1.2},
            MarketRegime.RANGING: {'stop_multiplier': 0.8, 'size_multiplier': 0.9},
            MarketRegime.VOLATILE: {'stop_multiplier': 2.0, 'size_multiplier': 0.7},
            MarketRegime.QUIET: {'stop_multiplier': 1.0, 'size_multiplier': 1.0},
            MarketRegime.BREAKOUT: {'stop_multiplier': 1.3, 'size_multiplier': 1.1}
        }
        
        return params.get(regime, {'stop_multiplier': 1.0, 'size_multiplier': 1.0})
    
    # ============================================================================
    # ENHANCEMENT 5: Strategy Parameters Integration
    # Added: January 4, 2026
    # Purpose: Provide regime-specific trading parameters for engine integration
    # ============================================================================
    
    def get_strategy_params(self) -> Dict[str, float]:
        """
        Get strategy parameters optimized for current regime.
        
        ENHANCEMENT 5: Direct integration with trading engine
        
        Returns:
            Dict with:
            - entry_threshold: Minimum confidence for trade entry (0.5-0.8)
            - tp_mult: Take profit multiplier (0.5-2.0)
            - position_size_mult: Position size adjustment (0.7-1.5)
            - stop_mult: Stop loss multiplier (0.8-2.0)
        """
        if not self._regime_history:
            return {
                'entry_threshold': 0.65,
                'tp_mult': 1.0,
                'position_size_mult': 1.0,
                'stop_mult': 1.0
            }
        
        current = self._regime_history[-1]
        regime = current['regime']
        confidence = current['confidence']
        
        # Regime-specific parameters
        strategy_params = {
            MarketRegime.TRENDING_UP: {
                'entry_threshold': 0.60,  # Lower threshold - trend is your friend
                'tp_mult': 1.5,  # Larger targets in trends
                'position_size_mult': 1.2,  # Larger positions
                'stop_mult': 1.5  # Wider stops
            },
            MarketRegime.TRENDING_DOWN: {
                'entry_threshold': 0.60,
                'tp_mult': 1.5,
                'position_size_mult': 1.2,
                'stop_mult': 1.5
            },
            MarketRegime.RANGING: {
                'entry_threshold': 0.75,  # Higher threshold - be selective
                'tp_mult': 0.8,  # Smaller targets in ranges
                'position_size_mult': 0.9,  # Smaller positions
                'stop_mult': 0.8  # Tighter stops
            },
            MarketRegime.VOLATILE: {
                'entry_threshold': 0.80,  # Very selective in volatility
                'tp_mult': 1.0,  # Normal targets
                'position_size_mult': 0.7,  # Much smaller positions
                'stop_mult': 2.0  # Much wider stops
            },
            MarketRegime.QUIET: {
                'entry_threshold': 0.65,  # Normal threshold
                'tp_mult': 1.0,  # Normal targets
                'position_size_mult': 1.0,  # Normal size
                'stop_mult': 1.0  # Normal stops
            },
            MarketRegime.BREAKOUT: {
                'entry_threshold': 0.55,  # Lower threshold - catch breakouts
                'tp_mult': 1.8,  # Large targets on breakouts
                'position_size_mult': 1.1,  # Slightly larger positions
                'stop_mult': 1.3  # Wider stops for volatility
            }
        }
        
        params = strategy_params.get(regime, {
            'entry_threshold': 0.65,
            'tp_mult': 1.0,
            'position_size_mult': 1.0,
            'stop_mult': 1.0
        })
        
        # Adjust based on confidence
        # If low confidence in regime detection, be more conservative
        if confidence < 0.6:
            params['entry_threshold'] = min(0.80, params['entry_threshold'] + 0.10)
            params['position_size_mult'] *= 0.9
        
        return params
    
    def get_current_regime(self) -> Optional[MarketRegime]:
        """
        Get the current detected regime.
        
        Returns:
            Current MarketRegime or None if no detection yet
        """
        if not self._regime_history:
            return None
        return self._regime_history[-1]['regime']
    
    def get_regime_confidence(self) -> float:
        """
        Get confidence in current regime detection.
        
        Returns:
            Confidence score (0.0-1.0)
        """
        if not self._regime_history:
            return 0.0
        return self._regime_history[-1]['confidence']
    
    def should_trade_in_regime(self, signal_confidence: float) -> tuple:
        """
        Determine if trading should proceed based on regime and signal confidence.
        
        Args:
            signal_confidence: AI signal confidence (0.0-1.0)
            
        Returns:
            (should_trade, reason)
        """
        if not self._regime_history:
            return True, "No regime data - proceed normally"
        
        regime = self._regime_history[-1]['regime']
        regime_confidence = self._regime_history[-1]['confidence']
        
        # Get regime-specific threshold
        params = self.get_strategy_params()
        threshold = params['entry_threshold']
        
        # Check if signal meets regime-adjusted threshold
        if signal_confidence < threshold:
            return False, f"Signal confidence {signal_confidence:.2f} below {regime.value} threshold {threshold:.2f}"
        
        # Additional checks for specific regimes
        if regime == MarketRegime.VOLATILE and regime_confidence > 0.8:
            if signal_confidence < 0.85:
                return False, "High volatility regime - require very high confidence"
        
        return True, f"Signal approved for {regime.value} regime"
