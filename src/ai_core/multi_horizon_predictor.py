"""
Multi-Horizon Predictor - Predicts market movement across multiple timeframes

This module provides intelligent prediction of price movement for the next
1, 3, and 5 candles to enable smart hedge timing and reversal detection.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("MultiHorizonPredictor")


@dataclass
class HorizonPrediction:
    """Prediction for a specific time horizon."""
    direction: str  # 'UP', 'DOWN', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    expected_pips: float  # Expected movement in pips
    reversal_detected: bool  # True if reversal is predicted
    reversal_in_candles: Optional[int]  # Candles until reversal
    reasoning: str


@dataclass
class MultiHorizonResult:
    """Combined prediction across all horizons."""
    immediate: HorizonPrediction  # Next 1 candle
    short_term: HorizonPrediction  # Next 3 candles
    medium_term: HorizonPrediction  # Next 5 candles
    consensus_direction: str
    consensus_confidence: float
    should_wait_for_reversal: bool
    optimal_action: str  # 'TRADE_NOW', 'WAIT', 'HEDGE_NOW'
    reasoning: str


class MultiHorizonPredictor:
    """
    Predicts price movement across multiple time horizons.
    
    Uses pattern recognition and indicator analysis to predict:
    - Immediate (1 candle): For quick decisions
    - Short-term (3 candles): For trend confirmation
    - Medium-term (5 candles): For reversal detection
    """
    
    def __init__(self):
        self.prediction_history = []
        logger.info("MultiHorizonPredictor initialized")
    
    def predict_all_horizons(self, candles: List[Dict], current_tick: Dict) -> MultiHorizonResult:
        """
        Predict market movement across all time horizons.
        
        Args:
            candles: List of recent candle data (OHLCV)
            current_tick: Current tick data
            
        Returns:
            MultiHorizonResult with predictions for all horizons
        """
        try:
            # Validate input data
            if not candles or len(candles) < 20:
                logger.warning("[PREDICTOR_ERROR] Insufficient candle data")
                return self._create_neutral_result("Insufficient candle data")
            
            if not current_tick:
                logger.warning("[PREDICTOR_ERROR] Missing tick data")
                return self._create_neutral_result("Missing tick data")
            
            # Predict each horizon with individual error handling
            immediate = self._predict_immediate_safe(candles, current_tick)
            short_term = self._predict_short_term_safe(candles, current_tick)
            medium_term = self._predict_medium_term_safe(candles, current_tick)
            
            # Determine consensus
            try:
                consensus = self._determine_consensus(immediate, short_term, medium_term)
                return consensus
            except Exception as e:
                logger.error(f"[PREDICTOR_ERROR] Consensus calculation failed: {e}")
                return self._create_neutral_result(f"Consensus error: {str(e)}")
                
        except Exception as e:
            logger.error(f"[PREDICTOR_ERROR] Prediction failed: {e}")
            return self._create_neutral_result(f"Prediction error: {str(e)}")
    
    def _predict_immediate_safe(self, candles: List[Dict], tick: Dict) -> HorizonPrediction:
        """Safe wrapper for immediate prediction."""
        try:
            return self._predict_immediate(candles, tick)
        except Exception as e:
            logger.warning(f"[PREDICTOR_ERROR] Immediate prediction failed: {e}")
            return HorizonPrediction(
                direction='NEUTRAL',
                confidence=0.5,
                expected_pips=0.0,
                reversal_detected=False,
                reversal_in_candles=None,
                reasoning=f'Immediate prediction error: {str(e)}'
            )
    
    def _predict_short_term_safe(self, candles: List[Dict], tick: Dict) -> HorizonPrediction:
        """Safe wrapper for short-term prediction."""
        try:
            return self._predict_short_term(candles, tick)
        except Exception as e:
            logger.warning(f"[PREDICTOR_ERROR] Short-term prediction failed: {e}")
            return HorizonPrediction(
                direction='NEUTRAL',
                confidence=0.5,
                expected_pips=0.0,
                reversal_detected=False,
                reversal_in_candles=None,
                reasoning=f'Short-term prediction error: {str(e)}'
            )
    
    def _predict_medium_term_safe(self, candles: List[Dict], tick: Dict) -> HorizonPrediction:
        """Safe wrapper for medium-term prediction."""
        try:
            return self._predict_medium_term(candles, tick)
        except Exception as e:
            logger.warning(f"[PREDICTOR_ERROR] Medium-term prediction failed: {e}")
            return HorizonPrediction(
                direction='NEUTRAL',
                confidence=0.5,
                expected_pips=0.0,
                reversal_detected=False,
                reversal_in_candles=None,
                reasoning=f'Medium-term prediction error: {str(e)}'
            )
    
    def _predict_immediate(self, candles: List[Dict], tick: Dict) -> HorizonPrediction:
        """
        Predict next 1 candle using fast indicators.
        
        Focus: Quick momentum, RSI extremes, recent price action
        """
        recent = candles[-20:]  # Last 20 candles
        
        # Extract closes
        closes = [c.get('close', 0) for c in recent]
        highs = [c.get('high', 0) for c in recent]
        lows = [c.get('low', 0) for c in recent]
        
        current_price = tick.get('bid', closes[-1])
        
        # Calculate RSI
        rsi = self._calculate_rsi(closes, period=14)
        
        # Calculate momentum
        momentum = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
        
        # Check for extreme RSI
        rsi_signal = None
        if rsi < 30:
            rsi_signal = ('UP', 0.7, 'RSI oversold')
        elif rsi > 70:
            rsi_signal = ('DOWN', 0.7, 'RSI overbought')
        
        # Check momentum
        momentum_signal = None
        if momentum > 0.5:
            momentum_signal = ('UP', 0.6, 'Strong upward momentum')
        elif momentum < -0.5:
            momentum_signal = ('DOWN', 0.6, 'Strong downward momentum')
        
        # Price action pattern
        last_3_candles = recent[-3:]
        bullish_candles = sum(1 for c in last_3_candles if c.get('close', 0) > c.get('open', 0))
        bearish_candles = 3 - bullish_candles
        
        pattern_signal = None
        if bullish_candles == 3:
            pattern_signal = ('UP', 0.65, 'Three consecutive bullish candles')
        elif bearish_candles == 3:
            pattern_signal = ('DOWN', 0.65, 'Three consecutive bearish candles')
        
        # Combine signals
        signals = [s for s in [rsi_signal, momentum_signal, pattern_signal] if s is not None]
        
        if not signals:
            return HorizonPrediction(
                direction='NEUTRAL',
                confidence=0.5,
                expected_pips=0.0,
                reversal_detected=False,
                reversal_in_candles=None,
                reasoning='No clear immediate signal'
            )
        
        # Vote on direction
        up_votes = sum(conf for dir, conf, _ in signals if dir == 'UP')
        down_votes = sum(conf for dir, conf, _ in signals if dir == 'DOWN')
        
        if up_votes > down_votes:
            direction = 'UP'
            confidence = min(up_votes / len(signals), 1.0)
            reasoning = ' | '.join([r for d, c, r in signals if d == 'UP'])
        elif down_votes > up_votes:
            direction = 'DOWN'
            confidence = min(down_votes / len(signals), 1.0)
            reasoning = ' | '.join([r for d, c, r in signals if d == 'DOWN'])
        else:
            direction = 'NEUTRAL'
            confidence = 0.5
            reasoning = 'Mixed signals'
        
        # Estimate expected move (based on recent ATR)
        atr = self._calculate_atr(highs, lows, closes, period=14)
        expected_pips = atr * 0.3 if direction == 'UP' else -atr * 0.3 if direction == 'DOWN' else 0.0
        
        return HorizonPrediction(
            direction=direction,
            confidence=confidence,
            expected_pips=expected_pips,
            reversal_detected=False,
            reversal_in_candles=None,
            reasoning=reasoning
        )
    
    def _predict_short_term(self, candles: List[Dict], tick: Dict) -> HorizonPrediction:
        """
        Predict next 3 candles using trend indicators.
        
        Focus: MACD, moving averages, trend strength
        """
        recent = candles[-30:]  # Last 30 candles
        closes = [c.get('close', 0) for c in recent]
        highs = [c.get('high', 0) for c in recent]
        lows = [c.get('low', 0) for c in recent]
        
        # Calculate moving averages
        ma_fast = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
        ma_slow = np.mean(closes[-15:]) if len(closes) >= 15 else closes[-1]
        
        # MA crossover
        ma_signal = None
        if ma_fast > ma_slow * 1.001:  # Fast above slow (bullish)
            ma_signal = ('UP', 0.7, 'MA crossover bullish')
        elif ma_fast < ma_slow * 0.999:  # Fast below slow (bearish)
            ma_signal = ('DOWN', 0.7, 'MA crossover bearish')
        
        # Trend strength
        trend_strength = abs(ma_fast - ma_slow) / ma_slow * 100
        
        # Higher highs / Lower lows
        recent_5 = recent[-5:]
        higher_highs = all(recent_5[i].get('high', 0) >= recent_5[i-1].get('high', 0) for i in range(1, len(recent_5)))
        lower_lows = all(recent_5[i].get('low', 0) <= recent_5[i-1].get('low', 0) for i in range(1, len(recent_5)))
        
        structure_signal = None
        if higher_highs:
            structure_signal = ('UP', 0.65, 'Higher highs pattern')
        elif lower_lows:
            structure_signal = ('DOWN', 0.65, 'Lower lows pattern')
        
        # Combine signals
        signals = [s for s in [ma_signal, structure_signal] if s is not None]
        
        if not signals:
            return HorizonPrediction(
                direction='NEUTRAL',
                confidence=0.5,
                expected_pips=0.0,
                reversal_detected=False,
                reversal_in_candles=None,
                reasoning='No clear short-term trend'
            )
        
        # Vote
        up_votes = sum(conf for dir, conf, _ in signals if dir == 'UP')
        down_votes = sum(conf for dir, conf, _ in signals if dir == 'DOWN')
        
        if up_votes > down_votes:
            direction = 'UP'
            confidence = min(up_votes / max(len(signals), 1), 1.0)
            reasoning = ' | '.join([r for d, c, r in signals if d == 'UP'])
        elif down_votes > up_votes:
            direction = 'DOWN'
            confidence = min(down_votes / max(len(signals), 1), 1.0)
            reasoning = ' | '.join([r for d, c, r in signals if d == 'DOWN'])
        else:
            direction = 'NEUTRAL'
            confidence = 0.5
            reasoning = 'Balanced short-term signals'
        
        atr = self._calculate_atr(highs, lows, closes, period=14)
        expected_pips = atr * 0.8 if direction == 'UP' else -atr * 0.8 if direction == 'DOWN' else 0.0
        
        return HorizonPrediction(
            direction=direction,
            confidence=confidence,
            expected_pips=expected_pips,
            reversal_detected=False,
            reversal_in_candles=None,
            reasoning=reasoning
        )
    
    def _predict_medium_term(self, candles: List[Dict], tick: Dict) -> HorizonPrediction:
        """
        Predict next 5 candles with focus on REVERSAL DETECTION.
        
        Focus: Divergences, exhaustion patterns, support/resistance
        """
        if len(candles) < 60:
            return HorizonPrediction(
                direction='NEUTRAL',
                confidence=0.5,
                expected_pips=0.0,
                reversal_detected=False,
                reversal_in_candles=None,
                reasoning='Insufficient data for medium-term prediction'
            )
        
        recent = candles[-60:]
        closes = [c.get('close', 0) for c in recent]
        highs = [c.get('high', 0) for c in recent]
        lows = [c.get('low', 0) for c in recent]
        volumes = [c.get('tick_volume', 0) for c in recent]
        
        # Calculate RSI for divergence detection
        rsi_values = [self._calculate_rsi(closes[:i+1], period=14) for i in range(14, len(closes))]
        
        # Check for RSI divergence (reversal signal)
        reversal_detected = False
        reversal_direction = None
        reversal_in = None
        
        if len(rsi_values) >= 10:
            # Bullish divergence: Price making lower lows, RSI making higher lows
            price_trend = closes[-1] < closes[-10]
            rsi_trend = rsi_values[-1] > rsi_values[-10]
            
            if price_trend and rsi_trend and rsi_values[-1] < 35:
                reversal_detected = True
                reversal_direction = 'UP'
                reversal_in = 3  # Estimate 3 candles
            
            # Bearish divergence: Price making higher highs, RSI making lower highs
            price_trend_up = closes[-1] > closes[-10]
            rsi_trend_down = rsi_values[-1] < rsi_values[-10]
            
            if price_trend_up and rsi_trend_down and rsi_values[-1] > 65:
                reversal_detected = True
                reversal_direction = 'DOWN'
                reversal_in = 3
        
        # Volume analysis
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
        recent_volume = np.mean(volumes[-3:]) if len(volumes) >= 3 else 0
        
        volume_exhaustion = recent_volume < avg_volume * 0.6  # Low volume = exhaustion
        
        if reversal_detected:
            direction = reversal_direction
            confidence = 0.75
            reasoning = f"Reversal detected: {'Bullish' if direction == 'UP' else 'Bearish'} divergence"
            
            if volume_exhaustion:
                confidence = min(confidence + 0.1, 1.0)
                reasoning += " + Volume exhaustion"
        else:
            # No reversal - predict continuation
            ma_20 = np.mean(closes[-20:])
            current_price = closes[-1]
            
            if current_price > ma_20:
                direction = 'UP'
                confidence = 0.60
                reasoning = 'Continuation: Price above MA20'
            elif current_price < ma_20:
                direction = 'DOWN'
                confidence = 0.60
                reasoning = 'Continuation: Price below MA20'
            else:
                direction = 'NEUTRAL'
                confidence = 0.5
                reasoning = 'No clear medium-term direction'
        
        atr = self._calculate_atr(highs, lows, closes, period=14)
        expected_pips = atr * 1.5 if direction == 'UP' else -atr * 1.5 if direction == 'DOWN' else 0.0
        
        return HorizonPrediction(
            direction=direction,
            confidence=confidence,
            expected_pips=expected_pips,
            reversal_detected=reversal_detected,
            reversal_in_candles=reversal_in,
            reasoning=reasoning
        )
    
    def _determine_consensus(self, immediate: HorizonPrediction, 
                           short_term: HorizonPrediction,
                           medium_term: HorizonPrediction) -> MultiHorizonResult:
        """
        Combine all horizon predictions into a consensus decision.
        
        Weighting:
        - Immediate: 40% (most important for quick decisions)
        - Short-term: 35% (trend confirmation)
        - Medium-term: 25% (reversal warning)
        """
        # Convert to numeric scores
        def direction_to_score(direction: str) -> float:
            return 1.0 if direction == 'UP' else -1.0 if direction == 'DOWN' else 0.0
        
        immediate_score = direction_to_score(immediate.direction) * immediate.confidence * 0.40
        short_score = direction_to_score(short_term.direction) * short_term.confidence * 0.35
        medium_score = direction_to_score(medium_term.direction) * medium_term.confidence * 0.25
        
        total_score = immediate_score + short_score + medium_score
        
        # Determine consensus direction
        if total_score > 0.3:
            consensus_direction = 'UP'
        elif total_score < -0.3:
            consensus_direction = 'DOWN'
        else:
            consensus_direction = 'NEUTRAL'
        
        consensus_confidence = min(abs(total_score), 1.0)
        
        # Determine optimal action
        should_wait = False
        optimal_action = 'TRADE_NOW'
        reasoning_parts = []
        
        # Check for reversal warning
        if medium_term.reversal_detected:
            should_wait = True
            reversal_in = medium_term.reversal_in_candles or 5
            
            if reversal_in <= 2:
                optimal_action = 'WAIT'
                reasoning_parts.append(f"Reversal in {reversal_in} candles - WAIT")
            elif reversal_in <= 5:
                optimal_action = 'HEDGE_NOW'
                reasoning_parts.append(f"Reversal in {reversal_in} candles - Hedge to protect")
        
        # Build reasoning
        if immediate.confidence > 0.7:
            reasoning_parts.append(f"Immediate: {immediate.direction} ({immediate.confidence:.0%})")
        if short_term.confidence > 0.6:
            reasoning_parts.append(f"Short: {short_term.direction} ({short_term.confidence:.0%})")
        if medium_term.confidence > 0.6:
            reasoning_parts.append(f"Medium: {medium_term.direction} ({medium_term.confidence:.0%})")
        
        reasoning = ' | '.join(reasoning_parts) if reasoning_parts else 'Mixed signals across horizons'
        
        return MultiHorizonResult(
            immediate=immediate,
            short_term=short_term,
            medium_term=medium_term,
            consensus_direction=consensus_direction,
            consensus_confidence=consensus_confidence,
            should_wait_for_reversal=should_wait,
            optimal_action=optimal_action,
            reasoning=reasoning
        )
    
    def _create_neutral_result(self, reason: str) -> MultiHorizonResult:
        """Create a neutral result when prediction is not possible."""
        neutral_pred = HorizonPrediction(
            direction='NEUTRAL',
            confidence=0.5,
            expected_pips=0.0,
            reversal_detected=False,
            reversal_in_candles=None,
            reasoning=reason
        )
        
        return MultiHorizonResult(
            immediate=neutral_pred,
            short_term=neutral_pred,
            medium_term=neutral_pred,
            consensus_direction='NEUTRAL',
            consensus_confidence=0.5,
            should_wait_for_reversal=False,
            optimal_action='TRADE_NOW',
            reasoning=reason
        )
    
    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = np.diff(closes[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, highs: List[float], lows: List[float], 
                       closes: List[float], period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(highs) < period + 1:
            return 10.0  # Default fallback
        
        tr_list = []
        for i in range(1, len(highs)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
            tr_list.append(tr)
        
        atr = np.mean(tr_list[-period:])
        return atr


# Singleton instance
_predictor_instance = None

def get_multi_horizon_predictor() -> MultiHorizonPredictor:
    """Get or create the singleton MultiHorizonPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MultiHorizonPredictor()
    return _predictor_instance
