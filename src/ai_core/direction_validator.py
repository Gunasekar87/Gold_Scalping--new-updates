"""
Direction Validator - 7-Factor Intelligence for Trade Direction Accuracy

This module provides rigorous validation of trade direction to prevent
"Buy when market goes down" scenarios. It acts as a guidance system that
adjusts confidence and can invert signals when necessary.
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("DirectionValidator")


@dataclass
class ValidationResult:
    """Result of direction validation."""
    score: float  # 0.0 to 1.0 (Alignment Score)
    confidence_multiplier: float  # 0.5 to 1.5
    should_invert: bool
    passed_factors: int # Kept for compatibility
    total_factors: int # Kept for compatibility
    failed_factors: list # Now used for "Opposing Factors"
    reasoning: str


class DirectionValidator:
    """
    7-Factor 'Market Analyst' for Strategic Directional Advice.
    
    Instead of 'Blocking', this module analyzes the market to determine the
    dominant bias (Bullish vs Bearish) and advises the Trading Engine.
    
    It calculates a Weighted Bias Score (-1.0 to +1.0).
    """
    
    def __init__(self):
        self.validation_history = []
        logger.info("DirectionValidator initialized as Market Analyst")
    
    def validate_direction(self, 
                          direction: str,  # 'BUY' or 'SELL'
                          market_data: Dict,
                          worker_confidence: float) -> ValidationResult:
        """
        Analyze trade direction against market bias.
        """
        # 1. Calculate Biases for each Factor (-1.0 Sell, 0.0 Neutral, 1.0 Buy)
        biases = {}
        
        try:
            # [INTELLIGENCE FIX] Analyze Trend & Regime FIRST to establish context
            # We need to know if we are Trending or Ranging to interpret Momentum/Levels correctly.
            biases['trend'] = self._analyze_trend(market_data)
            trend_score = biases['trend']
            
            # Extract Regime Name safely
            regime_obj = market_data.get('regime', 'RANGE')
            regime_name = regime_obj.name if hasattr(regime_obj, 'name') else str(regime_obj)
            regime_name = regime_name.upper()

            # Pass context to other analyzers
            biases['momentum'] = self._analyze_momentum(market_data, regime_name, trend_score)
            biases['levels'] = self._analyze_levels(market_data, regime_name, trend_score)
            biases['mtf'] = self._analyze_mtf(market_data)
            biases['flow'] = self._analyze_flow(market_data)
            biases['ai'] = self._analyze_ai(direction, market_data)
            biases['trajectory'] = self._analyze_trajectory(market_data)
        except Exception as e:
            logger.error(f"[ANALYST] Analysis failed: {e}", exc_info=True)
            return ValidationResult(0.5, 1.0, False, 0, 0, [], "Analysis Error (Defaulting to Neutral)")

        # 2. Apply "Council of Minds" Weights (Physics & Flow Priority)
        weights = {
            'trajectory': 0.25, # High Priority: Physics (Next 5 candles)
            'flow': 0.15,       # High Priority: Micro-Structure (Order Flow)
            'trend': 0.15,      # Medium: Context
            'levels': 0.15,     # Medium: Structure
            'momentum': 0.10,   # Low: Lagging
            'ai': 0.10,         # Low: Oracle Class Label (Trajectory is better)
            'mtf': 0.10         # Low: Confirmation
        }
        
        total_bias = 0.0
        total_weight = 0.0
        details = []
        
        for factor, bias in biases.items():
            w = weights.get(factor, 0.0)
            if w > 0:
                total_bias += bias * w
                total_weight += w
            
                if abs(bias) > 0.3:
                    sentiment = "BULLISH" if bias > 0 else "BEARISH"
                    details.append(f"{factor.upper()}: {sentiment}({bias:.1f})")

        # 3. Normalize (-1.0 to 1.0)
        market_bias = total_bias / total_weight if total_weight > 0 else 0.0

        # [INTELLIGENCE] Strict Trend Guard: Prevent fighting strong trends
        # Only applies in RANGE or UNKNOWN regimes where we might mistakenly fade a new trend.
        # If we are ALREADY in TREND regime, the analytics above handle it.
        
        rsi_val = float(market_data.get('rsi', 50.0))
        
        # Guard against Selling in Strong Uptrend (if analytics missed it)
        if trend_score > 0.6 and direction == "SELL" and regime_name != "TREND":
            if rsi_val < 80: 
                market_bias += 0.5 
                details.append(f"TREND_GUARD: BLOCKED SELL (Uptrend+RSI{rsi_val:.1f})")

        # Guard against Buying in Strong Downtrend
        if trend_score < -0.6 and direction == "BUY" and regime_name != "TREND":
            if rsi_val > 20: 
                market_bias -= 0.5 
                details.append(f"TREND_GUARD: BLOCKED BUY (Downtrend+RSI{rsi_val:.1f})")

        
        # 4. Compare Market Bias to Proposed Direction
        proposed_numeric = 1.0 if direction == 'BUY' else -1.0
        alignment = market_bias * proposed_numeric
        score = (alignment + 1.0) / 2.0
        
        # 5. Formulate Advice
        confidence_multiplier = 1.0
        should_invert = False
        opposing_factors = [k for k, v in biases.items() if (v * proposed_numeric) < -0.2]
        supporting_factors = [k for k, v in biases.items() if (v * proposed_numeric) > 0.2]
        
        narrative = f"Market Bias: {market_bias:.2f}."
        
        if score > 0.65:
            confidence_multiplier = 1.2
            narrative = f"Strong {direction} Alignment. bias={market_bias:.2f}. Supports: {','.join(supporting_factors[:3])}"
        elif score > 0.40:
            confidence_multiplier = 1.0
            narrative = f"Neutral/Moderate Alignment. bias={market_bias:.2f}."
        elif score > 0.25:
            confidence_multiplier = 0.7
            narrative = f"Weak Alignment (Caution). Market leans opposite. Opposes: {','.join(opposing_factors[:3])}"
        else:
            should_invert = True
            confidence_multiplier = 0.5
            narrative = f"CRITICAL DIVERGENCE! Market strongly opposes {direction}. Suggest INVERSION."

        # [COUNCIL OF MINDS] ABSOLUTE VETO POWERS
        # If Physics (Trajectory) or Flow (Pressure) strongly oppose, we BLOCK.
        # This overrides the "Invert" suggestion because inverting into a Chop is bad.
        
        # Check Trajectory Veto
        traj_bias = biases.get('trajectory', 0.0)
        if (direction == "BUY" and traj_bias < -0.5) or (direction == "SELL" and traj_bias > 0.5):
            confidence_multiplier = 0.0 # Strict Block
            narrative = f"[COUNCIL VETO] Physics Veto. Trajectory predicts opposite move ({traj_bias:.2f})."
            should_invert = False # Don't invert, just BLOCK

        # Check Flow Veto
        flow_bias = biases.get('flow', 0.0)
        if (direction == "BUY" and flow_bias < -0.6) or (direction == "SELL" and flow_bias > 0.6):
             confidence_multiplier = 0.0 # Strict Block
             narrative = f"[COUNCIL VETO] Flow Veto. Order Flow strongly opposes ({flow_bias:.2f})."
             should_invert = False

        return ValidationResult(
            score=score,
            confidence_multiplier=confidence_multiplier,
            should_invert=should_invert,
            passed_factors=len(supporting_factors),
            total_factors=len(weights),
            failed_factors=opposing_factors, 
            reasoning=narrative
        )

    # --- ANALYTIC METHODS (Return Bias: -1.0 Sell, 1.0 Buy) ---

    def _analyze_trend(self, market_data: Dict) -> float:
        """Analyze Trend Direction and Strength."""
        trend_str = str(market_data.get('trend', 'NEUTRAL')).upper()
        regime_obj = market_data.get('regime', '')
        regime_name = regime_obj.name if hasattr(regime_obj, 'name') else str(regime_obj)
        regime_name = regime_name.upper()
        
        score = 0.0
        if 'UP' in regime_name or 'BULL' in trend_str: score += 0.8
        elif 'DOWN' in regime_name or 'BEAR' in trend_str: score -= 0.8
        
        return score

    def _analyze_momentum(self, market_data: Dict, regime_name: str, trend_score: float) -> float:
        """
        Analyze Momentum (RSI + MACD).
        Context-Aware: Interprets RSI differently in Trends vs Ranges.
        """
        rsi = market_data.get('rsi', 50)
        macd = market_data.get('macd', {})
        score = 0.0
        
        # RSI Logic
        if "TREND" in regime_name and abs(trend_score) > 0.4:
            # TREND REGIME: Extremes confirm strength (Momentum)
            if rsi > 70 and trend_score > 0: 
                score += 0.5 # Bullish Momentum (don't fade!)
            elif rsi < 30 and trend_score < 0: 
                score -= 0.5 # Bearish Momentum (don't fade!)
            elif rsi > 80: # Extreme Overbought -> Warning even in trend
                score -= 0.2 
            elif rsi < 20: # Extreme Oversold -> Warning even in trend
                score += 0.2
            else:
                # Normal RSI in trend check
                if rsi > 55: score += 0.2
                elif rsi < 45: score -= 0.2
                
        else:
            # RANGE/UNKNOWN REGIME: Mean Reversion
            if rsi > 70: score -= 0.8 # Overbought -> Bearish Pressure
            elif rsi < 30: score += 0.8 # Oversold -> Bullish Pressure
            elif rsi > 55: score += 0.3 # Mild Bullish
            elif rsi < 45: score -= 0.3 # Mild Bearish
        
        # MACD
        hist = macd.get('histogram', 0)
        if hist > 0: score += 0.2
        elif hist < 0: score -= 0.2
        
        return max(-1.0, min(1.0, score))

    def _analyze_levels(self, market_data: Dict, regime_name: str, trend_score: float) -> float:
        """
        Analyze proximity to Support/Resistance.
        Context-Aware: Trends break levels; Ranges respect them.
        """
        price = market_data.get('current_price', 0)
        if not price: return 0.0
        
        support = market_data.get('support', price - 10)
        resistance = market_data.get('resistance', price + 10)
        
        dist_supp = abs(price - support)
        dist_res = abs(price - resistance)
        total = dist_supp + dist_res
        if total == 0: return 0.0
        
        # Base Proximity Score (-1 at Res, +1 at Sup)
        proximity = (dist_res - dist_supp) / total
        
        # Context Adjustment
        if "TREND" in regime_name and abs(trend_score) > 0.5:
            # TREND REGIME: Levels are meant to be broken
            if trend_score > 0: # Uptrend
                # If at Resistance (proximity -> -1), ignore it or treat as breakout
                if proximity < -0.5: return 0.2 # Breakout potential
                if proximity > 0.5: return 0.8 # Support hold in uptrend is good
            else: # Downtrend
                # If at Support (proximity -> +1), ignore it or treat as breakdown
                if proximity > 0.5: return -0.2 # Breakdown potential
                if proximity < -0.5: return -0.8 # Resistance hold in downtrend is good
                
            return proximity * 0.5 # Weaken level respect in trends
            
        else:
            # RANGE REGIME: Respect Levels (Mean Reversion)
            return proximity

    def _analyze_mtf(self, market_data: Dict) -> float:
        """Analyze Multi-Timeframe Alignment."""
        m1 = str(market_data.get('m1_trend', '')).upper()
        m5 = str(market_data.get('m5_trend', '')).upper()
        m15 = str(market_data.get('m15_trend', '')).upper()
        
        score = 0.0
        for t in [m1, m5, m15]:
            if 'UP' in t or 'BULL' in t: score += 0.33
            elif 'DOWN' in t or 'BEAR' in t: score -= 0.33
            
        return score

    def _analyze_ai(self, direction: str, market_data: Dict) -> float:
        """Analyze AI Consensus Bias."""
        oracle_pred = str(market_data.get('oracle_prediction', 'NEUTRAL')).upper()
        if oracle_pred in ('UP', 'BUY'): return 0.8
        if oracle_pred in ('DOWN', 'SELL'): return -0.8
        return 0.0

    def _analyze_trajectory(self, market_data: Dict) -> float:
        """Analyze predicted path of next few candles."""
        traj = market_data.get('trajectory', [])
        current_price = market_data.get('current_price', 0)
        
        if not traj or not current_price or len(traj) == 0:
            return 0.0
            
        end_price = traj[-1]
        pct_change = (end_price - current_price) / current_price if current_price > 0 else 0
        
        if abs(pct_change) < 0.0002: # Ignore < 0.02% noise (approx $0.50 on Gold)
            return 0.0
        elif pct_change > 0.001: # > 0.1% move (approx $2.50) is Strong Bullish
            return 1.0
        elif pct_change < -0.001: # < -0.1% move (approx -$2.50) is Strong Bearish
            return -1.0
        else:
            # Scale linearly: 0.0002 to 0.001 maps to Score 0.2 to 1.0
            # factor = 1000? 0.001 * 1000 = 1.0
            return max(-1.0, min(1.0, pct_change * 1000))

    def _analyze_flow(self, market_data: Dict) -> float:
        """
        Analyze Order Flow / Tick Pressure.
        Returns: -1.0 (Bearish Pressure) to 1.0 (Bullish Pressure)
        """
        pressure = market_data.get('pressure', {})
        if not pressure: return 0.0
        
        dominance = pressure.get('dominance', 'NEUTRAL')
        intensity = pressure.get('intensity', 'LOW')
        
        score = 0.0
        if dominance == 'BUY': score = 0.5
        elif dominance == 'SELL': score = -0.5
        
        if intensity == 'HIGH': score *= 1.5
        elif intensity == 'EXTREME': score *= 2.0
        
        return max(-1.0, min(1.0, score))

# Singleton instance
_validator_instance = None

def get_direction_validator() -> DirectionValidator:
    """Get or create the singleton DirectionValidator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = DirectionValidator()
    return _validator_instance
