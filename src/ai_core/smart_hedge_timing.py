"""
Smart Hedge Timing - Intelligent hedge placement using multi-horizon predictions

This module decides WHEN to hedge based on predictions rather than just
reacting to zone breaches. Prevents "hedging at the bottom" by waiting
for reversals when predicted.
"""

import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger("SmartHedgeTiming")


@dataclass
class HedgeDecision:
    """Decision on whether and how to hedge."""
    should_hedge: bool
    timing: str  # 'NOW', 'WAIT_1_CANDLE', 'WAIT_2_CANDLES', 'WAIT_FOR_REVERSAL', 'NO_HEDGE'
    direction: Optional[str]  # 'BUY' or 'SELL'
    confidence: float
    size_multiplier: float  # 0.5 to 1.5 (adjust hedge size based on confidence)
    reasoning: str
    expected_outcome: Dict[str, float]  # Expected P&L for different scenarios


class SmartHedgeTiming:
    """
    Intelligent hedge timing using multi-horizon predictions.
    
    Key Intelligence:
    1. Don't hedge at the bottom (wait for reversal if predicted)
    2. Hedge aggressively if continuation predicted
    3. Use partial hedges for uncertain scenarios
    """
    
    def __init__(self, multi_horizon_predictor):
        self.predictor = multi_horizon_predictor
        logger.info("SmartHedgeTiming initialized")
    
    def should_hedge_now(self, position: Dict, market_data: Dict, 
                        candles: List[Dict]) -> HedgeDecision:
        """
        Determine if and when to hedge a position.
        
        Args:
            position: Current position data
            market_data: Current market state
            candles: Recent candle history
            
        Returns:
            HedgeDecision with timing and sizing recommendations
        """
        # Get multi-horizon prediction
        tick = market_data.get('tick', {})
        prediction = self.predictor.predict_all_horizons(candles, tick)
        
        # Analyze position
        position_type = position.get('type', 0)  # 0=BUY, 1=SELL
        position_direction = 'UP' if position_type == 0 else 'DOWN'
        current_pnl = position.get('profit', 0.0)
        is_losing = current_pnl < 0
        
        # Get predictions
        immediate = prediction.immediate
        short_term = prediction.short_term
        medium_term = prediction.medium_term
        
        # === DECISION LOGIC ===
        
        # Scenario 1: Position LOSING + Reversal predicted soon
        if is_losing and medium_term.reversal_detected:
            reversal_in = medium_term.reversal_in_candles or 5
            
            # Check if reversal favors our position
            reversal_helps = (
                (position_direction == 'UP' and medium_term.direction == 'UP') or
                (position_direction == 'DOWN' and medium_term.direction == 'DOWN')
            )
            
            if reversal_helps and reversal_in <= 3:
                # Reversal will save us - WAIT
                return HedgeDecision(
                    should_hedge=False,
                    timing='WAIT_FOR_REVERSAL',
                    direction=None,
                    confidence=medium_term.confidence,
                    size_multiplier=0.0,
                    reasoning=f"Reversal in {reversal_in} candles will help position - WAIT",
                    expected_outcome=self._calculate_outcomes(position, prediction, 'WAIT')
                )
        
        # Scenario 2: Position LOSING + Immediate prediction confirms further loss
        if is_losing:
            # Check if immediate prediction says it will get worse
            gets_worse = (
                (position_direction == 'UP' and immediate.direction == 'DOWN' and immediate.confidence > 0.70) or
                (position_direction == 'DOWN' and immediate.direction == 'UP' and immediate.confidence > 0.70)
            )
            
            if gets_worse:
                # Hedge NOW - it's getting worse
                hedge_direction = 'SELL' if position_direction == 'UP' else 'BUY'
                
                return HedgeDecision(
                    should_hedge=True,
                    timing='NOW',
                    direction=hedge_direction,
                    confidence=immediate.confidence,
                    size_multiplier=1.0,  # Full hedge
                    reasoning=f"Position losing + {immediate.confidence:.0%} confident it will worsen - HEDGE NOW",
                    expected_outcome=self._calculate_outcomes(position, prediction, 'HEDGE_NOW')
                )
        
        # Scenario 3: Position WINNING but reversal coming
        if not is_losing and medium_term.reversal_detected:
            reversal_in = medium_term.reversal_in_candles or 5
            
            # Check if reversal threatens our profit
            reversal_threatens = (
                (position_direction == 'UP' and medium_term.direction == 'DOWN') or
                (position_direction == 'DOWN' and medium_term.direction == 'UP')
            )
            
            if reversal_threatens and reversal_in <= 2:
                # Protect profit - hedge NOW
                hedge_direction = 'SELL' if position_direction == 'UP' else 'BUY'
                
                return HedgeDecision(
                    should_hedge=True,
                    timing='NOW',
                    direction=hedge_direction,
                    confidence=medium_term.confidence,
                    size_multiplier=0.8,  # Partial hedge to protect profit
                    reasoning=f"Reversal in {reversal_in} candles threatens profit - Partial HEDGE",
                    expected_outcome=self._calculate_outcomes(position, prediction, 'HEDGE_NOW')
                )
            elif reversal_threatens and reversal_in <= 4:
                # Reversal soon - wait 1 candle for confirmation
                hedge_direction = 'SELL' if position_direction == 'UP' else 'BUY'
                
                return HedgeDecision(
                    should_hedge=True,
                    timing='WAIT_1_CANDLE',
                    direction=hedge_direction,
                    confidence=medium_term.confidence,
                    size_multiplier=0.8,
                    reasoning=f"Reversal in {reversal_in} candles - Wait for confirmation",
                    expected_outcome=self._calculate_outcomes(position, prediction, 'WAIT_1')
                )
        
        # Scenario 4: Position LOSING but recovery predicted
        if is_losing:
            # Check if short-term predicts recovery
            recovers = (
                (position_direction == 'UP' and short_term.direction == 'UP' and short_term.confidence > 0.65) or
                (position_direction == 'DOWN' and short_term.direction == 'DOWN' and short_term.confidence > 0.65)
            )
            
            if recovers:
                # Recovery predicted - DON'T hedge
                return HedgeDecision(
                    should_hedge=False,
                    timing='WAIT_FOR_REVERSAL',
                    direction=None,
                    confidence=short_term.confidence,
                    size_multiplier=0.0,
                    reasoning=f"Position losing but {short_term.confidence:.0%} confident recovery in 3 candles - WAIT",
                    expected_outcome=self._calculate_outcomes(position, prediction, 'WAIT')
                )
        
        # Scenario 5: Uncertain - use defensive partial hedge
        if is_losing and abs(current_pnl) > 20:  # Significant loss
            # Uncertain but losing - partial hedge
            hedge_direction = 'SELL' if position_direction == 'UP' else 'BUY'
            
            return HedgeDecision(
                should_hedge=True,
                timing='NOW',
                direction=hedge_direction,
                confidence=0.60,
                size_multiplier=0.5,  # Half hedge (defensive)
                reasoning="Uncertain prediction but significant loss - Defensive partial hedge",
                expected_outcome=self._calculate_outcomes(position, prediction, 'PARTIAL_HEDGE')
            )
        
        # Default: No hedge needed
        return HedgeDecision(
            should_hedge=False,
            timing='NO_HEDGE',
            direction=None,
            confidence=0.5,
            size_multiplier=0.0,
            reasoning="No clear hedge signal - Continue monitoring",
            expected_outcome=self._calculate_outcomes(position, prediction, 'NO_HEDGE')
        )
    
    def _calculate_outcomes(self, position: Dict, prediction, scenario: str) -> Dict[str, float]:
        """
        Calculate expected P&L for different scenarios.
        
        Args:
            position: Position data
            prediction: Multi-horizon prediction
            scenario: 'HEDGE_NOW', 'WAIT', 'PARTIAL_HEDGE', 'NO_HEDGE'
            
        Returns:
            Dict with expected outcomes
        """
        current_pnl = position.get('profit', 0.0)
        position_size = position.get('volume', 0.1)
        
        # Estimate expected move from immediate prediction
        immediate_pips = prediction.immediate.expected_pips
        short_pips = prediction.short_term.expected_pips
        
        # Simple P&L estimation (rough)
        pip_value = 10.0  # Approximate for XAUUSD
        
        if scenario == 'HEDGE_NOW':
            # Full hedge locks in current P&L (approximately)
            expected = current_pnl - (abs(immediate_pips) * pip_value * 0.1)  # Small slippage
        elif scenario == 'PARTIAL_HEDGE':
            # Partial hedge reduces exposure
            expected = current_pnl + (immediate_pips * pip_value * position_size * 0.5)
        elif scenario == 'WAIT' or scenario == 'WAIT_1':
            # Wait for reversal - use short-term prediction
            expected = current_pnl + (short_pips * pip_value * position_size)
        else:  # NO_HEDGE
            # No hedge - full exposure to immediate move
            expected = current_pnl + (immediate_pips * pip_value * position_size)
        
        return {
            'expected_pnl': round(expected, 2),
            'current_pnl': round(current_pnl, 2),
            'potential_change': round(expected - current_pnl, 2)
        }


# Singleton instance
_smart_hedge_instance = None

def get_smart_hedge_timing(multi_horizon_predictor) -> SmartHedgeTiming:
    """Get or create the singleton SmartHedgeTiming instance."""
    global _smart_hedge_instance
    if _smart_hedge_instance is None:
        _smart_hedge_instance = SmartHedgeTiming(multi_horizon_predictor)
    return _smart_hedge_instance
