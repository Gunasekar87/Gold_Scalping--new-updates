"""
Wick Intelligence Module - Prevents trading at dangerous wick extremes

This module analyzes candle wicks (shadows) to identify rejection zones
and prevents the bot from placing orders at these high-risk areas.

Key Concepts:
- Upper wick = Buyers rejected, sellers took control
- Lower wick = Sellers rejected, buyers took control
- Trading at wick extremes = High probability of reversal

Author: AETHER Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import centralized configuration
try:
    from src.config.ai_settings import WickIntelligenceConfig
except ImportError:
    # Fallback if config not available
    class WickIntelligenceConfig:
        WICK_THRESHOLD_PCT = 30.0
        PROXIMITY_THRESHOLD_PCT = 20.0
        NUM_CANDLES_TO_ANALYZE = 5

logger = logging.getLogger("WickIntelligence")


@dataclass
class WickAnalysis:
    """Analysis of candle wick structure."""
    has_significant_upper_wick: bool
    has_significant_lower_wick: bool
    upper_wick_pct: float  # % of total candle range
    lower_wick_pct: float  # % of total candle range
    is_at_upper_wick: bool  # Current price near upper wick
    is_at_lower_wick: bool  # Current price near lower wick
    rejection_zone: Optional[str]  # "UPPER", "LOWER", or None
    safe_to_buy: bool
    safe_to_sell: bool
    reasoning: str


class WickIntelligence:
    """
    Analyzes candle wicks to prevent trading at rejection zones.
    
    Institutional traders use wicks to trap retail traders:
    - Upper wick = Stop hunt above, then price drops
    - Lower wick = Stop hunt below, then price rises
    
    We avoid these traps by detecting wick patterns.
    """
    
    def __init__(
        self,
        wick_threshold_pct: float = None,
        proximity_threshold_pct: float = None
    ):
        """
        Initialize wick intelligence.
        
        Args:
            wick_threshold_pct: Minimum wick size (% of candle) to be significant
            proximity_threshold_pct: How close to wick extreme to trigger warning
        """
        # Use config values if not provided
        self.wick_threshold_pct = (
            wick_threshold_pct if wick_threshold_pct is not None 
            else WickIntelligenceConfig.WICK_THRESHOLD_PCT
        )
        self.proximity_threshold_pct = (
            proximity_threshold_pct if proximity_threshold_pct is not None
            else WickIntelligenceConfig.PROXIMITY_THRESHOLD_PCT
        )
        
        logger.info(
            f"WickIntelligence initialized: "
            f"wick_threshold={self.wick_threshold_pct}%, "
            f"proximity={self.proximity_threshold_pct}%"
        )
    
    def analyze_current_position(
        self,
        current_price: float,
        recent_candles: List[Dict],
        num_candles: int = 5
    ) -> WickAnalysis:
        """
        Analyze if current price is at a dangerous wick extreme.
        
        Args:
            current_price: Current market price
            recent_candles: List of recent candle dicts (OHLC)
            num_candles: Number of recent candles to analyze
            
        Returns:
            WickAnalysis with safety assessment
        """
        if not recent_candles or len(recent_candles) < 1:
            return WickAnalysis(
                has_significant_upper_wick=False,
                has_significant_lower_wick=False,
                upper_wick_pct=0.0,
                lower_wick_pct=0.0,
                is_at_upper_wick=False,
                is_at_lower_wick=False,
                rejection_zone=None,
                safe_to_buy=True,
                safe_to_sell=True,
                reasoning="No candle data available"
            )
        
        # Analyze last N candles for wick patterns
        candles_to_check = recent_candles[-num_candles:]
        
        # Find the most significant wick in recent candles
        max_upper_wick_pct = 0.0
        max_lower_wick_pct = 0.0
        highest_high = max(c.get('high', 0) for c in candles_to_check)
        lowest_low = min(c.get('low', 0) for c in candles_to_check)
        
        for candle in candles_to_check:
            analysis = self._analyze_single_candle(candle)
            max_upper_wick_pct = max(max_upper_wick_pct, analysis['upper_wick_pct'])
            max_lower_wick_pct = max(max_lower_wick_pct, analysis['lower_wick_pct'])
        
        # Check if current price is near wick extremes
        total_range = highest_high - lowest_low
        if total_range <= 0:
            total_range = 0.0001  # Avoid division by zero
        
        # Distance from extremes as percentage
        dist_from_high = ((highest_high - current_price) / total_range) * 100
        dist_from_low = ((current_price - lowest_low) / total_range) * 100
        
        # Determine if we're at a wick extreme
        is_at_upper_wick = dist_from_high < self.proximity_threshold_pct
        is_at_lower_wick = dist_from_low < self.proximity_threshold_pct
        
        # Determine if wicks are significant
        has_significant_upper = max_upper_wick_pct > self.wick_threshold_pct
        has_significant_lower = max_lower_wick_pct > self.wick_threshold_pct
        
        # Determine rejection zone
        rejection_zone = None
        if is_at_upper_wick and has_significant_upper:
            rejection_zone = "UPPER"
        elif is_at_lower_wick and has_significant_lower:
            rejection_zone = "LOWER"
        
        # Safety assessment - ONLY block at EXTREME positions
        safe_to_buy = True
        safe_to_sell = True
        reasoning = "No wick rejection detected"
        
        # REFINED LOGIC: Only block at the EXTREME of wicks
        # Upper wick: Only block BUY if very close to the top (<10%)
        # Lower wick: Only block SELL if very close to the bottom (<10%)
        
        if rejection_zone == "UPPER":
            # At upper wick - check if we're at the EXTREME top
            if dist_from_high < 10.0:  # Within 10% of the absolute high
                # DON'T BUY at the extreme top (price likely to drop)
                safe_to_buy = False
                reasoning = (
                    f"⚠️ UPPER WICK REJECTION: Price at {current_price:.2f} "
                    f"is {dist_from_high:.1f}% from recent high {highest_high:.2f}. "
                    f"Upper wick is {max_upper_wick_pct:.1f}% of candle. "
                    f"Buyers were rejected at extreme - expect downward pressure."
                )
            else:
                # Price is near upper wick but not at extreme - ALLOW trades
                reasoning = (
                    f"✅ Near upper wick ({dist_from_high:.1f}% from high) but not at extreme. "
                    f"Trades allowed."
                )
                
        elif rejection_zone == "LOWER":
            # At lower wick - check if we're at the EXTREME bottom
            if dist_from_low < 10.0:  # Within 10% of the absolute low
                # DON'T SELL at the extreme bottom (price likely to rise)
                safe_to_sell = False
                reasoning = (
                    f"⚠️ LOWER WICK REJECTION: Price at {current_price:.2f} "
                    f"is {dist_from_low:.1f}% from recent low {lowest_low:.2f}. "
                    f"Lower wick is {max_lower_wick_pct:.1f}% of candle. "
                    f"Sellers were rejected at extreme - expect upward pressure."
                )
            else:
                # Price is near lower wick but not at extreme - ALLOW trades
                reasoning = (
                    f"✅ Near lower wick ({dist_from_low:.1f}% from low) but not at extreme. "
                    f"Trades allowed."
                )
        
        return WickAnalysis(
            has_significant_upper_wick=has_significant_upper,
            has_significant_lower_wick=has_significant_lower,
            upper_wick_pct=max_upper_wick_pct,
            lower_wick_pct=max_lower_wick_pct,
            is_at_upper_wick=is_at_upper_wick,
            is_at_lower_wick=is_at_lower_wick,
            rejection_zone=rejection_zone,
            safe_to_buy=safe_to_buy,
            safe_to_sell=safe_to_sell,
            reasoning=reasoning
        )

    
    def _analyze_single_candle(self, candle: Dict) -> Dict:
        """
        Analyze a single candle's wick structure.
        
        Args:
            candle: Candle dict with OHLC data
            
        Returns:
            Dict with wick analysis
        """
        open_price = float(candle.get('open', 0))
        high = float(candle.get('high', 0))
        low = float(candle.get('low', 0))
        close = float(candle.get('close', 0))
        
        # Determine candle body
        body_top = max(open_price, close)
        body_bottom = min(open_price, close)
        
        # Calculate wick sizes
        upper_wick = high - body_top
        lower_wick = body_bottom - low
        total_range = high - low
        
        if total_range <= 0:
            return {
                'upper_wick_pct': 0.0,
                'lower_wick_pct': 0.0,
                'body_pct': 0.0
            }
        
        # Calculate percentages
        upper_wick_pct = (upper_wick / total_range) * 100
        lower_wick_pct = (lower_wick / total_range) * 100
        body_pct = ((body_top - body_bottom) / total_range) * 100
        
        return {
            'upper_wick_pct': upper_wick_pct,
            'lower_wick_pct': lower_wick_pct,
            'body_pct': body_pct,
            'upper_wick_size': upper_wick,
            'lower_wick_size': lower_wick
        }
    
    def should_block_trade(
        self,
        direction: str,
        current_price: float,
        recent_candles: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Determine if a trade should be blocked due to wick rejection.
        
        Args:
            direction: "BUY" or "SELL"
            current_price: Current market price
            recent_candles: Recent candle data
            
        Returns:
            Tuple of (should_block, reason)
        """
        analysis = self.analyze_current_position(current_price, recent_candles)
        
        if direction == "BUY" and not analysis.safe_to_buy:
            return True, analysis.reasoning
        
        if direction == "SELL" and not analysis.safe_to_sell:
            return True, analysis.reasoning
        
        return False, "Wick analysis passed - safe to trade"
    
    def get_safe_entry_suggestion(
        self,
        direction: str,
        current_price: float,
        recent_candles: List[Dict]
    ) -> Optional[float]:
        """
        Suggest a safer entry price away from wick extremes.
        
        Args:
            direction: "BUY" or "SELL"
            current_price: Current market price
            recent_candles: Recent candle data
            
        Returns:
            Suggested entry price or None if current price is safe
        """
        analysis = self.analyze_current_position(current_price, recent_candles)
        
        if not recent_candles:
            return None
        
        # Get recent range
        candles_to_check = recent_candles[-5:]
        highest_high = max(c.get('high', 0) for c in candles_to_check)
        lowest_low = min(c.get('low', 0) for c in candles_to_check)
        mid_point = (highest_high + lowest_low) / 2
        
        # If at upper wick and want to BUY, suggest waiting for pullback
        if direction == "BUY" and analysis.is_at_upper_wick:
            # Suggest entry at 50% retracement
            return mid_point
        
        # If at lower wick and want to SELL, suggest waiting for bounce
        if direction == "SELL" and analysis.is_at_lower_wick:
            # Suggest entry at 50% retracement
            return mid_point
        
        return None  # Current price is safe
    
    def should_exit_at_wick(
        self,
        position_type: str,
        current_price: float,
        recent_candles: List[Dict],
        profit_pips: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Determine if we should EXIT a position at a wick extreme.
        
        OPPOSITE LOGIC FROM ENTRIES:
        - Upper wick = GOOD place to close BUY (take profit at resistance)
        - Lower wick = GOOD place to close SELL (take profit at support)
        
        Args:
            position_type: "BUY" or "SELL" (current position)
            current_price: Current market price
            recent_candles: Recent candle data
            profit_pips: Current profit in pips (optional, for decision weighting)
            
        Returns:
            Tuple of (should_exit, reason)
        """
        analysis = self.analyze_current_position(current_price, recent_candles)
        
        # If in profit and at a wick extreme, TAKE PROFIT
        if position_type == "BUY" and analysis.is_at_upper_wick and analysis.has_significant_upper_wick:
            # BUY position at upper wick = Perfect exit (resistance)
            confidence = "STRONG" if profit_pips > 10 else "MODERATE"
            return True, (
                f"✅ {confidence} EXIT SIGNAL: BUY position at upper wick resistance. "
                f"Price {current_price:.2f} near high {analysis.rejection_zone}. "
                f"Upper wick {analysis.upper_wick_pct:.1f}% indicates selling pressure. "
                f"TAKE PROFIT NOW before reversal."
            )
        
        if position_type == "SELL" and analysis.is_at_lower_wick and analysis.has_significant_lower_wick:
            # SELL position at lower wick = Perfect exit (support)
            confidence = "STRONG" if profit_pips > 10 else "MODERATE"
            return True, (
                f"✅ {confidence} EXIT SIGNAL: SELL position at lower wick support. "
                f"Price {current_price:.2f} near low {analysis.rejection_zone}. "
                f"Lower wick {analysis.lower_wick_pct:.1f}% indicates buying pressure. "
                f"TAKE PROFIT NOW before reversal."
            )
        
        # If in loss and at opposite wick, consider cutting losses
        if position_type == "BUY" and analysis.is_at_lower_wick and profit_pips < -20:
            # BUY position at lower wick while losing = Might bounce, but risky
            return False, (
                f"⚠️ HOLD: BUY position at lower wick support while losing {profit_pips:.1f} pips. "
                f"Price may bounce from support. Monitor closely."
            )
        
        if position_type == "SELL" and analysis.is_at_upper_wick and profit_pips < -20:
            # SELL position at upper wick while losing = Might drop, but risky
            return False, (
                f"⚠️ HOLD: SELL position at upper wick resistance while losing {profit_pips:.1f} pips. "
                f"Price may drop from resistance. Monitor closely."
            )
        
        return False, "No wick-based exit signal"


# Global singleton instance
_wick_intelligence_instance = None


def get_wick_intelligence() -> WickIntelligence:
    """Get or create the singleton WickIntelligence instance."""
    global _wick_intelligence_instance
    if _wick_intelligence_instance is None:
        _wick_intelligence_instance = WickIntelligence()
    return _wick_intelligence_instance

