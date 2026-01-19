import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple

logger = logging.getLogger("HedgeIntel")

@dataclass
class HedgeDecision:
    should_hedge: bool
    reason: str
    suggested_lot: float = 0.0
    wait_time: float = 0.0

class HedgeIntelligence:
    """
    The 'Oracle' for Hedging Decisions.
    Uses Regime Detection, Volatility Analysis, and Micro-Structure filters
    to decide EXACTLY when to place a recovery trade.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.RSI_OVERBOUGHT = 70
        self.RSI_OVERSOLD = 30
        self.STRONG_TREND_ADX = 25
        self.CHAOS_THRESHOLD = 3.0 # 3x Normal Volatility

    def evaluate_hedge_opportunity(self, 
                                 market_data: Dict[str, Any], 
                                 position_type: str, 
                                 current_drawdown: float,
                                 regime: str) -> HedgeDecision:
        """
        Determines if a hedge should be placed NOW or DELAYED.
        """
        # 1. CRITICAL: CHAOS LOCK
        # If volatility is extreme (News/Crash), DO NOT ADD RISK.
        volatility_ratio = market_data.get('volatility_ratio', 1.0)
        if volatility_ratio is None or not isinstance(volatility_ratio, (int, float)):
            volatility_ratio = 1.0
        if volatility_ratio > self.CHAOS_THRESHOLD:
            return HedgeDecision(False, f"CHAOS_LOCK (Vol {volatility_ratio:.1f}x)")

        # 2. MOMENTUM EXHAUSTION (RSI + Bollinger Bands)
        # Only enforce strict exhaustion checks if volatility is elevated.
        # In calm markets, standard grid logic is fine.
        rsi = market_data.get('rsi', 50)
        if rsi is None or not isinstance(rsi, (int, float)):
            rsi = 50
        close_price = market_data.get('close', 0.0)
        bb_upper = market_data.get('bb_upper', None)
        bb_lower = market_data.get('bb_lower', None)
        
        if volatility_ratio > 1.5:
            if position_type == 'buy':
                # We are Long, need to Buy more.
                # Wait if RSI is not oversold AND Price is above Lower Band
                # This prevents "catching a falling knife"
                 if rsi > 35 and (bb_lower is not None and close_price > bb_lower):
                     return HedgeDecision(False, f"WAIT_FOR_EXHAUSTION (Vol {volatility_ratio:.1f}x, RSI {rsi:.1f})")
            
            elif position_type == 'sell':
                # We are Short, need to Sell more.
                # Wait if RSI is not overbought AND Price is below Upper Band
                # This prevents selling into a rocket
                if rsi < 65 and (bb_upper is not None and close_price < bb_upper):
                    return HedgeDecision(False, f"WAIT_FOR_EXHAUSTION (Vol {volatility_ratio:.1f}x, RSI {rsi:.1f})")

        # 3. REGIME FILTER (Secondary Check)
        # In a strong trend, don't hedge against it blindly.
        adx = market_data.get('adx', 0)
        if adx is None or not isinstance(adx, (int, float)):
            adx = 0
        
        if regime == "TREND":
            if position_type == 'buy' and rsi > 40 and adx > self.STRONG_TREND_ADX:
                # We are Long, Trend is Down (implied by need to hedge), RSI not oversold.
                return HedgeDecision(False, f"WAIT_FOR_BOTTOM (Trend Strong, RSI {rsi})")
            
            if position_type == 'sell' and rsi < 60 and adx > self.STRONG_TREND_ADX:
                # We are Short, Trend is Up, RSI not overbought.
                return HedgeDecision(False, f"WAIT_FOR_TOP (Trend Strong, RSI {rsi})")

        # 3. MICRO-STRUCTURE FILTER (The "Sniper Wait")
        # Check the current candle color/momentum.
        # If we want to BUY, but the last tick was a massive drop, wait 10s.
        # (This requires tick velocity from market_data, simplified here)
        
        # 4. SMART GRID SIZING
        # If we pass filters, calculate the lot size.
        # In high volatility, we might reduce lot size to survive longer.
        
        return HedgeDecision(True, "SIGNAL_VALID")

    def calculate_dynamic_grid_step(self, base_step: float, market_data: Dict[str, Any]) -> float:
        """
        Expands the grid dynamically based on volatility.
        Prevents stacking trades during violent moves.
        """
        current_atr = market_data.get('atr', 0.0)
        avg_atr = market_data.get('avg_atr', 0.0)
        
        if not avg_atr or avg_atr == 0:
            return base_step

        volatility_ratio = current_atr / avg_atr
        
        # If volatility is 2x normal, grid step becomes 2x wider.
        # Cap at 3x to avoid waiting forever.
        multiplier = min(max(1.0, volatility_ratio), 3.0)
        
        dynamic_step = base_step * multiplier
        return round(dynamic_step, 2)

    def find_shedding_opportunity(self, positions: list) -> list:
        """
        Identifies a 'Winner' and a 'Loser' that can be closed together
        to free up margin with near-zero impact on balance.
        """
        if len(positions) < 2:
            return None

        # Sort: Highest Profit first
        sorted_pos = sorted(positions, key=lambda x: x['profit'], reverse=True)
        winner = sorted_pos[0]
        loser = sorted_pos[-1]

        # Logic: If Winner Profit covers Loser Loss (or close to it)
        if winner['profit'] > 0 and loser['profit'] < 0:
            net = winner['profit'] + loser['profit']
            # Accept a tiny loss ($5) to reduce risk/exposure
            if net > -5.0: 
                return [winner, loser]
        
        return None
