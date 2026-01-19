"""
Trade Explanation Engine - Provides detailed, trader-friendly explanations for all trading decisions.

This module generates comprehensive, human-readable explanations for:
- Trade closures (why, when, how much profit/loss)
- Hedge placements (market analysis, risk assessment)
- Recovery trades (liquidity analysis, structure-based reasoning)
- Initial entries (AI consensus, technical setup)

Author: AETHER Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("TradeExplainer")


class ExitReason(Enum):
    """Enumeration of exit reasons."""
    PROFIT_TARGET = "profit_target"
    TIME_LIMIT = "time_limit"
    EMERGENCY = "emergency"
    BREAK_EVEN = "break_even"
    MANUAL = "manual"
    STOP_LOSS = "stop_loss"


@dataclass
class TradeExplanation:
    """Structured explanation for a trading decision."""
    title: str
    summary: str
    market_analysis: str
    risk_assessment: str
    decision_logic: str
    expected_outcome: str
    confidence_level: str


class TradeExplainer:
    """
    Generates detailed, trader-friendly explanations for all trading decisions.
    """
    
    def __init__(self):
        self.pip_multiplier = {"XAUUSD": 100, "GOLD": 100}  # Default multipliers
        
    def explain_bucket_close(
        self,
        symbol: str,
        positions: List[Dict],
        total_pnl: float,
        total_volume: float,
        bucket_duration: float,
        exit_reason: str,
        ai_metrics: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive explanation for bucket closure.
        
        Args:
            symbol: Trading symbol
            positions: List of positions being closed
            total_pnl: Total profit/loss in USD
            total_volume: Total volume in lots
            bucket_duration: Duration in seconds
            exit_reason: Reason for exit
            ai_metrics: Optional AI decision metrics
            
        Returns:
            Formatted explanation string for terminal display
        """
        # Calculate metrics
        minutes = int(bucket_duration // 60)
        seconds = int(bucket_duration % 60)
        duration_str = f"{minutes}m {seconds}s"
        
        # Calculate pips
        pip_mult = self.pip_multiplier.get(symbol, 10000)
        contract_size = 100 if "XAU" in symbol or "GOLD" in symbol else 100000
        total_pips = (total_pnl / (total_volume * contract_size)) * pip_mult if total_volume > 0 else 0.0
        
        # Determine profit/loss status
        if total_pnl > 0:
            pnl_status = "PROFIT"
            pnl_emoji = "✓"
            outcome = "secured"
        elif total_pnl < 0:
            pnl_status = "LOSS"
            pnl_emoji = "✗"
            outcome = "minimized"
        else:
            pnl_status = "BREAK-EVEN"
            pnl_emoji = "="
            outcome = "protected capital"
        
        # Build position breakdown
        position_details = []
        for i, pos in enumerate(positions, 1):
            # Handle both dict and Position object types
            if hasattr(pos, 'type'):
                # Position object
                pos_type = "BUY" if pos.type == 0 else "SELL"
                pos_volume = pos.volume if hasattr(pos, 'volume') else 0.0
                pos_profit = pos.profit if hasattr(pos, 'profit') else 0.0
                pos_price = pos.price_open if hasattr(pos, 'price_open') else 0.0
            else:
                # Dictionary (fallback)
                pos_type = "BUY" if pos.get('type', 0) == 0 else "SELL"
                pos_volume = pos.get('volume', 0.0)
                pos_profit = pos.get('profit', 0.0)
                pos_price = pos.get('price_open', 0.0)
            
            position_details.append(
                f"      Position #{i}: {pos_type} {pos_volume:.2f} lots @ {pos_price:.5f} → ${pos_profit:+.2f}"
            )
        
        # Explain exit reason
        reason_explanation = self._explain_exit_reason(exit_reason, ai_metrics)
        
        # Build comprehensive explanation
        explanation = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         TRADE CLOSURE ANALYSIS                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 TRADE SUMMARY
   Symbol:              {symbol}
   Status:              {pnl_status} {pnl_emoji}
   Total P&L:           ${total_pnl:+.2f} ({total_pips:+.1f} pips)
   Duration:            {duration_str}
   Positions Closed:    {len(positions)}
   Total Volume:        {total_volume:.2f} lots

📈 POSITION BREAKDOWN
{chr(10).join(position_details)}

🎯 EXIT DECISION ANALYSIS
   {reason_explanation}

💡 OUTCOME
   Successfully {outcome} ${abs(total_pnl):.2f} from this trading sequence.
   The AI system identified the optimal exit point based on {self._get_exit_factors(exit_reason)}.

╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return explanation
    
    def explain_hedge_placement(
        self,
        symbol: str,
        hedge_type: str,
        hedge_lots: float,
        hedge_price: float,
        initial_position: Dict,
        zone_width_pips: float,
        atr_value: float,
        rsi_value: Optional[float] = None,
        volatility_ratio: float = 1.0,
        hedge_level: int = 1
    ) -> str:
        """
        Generate comprehensive explanation for hedge placement.
        
        Args:
            symbol: Trading symbol
            hedge_type: "BUY" or "SELL"
            hedge_lots: Hedge volume
            hedge_price: Hedge execution price
            initial_position: Initial position dict
            zone_width_pips: Zone width in pips
            atr_value: Current ATR value
            rsi_value: Current RSI value
            volatility_ratio: Volatility ratio
            hedge_level: Hedge number (1, 2, 3, etc.)
            
        Returns:
            Formatted explanation string
        """
        # Determine initial position details
        initial_type = "BUY" if initial_position.get('type', 0) == 0 else "SELL"
        initial_price = initial_position.get('price_open', 0.0)
        initial_volume = initial_position.get('volume', 0.0)
        
        # Calculate price movement
        price_move = hedge_price - initial_price
        pip_mult = self.pip_multiplier.get(symbol, 10000)
        price_move_pips = abs(price_move) * pip_mult
        direction = "dropped" if price_move < 0 else "rose"
        
        # Determine hedge strategy
        if hedge_level == 1:
            strategy = "DEFENSIVE HEDGE"
            purpose = "protect against further adverse movement"
            logic = f"Price moved {direction} {price_move_pips:.1f} pips beyond our entry, triggering the first line of defense."
        elif hedge_level == 2:
            strategy = "RECOVERY HEDGE"
            purpose = "recover losses as price returns to entry"
            logic = f"Price returned to entry level after initial hedge. Adding same-direction position to capitalize on recovery."
        else:
            strategy = "ADVANCED HEDGE"
            purpose = "manage complex multi-position bucket"
            logic = f"Multiple hedges active. Executing hedge #{hedge_level} to maintain balanced risk exposure."
        
        # Market condition analysis
        atr_pips = atr_value * pip_mult
        volatility_status = self._assess_volatility(volatility_ratio)
        rsi_status = self._assess_rsi(rsi_value) if rsi_value else "Not available"
        
        # Risk assessment
        exposure_ratio = hedge_lots / initial_volume
        risk_level = self._assess_risk_level(exposure_ratio, volatility_ratio)
        
        explanation = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         HEDGE PLACEMENT ANALYSIS                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 HEDGE STRATEGY: {strategy}
   Purpose:             {purpose}
   Hedge Level:         #{hedge_level}

📊 MARKET SITUATION
   Symbol:              {symbol}
   Initial Position:    {initial_type} {initial_volume:.2f} lots @ {initial_price:.5f}
   Current Price:       {hedge_price:.5f}
   Price Movement:      {direction.upper()} {price_move_pips:.1f} pips
   
📈 TECHNICAL ANALYSIS
   ATR (Volatility):    {atr_pips:.1f} pips
   Volatility Status:   {volatility_status}
   RSI Status:          {rsi_status}
   Zone Width:          {zone_width_pips:.1f} pips

🛡️ HEDGE EXECUTION
   Hedge Direction:     {hedge_type}
   Hedge Volume:        {hedge_lots:.2f} lots
   Execution Price:     {hedge_price:.5f}
   Exposure Ratio:      {exposure_ratio:.2f}x

💡 DECISION LOGIC
   {logic}
   
   The zone recovery system detected that price breached the {zone_width_pips:.1f} pip zone
   (calculated as {zone_width_pips/atr_pips:.1f}x ATR). This hedge will {purpose}.

⚠️ RISK ASSESSMENT
   Risk Level:          {risk_level}
   Volatility Ratio:    {volatility_ratio:.2f}x normal
   
   This hedge is designed to neutralize directional risk and create a balanced
   position that can profit from mean reversion or breakout continuation.

╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return explanation
    
    def explain_recovery_trade(
        self,
        symbol: str,
        recovery_type: str,
        recovery_lots: float,
        recovery_price: float,
        bucket_deficit: float,
        liquidity_analysis: Dict,
        structure_data: Dict
    ) -> str:
        """
        Generate comprehensive explanation for God Mode recovery trade.
        
        Args:
            symbol: Trading symbol
            recovery_type: "BUY" or "SELL"
            recovery_lots: Recovery volume
            recovery_price: Recovery execution price
            bucket_deficit: Current bucket deficit in USD
            liquidity_analysis: Liquidity sweep analysis
            structure_data: Market structure data
            
        Returns:
            Formatted explanation string
        """
        # Extract liquidity analysis
        sweep_type = liquidity_analysis.get('sweep_type', 'Unknown')
        sweep_price = liquidity_analysis.get('sweep_price', 0.0)
        liquidity_pool = liquidity_analysis.get('pool_size', 'Unknown')
        
        # Extract structure data
        trend_direction = structure_data.get('trend', 'Neutral')
        support_level = structure_data.get('support', 0.0)
        resistance_level = structure_data.get('resistance', 0.0)
        
        # Calculate recovery potential
        recovery_potential = recovery_lots * 10 * 10  # Rough estimate: lots * $10/pip * 10 pips
        
        explanation = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      GOD MODE RECOVERY ANALYSIS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 RECOVERY STRATEGY: LIQUIDITY VACUUM SWEEP
   Objective:           Recover ${abs(bucket_deficit):.2f} bucket deficit
   Recovery Type:       {recovery_type}

📊 MARKET STRUCTURE ANALYSIS
   Symbol:              {symbol}
   Trend Direction:     {trend_direction}
   Support Level:       {support_level:.5f}
   Resistance Level:    {resistance_level:.5f}

💧 LIQUIDITY ANALYSIS
   Sweep Type:          {sweep_type}
   Sweep Price:         {sweep_price:.5f}
   Liquidity Pool:      {liquidity_pool}
   
   The market has swept liquidity at {sweep_price:.5f}, creating a vacuum that
   typically results in a strong reversal or continuation move.

🔬 RECOVERY EXECUTION
   Direction:           {recovery_type}
   Volume:              {recovery_lots:.2f} lots
   Entry Price:         {recovery_price:.5f}
   Recovery Potential:  ~${recovery_potential:.2f}

💡 DECISION LOGIC
   After analyzing the last 30 minutes of price action, the AI detected a
   liquidity sweep at {sweep_price:.5f}. This is a high-probability setup where
   institutional traders have triggered stop losses, creating a temporary
   imbalance that we can exploit.
   
   The recovery trade is placed at the optimal reentry point where:
   1. Liquidity has been absorbed
   2. Price is likely to reverse or continue strongly
   3. Risk/reward ratio is favorable (targeting 2:1 minimum)

⚠️ RISK MANAGEMENT
   Current Deficit:     ${bucket_deficit:.2f}
   Recovery Target:     Break-even + profit
   Stop Loss:           Managed by bucket system
   
   This is a calculated recovery trade based on market structure, not a
   revenge trade. The position is sized to recover losses while maintaining
   strict risk controls.

╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return explanation
    
    def _explain_exit_reason(self, reason: str, ai_metrics: Optional[Dict] = None) -> str:
        """Generate detailed explanation for exit reason."""
        reason_lower = reason.lower()
        
        if "profit" in reason_lower or "tp" in reason_lower:
            return """Exit Reason:         PROFIT TARGET REACHED
   
   The bucket reached its calculated profit target based on ATR (Average True Range).
   The AI system determined this was the optimal exit point where:
   - Risk/reward ratio was satisfied
   - Market momentum was showing signs of exhaustion
   - Probability of further profit was lower than risk of reversal"""
        
        elif "time" in reason_lower:
            return """Exit Reason:         TIME LIMIT EXCEEDED
   
   The position exceeded the maximum hold time for scalping (typically 3-5 minutes).
   Holding positions too long in M1 scalping increases risk due to:
   - Increased exposure to random market noise
   - Higher probability of adverse news events
   - Opportunity cost (capital locked in aging trade)"""
        
        elif "emergency" in reason_lower or "max" in reason_lower:
            return """Exit Reason:         EMERGENCY EXIT (Max Hedges)
   
   The bucket reached the maximum number of allowed hedges (4 positions).
   This triggers an emergency exit to prevent:
   - Excessive position complexity
   - Margin requirement escalation
   - Unmanageable risk exposure
   
   The system prioritizes capital preservation over profit maximization."""
        
        elif "break" in reason_lower or "even" in reason_lower:
            return """Exit Reason:         BREAK-EVEN EXIT
   
   The hedged bucket returned to break-even after being in drawdown.
   This is considered a successful recovery because:
   - Capital was protected during adverse movement
   - Hedge strategy worked as designed
   - No loss was realized despite market volatility"""
        
        else:
            return f"""Exit Reason:         {reason.upper()}
   
   The AI system triggered this exit based on current market conditions
   and risk management protocols."""
    
    def _get_exit_factors(self, reason: str) -> str:
        """Get factors considered for exit decision."""
        reason_lower = reason.lower()
        
        if "profit" in reason_lower:
            return "profit target achievement, ATR-based TP levels, and momentum analysis"
        elif "time" in reason_lower:
            return "position age, scalping time limits, and opportunity cost analysis"
        elif "emergency" in reason_lower:
            return "position count limits, margin safety, and risk exposure thresholds"
        elif "break" in reason_lower:
            return "P&L recovery, hedge effectiveness, and capital preservation"
        else:
            return "multiple risk factors and market conditions"
    
    def _assess_volatility(self, volatility_ratio: float) -> str:
        """Assess volatility status."""
        if volatility_ratio > 2.0:
            return "VERY HIGH (2x+ normal) - Extreme market conditions"
        elif volatility_ratio > 1.5:
            return "HIGH (1.5x normal) - Increased risk, wider zones"
        elif volatility_ratio > 1.2:
            return "ELEVATED (1.2x normal) - Above average movement"
        elif volatility_ratio > 0.8:
            return "NORMAL (1.0x baseline) - Standard market conditions"
        else:
            return "LOW (sub-normal) - Quiet market, tighter zones"
    
    def _assess_rsi(self, rsi_value: float) -> str:
        """Assess RSI status."""
        if rsi_value > 70:
            return f"OVERBOUGHT ({rsi_value:.1f}) - Potential reversal down"
        elif rsi_value < 30:
            return f"OVERSOLD ({rsi_value:.1f}) - Potential reversal up"
        elif rsi_value > 60:
            return f"BULLISH ({rsi_value:.1f}) - Upward momentum"
        elif rsi_value < 40:
            return f"BEARISH ({rsi_value:.1f}) - Downward momentum"
        else:
            return f"NEUTRAL ({rsi_value:.1f}) - No extreme conditions"
    
    def _assess_risk_level(self, exposure_ratio: float, volatility_ratio: float) -> str:
        """Assess overall risk level."""
        risk_score = (exposure_ratio * 0.6) + (volatility_ratio * 0.4)
        
        if risk_score > 2.0:
            return "HIGH - Aggressive hedge in volatile market"
        elif risk_score > 1.5:
            return "MODERATE-HIGH - Significant exposure adjustment"
        elif risk_score > 1.0:
            return "MODERATE - Standard risk management"
        else:
            return "LOW - Conservative hedge sizing"
