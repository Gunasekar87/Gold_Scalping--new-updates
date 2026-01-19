"""
Trader-Focused Event-Driven Dashboard

Shows only meaningful events:
- Market changes (trend, volatility, regime)
- AI predictions and decisions
- Trade actions (entry, exit, hedge, recovery)
- Position status changes

NO spam, NO repetition, NO technical noise
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("TRADER")


class TraderDashboard:
    """
    Event-driven dashboard for traders
    Shows information only when something significant happens
    """
    
    def __init__(self):
        # Track state to prevent repetition
        self.last_trend = None
        self.last_regime = None
        self.last_atr = None
        self.last_prediction = None
        self.last_position_count = 0
        self.last_pnl_bucket = 0  # Track in $10 buckets
        
    def market_event(self, symbol: str, price: float, trend: str, 
                    regime: str, atr: float, rsi: float):
        """
        Show market event ONLY if trend/regime changed or volatility spiked
        """
        # Check if anything significant changed
        trend_changed = trend != self.last_trend
        regime_changed = regime != self.last_regime
        volatility_spiked = self.last_atr and atr > self.last_atr * 1.2
        
        if not (trend_changed or regime_changed or volatility_spiked):
            return  # Nothing significant, skip
        
        # Build message
        parts = []
        
        if trend_changed:
            parts.append(f"{self.last_trend or 'UNKNOWN'} → {trend}")
        
        if regime_changed:
            parts.append(f"Regime: {regime}")
        
        if volatility_spiked:
            spike_pct = ((atr / self.last_atr) - 1) * 100
            parts.append(f"Volatility +{spike_pct:.0f}%")
        
        # Trend emoji
        emoji = "📈" if trend == "BULLISH" else "📉" if trend == "BEARISH" else "➡️"
        
        msg = f"{emoji} MARKET: {' | '.join(parts)}"
        logger.info(msg)
        
        # Update state
        self.last_trend = trend
        self.last_regime = regime
        self.last_atr = atr
    
    def ai_decision(self, prediction: str, confidence: float, reason: str):
        """
        Show AI decision ONLY if prediction changed
        """
        # Same prediction? Skip
        if prediction == self.last_prediction:
            return
        
        # Confidence indicator
        if confidence > 0.7:
            conf_emoji = "🟢"
        elif confidence > 0.5:
            conf_emoji = "🟡"
        else:
            conf_emoji = "🔴"
        
        logger.info(f"🤖 AI: {prediction} {conf_emoji} {confidence*100:.0f}% - {reason}")
        
        self.last_prediction = prediction
    
    def trade_entry(self, action: str, lots: float, price: float, 
                   reason: str, trade_type: str = "ENTRY"):
        """
        Show trade entry (always show - it's an event)
        """
        emoji_map = {
            "ENTRY": "🟢" if action == "BUY" else "🔴",
            "HEDGE": "🛡️",
            "RECOVERY": "🔄"
        }
        
        emoji = emoji_map.get(trade_type, "📍")
        logger.info(f"{emoji} {trade_type}: {action} {lots:.2f} @ {price:.2f} - {reason}")
    
    def trade_exit(self, num_positions: int, profit: float, 
                  duration_str: str, reason: str):
        """
        Show trade exit (always show - it's an event)
        """
        profit_emoji = "💰" if profit > 0 else "❌"
        logger.info(f"{profit_emoji} EXIT: {num_positions} positions | ${profit:.2f} profit | {duration_str} | {reason}")
    
    def position_change(self, old_count: int, new_count: int, 
                       old_pnl: float, new_pnl: float):
        """
        Show position change ONLY if count changed or P&L crossed $10 threshold
        """
        # Count changed?
        if old_count != new_count:
            logger.info(f"💼 POSITIONS: {old_count} → {new_count} trades")
        
        # P&L crossed $10 threshold?
        old_bucket = int(old_pnl / 10)
        new_bucket = int(new_pnl / 10)
        
        if old_bucket != new_bucket and old_count > 0:
            pnl_emoji = "📈" if new_pnl > old_pnl else "📉"
            logger.info(f"{pnl_emoji} P&L: ${old_pnl:.2f} → ${new_pnl:.2f}")
        
        # Update state
        self.last_position_count = new_count
        self.last_pnl_bucket = new_bucket


# Global instance
_dashboard = None


def get_dashboard() -> TraderDashboard:
    """Get or create global dashboard instance"""
    global _dashboard
    if _dashboard is None:
        _dashboard = TraderDashboard()
    return _dashboard
