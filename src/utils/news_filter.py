import datetime
import logging

logger = logging.getLogger("NewsFilter")

class NewsFilter:
    """
    Simple Calendar Filter to block trading during high-impact news events.
    """
    def __init__(self):
        self.blocked_times = []
        # Example: Block NFP (First Friday of month, 8:30 AM EST)
        # This is a simplified implementation.
        
    def check_status(self) -> tuple[bool, str]:
        """
        Checks if it is safe to trade or if blocked by a time window.
        Returns: (is_safe, reason)
        """
        # [FIX] Use UTC for consistent market hours regardless of server location
        now = datetime.datetime.utcnow()
        
        # 1. BLOCK WEEKENDS (Friday 21:00 UTC to Sunday 21:00 UTC)
        # Forex/Gold usually closes ~22:00 UTC Friday and opens ~22:00 UTC Sunday
        if now.weekday() == 4 and now.hour >= 21: # Friday night (UTC)
            return False, "Market Closed (Weekend - Fri Night)"
        if now.weekday() == 5: # Saturday
            return False, "Market Closed (Weekend - Sat)"
        if now.weekday() == 6 and now.hour < 21: # Sunday morning (UTC)
            return False, "Market Closed (Weekend - Sun Morning)"
            
        # 2. BLOCK SPECIFIC HIGH VOLATILITY WINDOWS (UTC)
        # Example: US Market Open Volatility (13:30 - 14:00 UTC)
        current_time_utc = datetime.datetime.utcnow().time()
        
        # Convert to simple float for comparison (e.g. 13.5 = 13:30)
        current_hour = current_time_utc.hour + (current_time_utc.minute / 60.0)
        
        # Block 13:25 - 13:35 UTC (US Open)
        if 13.41 <= current_hour <= 13.58:
            return False, "US Market Open Volatility Window (13:25-13:35 UTC)"
            
        # 3. [REALIST] BLOCK ASIAN SESSION LOW LIQUIDITY (00:00 - 06:00 UTC)
        # [USER REQUEST] ENABLED 24/7 TRADING - REMOVED RESTRICTION
        # if 0.0 <= current_hour < 6.0:
        #     return False, "Asian Session Low Liquidity (00:00-06:00 UTC)"
            
        return True, "Safe to Trade"

    def should_trade(self) -> bool:
        """Legacy compatibility wrapper."""
        ok, _ = self.check_status()
        return ok
