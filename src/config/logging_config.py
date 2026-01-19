"""
Clean Trader-Focused Logging Configuration for v6.6.0

Removes noise, shows only essential market analysis and trade decisions.
"""

# Logging levels for different components
LOGGING_CONFIG = {
    # === CRITICAL: Always show ===
    "TRADE_EXECUTION": "INFO",      # Actual trades (entries, exits, hedges)
    "MARKET_STATUS": "INFO",         # Market regime, Oracle prediction, key levels
    "RISK_ALERTS": "WARNING",        # Risk warnings, emergency actions
    
    # === IMPORTANT: Show when relevant ===
    "HYBRID_HEDGE": "INFO",          # Hybrid hedge intelligence decisions
    "BUCKET_MANAGEMENT": "INFO",     # Bucket closes, profit targets
    "DIRECTION_VALIDATOR": "WARNING", # Only show weak/inverted signals, not every validation
    "WICK_INTELLIGENCE": "WARNING",  # Only show when blocking trades
    
    # === DEBUG: Suppress unless debugging ===
    "VOLATILITY": "ERROR",           # Too noisy - only show errors
    "DECISION_PATH": "ERROR",        # Forensic logging - only for debugging
    "ADAPTIVE": "ERROR",             # Adaptive multipliers - too detailed
    "AI_COUNCIL": "ERROR",           # Internal AI deliberations - too noisy
    "ORACLE": "ERROR",               # Oracle internal workings - suppress
    "PPO_GUARDIAN": "ERROR",         # PPO training - suppress unless error
    "CHAMELEON": "ERROR",            # Regime detection details - suppress
    
    # === SUPPRESS: Never show ===
    "PAUSED": "CRITICAL",            # Cooldown messages - too repetitive
    "POSITION": "CRITICAL",          # Position sizing details - too noisy
    "HEDGE_CALC": "CRITICAL",        # Hedge calculations - internal
}

# Message deduplication settings
DEDUP_CONFIG = {
    "enabled": True,
    "time_window": 60,  # Don't repeat same message within 60 seconds
    "max_repeats": 1,   # Show message once, then suppress for time_window
}

# Trader-focused summary intervals
SUMMARY_CONFIG = {
    "market_status_interval": 300,  # Show market summary every 5 minutes
    "position_summary_interval": 60, # Show position summary every 1 minute
    "ai_status_interval": 300,      # Show AI status every 5 minutes
}
