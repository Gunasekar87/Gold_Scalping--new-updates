"""
AI Configuration Settings

Centralized configuration for all AI components to avoid magic numbers
scattered throughout the codebase.

Author: AETHER Development Team
Version: 1.0.0
"""

# ============================================================================
# DIRECTION VALIDATOR SETTINGS
# ============================================================================

class DirectionValidatorConfig:
    """Configuration for DirectionValidator component."""
    
    # Factor weights (must sum to 1.0)
    WEIGHTS = {
        'trajectory': 0.20,  # Oracle trajectory prediction
        'trend': 0.20,       # Current trend direction
        'momentum': 0.15,    # RSI momentum
        'ai': 0.15,          # AI consensus
        'levels': 0.15,      # Support/Resistance proximity
        'mtf': 0.10,         # Multi-timeframe alignment
        'flow': 0.05         # Order flow (currently neutral)
    }
    
    # Trajectory thresholds
    TRAJECTORY_NEUTRAL_THRESHOLD = 0.001  # 0.1% - below this is neutral
    TRAJECTORY_STRONG_BULLISH = 0.005     # 0.5% - above this is strong bullish
    TRAJECTORY_STRONG_BEARISH = -0.005    # -0.5% - below this is strong bearish
    TRAJECTORY_MULTIPLIER = 200           # Scaling factor for weak signals
    
    # Validation thresholds
    STRONG_VALIDATION_THRESHOLD = 0.80    # Above this = strong validation
    WEAK_VALIDATION_THRESHOLD = 0.50      # Below this = weak validation
    INVERSION_THRESHOLD = 0.20            # Below this = consider inverting
    
    # Factor significance thresholds
    FACTOR_SIGNIFICANCE_THRESHOLD = 0.3   # Bias must be >0.3 to be significant


# ============================================================================
# WICK INTELLIGENCE SETTINGS
# ============================================================================

class WickIntelligenceConfig:
    """Configuration for WickIntelligence component."""
    
    # Wick detection thresholds
    WICK_THRESHOLD_PCT = 30.0             # Wick must be >30% of candle to be significant
    PROXIMITY_THRESHOLD_PCT = 35.0        # Within 35% of extreme triggers warning (was 20%)
    
    # Exit decision thresholds
    STRONG_EXIT_PROFIT_PIPS = 10.0        # >10 pips = strong exit signal
    MODERATE_EXIT_PROFIT_PIPS = 5.0       # >5 pips = moderate exit signal
    LOSS_HOLD_THRESHOLD_PIPS = -20.0      # <-20 pips = consider holding at support/resistance
    
    # Candle analysis
    NUM_CANDLES_TO_ANALYZE = 5            # Number of recent candles to check
    NUM_CANDLES_FOR_ENTRY = 10            # Number of candles for entry decision


# ============================================================================
# ORACLE SETTINGS
# ============================================================================

class OracleConfig:
    """Configuration for Oracle prediction component."""
    
    # Model paths
    MODEL_PATH = "models/nexus_transformer.pth"
    
    # Prediction settings
    SEQUENCE_LENGTH = 64                  # Number of candles for prediction
    HORIZON_CANDLES = 10                  # Number of candles to predict ahead
    
    # Confidence thresholds
    MIN_CONFIDENCE_THRESHOLD = 0.5        # Minimum confidence to act on prediction
    HIGH_CONFIDENCE_THRESHOLD = 0.7       # High confidence threshold
    
    # Class mapping (MUST match training)
    CLASS_MAPPING = {
        0: "UP",
        1: "DOWN",
        2: "NEUTRAL"
    }


# ============================================================================
# RISK MANAGEMENT SETTINGS
# ============================================================================

class RiskManagementConfig:
    """Configuration for risk management components."""
    
    # Zone recovery
    BASE_ZONE_PIPS = 20.0                 # Base zone width for first hedge
    ZONE_PIPS_HEDGE_2 = 30.0              # Zone width for second hedge
    ZONE_PIPS_HEDGE_3_PLUS = 50.0         # Zone width for third+ hedge
    
    # ATR-based adjustments
    ATR_EXTREME_THRESHOLD_PIPS = 50.0     # ATR >50 pips = extreme volatility
    ATR_ZONE_MULTIPLIER = 0.8             # Scale zone by 80% of ATR if extreme
    
    # Slippage protection
    SLIPPAGE_PAD_ATR_THRESHOLD = 20.0     # Add slippage pad if ATR >20 pips
    SLIPPAGE_PAD_MULTIPLIER = 0.10        # Pad = 10% of ATR
    
    # Profit buffer
    BASE_PROFIT_BUFFER_USD = 0.50         # Base buffer for normal volatility
    HIGH_VOLATILITY_BUFFER_USD = 2.0      # Buffer for high volatility
    VOLATILITY_THRESHOLD = 1.5            # Volatility ratio threshold
    DEEP_HEDGE_BUFFER_MULTIPLIER = 2.0    # 2x buffer for 4+ positions


# ============================================================================
# POSITION MANAGEMENT SETTINGS
# ============================================================================

class PositionManagementConfig:
    """Configuration for position management."""
    
    # Survival mode
    SURVIVAL_MODE_TRADES = 3              # 3+ trades = survival mode
    SURVIVAL_MODE_TARGET_USD = 5.0        # Just $5 in survival mode
    TWO_TRADE_TARGET_MULTIPLIER = 0.5     # 50% target for 2 trades
    
    # Stalemate breaker
    DEEP_HEDGE_ESCAPE_MINUTES = 45        # >45 min with 4+ trades = escape
    DEEP_HEDGE_DECAY_MINUTES = 20         # >20 min with 4+ trades = decay
    MED_HEDGE_ESCAPE_MINUTES = 90         # >90 min with 3 trades = escape
    ESCAPE_TARGET_USD = 0.50              # Escape at just $0.50
    DECAY_MULTIPLIER = 0.25               # Reduce target to 25%
    
    # Ratchet protection
    RATCHET_PEAK_THRESHOLD = 0.8          # Hit 80% of target
    RATCHET_DROP_THRESHOLD = 0.5          # Dropped to 50% of target
    BREAKEVEN_PEAK_THRESHOLD = 0.5        # Hit 50% of target
    BREAKEVEN_DROP_THRESHOLD = 0.1        # Dropped to 10% of target
    MIN_PROFIT_RATCHET = 1.0              # Minimum $1 profit for ratchet
    MIN_PROFIT_BREAKEVEN = 0.50           # Minimum $0.50 for breakeven assist
    
    # TP check thresholds
    TP_CHECK_LOG_THRESHOLD = 0.8          # Log when >80% of TP reached


# ============================================================================
# TRADING ENGINE SETTINGS
# ============================================================================

class TradingEngineConfig:
    """Configuration for trading engine."""
    
    # Freshness validation
    MAX_TICK_AGE_SECONDS = 5.0            # Tick older than 5 seconds = stale
    MAX_CANDLE_AGE_SECONDS = 120.0        # Candle older than 2 minutes = stale
    
    # Confidence thresholds
    MIN_SIGNAL_CONFIDENCE = 0.3           # Minimum confidence to consider signal
    HIGH_SIGNAL_CONFIDENCE = 0.7          # High confidence signal
    
    # Position sizing
    MIN_LOT_SIZE = 0.01                   # Minimum lot size
    MAX_LOT_SIZE = 1.0                    # Maximum lot size (safety limit)
    
    # Pip calculations
    PIP_MULTIPLIER_JPY = 100              # JPY pairs use 2 decimal places
    PIP_MULTIPLIER_XAU = 100              # Gold uses 2 decimal places
    PIP_MULTIPLIER_FOREX = 10000          # Major forex pairs use 4 decimal places


# ============================================================================
# LOGGING SETTINGS
# ============================================================================

class LoggingConfig:
    """Configuration for logging levels and formats."""
    
    # Log levels for different components
    WICK_INTELLIGENCE_LEVEL = "DEBUG"     # DEBUG, INFO, WARNING, ERROR
    DIRECTION_VALIDATOR_LEVEL = "INFO"
    ORACLE_LEVEL = "INFO"
    RISK_MANAGER_LEVEL = "INFO"
    POSITION_MANAGER_LEVEL = "INFO"
    TRADING_ENGINE_LEVEL = "INFO"
    
    # What to log
    LOG_WICK_CHECKS = False               # Log every wick check (verbose)
    LOG_DIRECTION_VALIDATION = True       # Log direction validation
    LOG_PROFIT_CALCULATIONS = True        # Log profit calculations
    LOG_POSITION_UPDATES = True           # Log position updates


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pip_multiplier(symbol: str) -> float:
    """
    Get pip multiplier for a symbol.
    
    Args:
        symbol: Trading symbol (e.g., "XAUUSD", "EURUSD")
        
    Returns:
        Pip multiplier (100 for JPY/XAU, 10000 for forex)
    """
    if "JPY" in symbol or "XAU" in symbol or "GOLD" in symbol:
        return TradingEngineConfig.PIP_MULTIPLIER_XAU
    else:
        return TradingEngineConfig.PIP_MULTIPLIER_FOREX


def get_pip_value_per_lot(symbol: str) -> float:
    """
    Get pip value per lot for a symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        USD value per pip per lot
    """
    if "XAU" in symbol or "GOLD" in symbol:
        return 1.0  # $1 per pip per lot for Gold
    else:
        return 10.0  # $10 per pip per lot for forex
