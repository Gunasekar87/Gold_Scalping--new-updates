"""
Global Constants for AETHER Trading System.

This module centralizes all magic numbers, configuration defaults, and system
constants following enterprise-grade practices.

Author: AETHER Development Team
License: MIT
Version: 2.0.0
"""

from typing import Final
from enum import Enum, auto


# ============================================================================
# System Constants
# ============================================================================

SYSTEM_NAME: Final[str] = "A.E.T.H.E.R."
SYSTEM_VERSION: Final[str] = "6.6.4"  # LOG CLEANUP & STABILITY (Jan 19, 2026)
SYSTEM_FULL_NAME: Final[str] = "Adaptive Evolution Trading & Hedging Execution Robot"


# ============================================================================
# Trading Universe
# ============================================================================

PRIME_CRYPTO: Final[list] = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]
PRIME_FOREX: Final[list] = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
PRIME_COMMODITIES: Final[list] = ["XAUUSD", "XAGUSD", "USOIL", "WTI"]
PRIME_INDICES: Final[list] = ["US30", "NAS100", "SPX500", "DE30"]


# ============================================================================
# Risk Management Constants
# ============================================================================

class RiskLimits:
    """Risk management thresholds."""
    __slots__ = ()
    
    MAX_EQUITY_RISK_PCT: Final[float] = 0.02  # 2% per trade
    MAX_DRAWDOWN_PCT: Final[float] = 0.20  # 20% emergency mode trigger (intelligent exit)
    MAX_DAILY_LOSS_PCT: Final[float] = 0.05  # 5% daily stop (legacy, not used in v6.2.0)
    MAX_POSITIONS_PER_SYMBOL: Final[int] = 4  # Bucket limit
    MIN_FREE_MARGIN_PCT: Final[float] = 0.30  # 30% buffer
    MAX_LOSS_STREAK: Final[int] = 3  # Max consecutive losses
    MIN_ACCOUNT_BALANCE: Final[float] = 100.0  # Minimum account balance
    
    def __setattr__(self, name, value):
        raise AttributeError(f"can't set attribute '{name}'")


class LotSizing:
    """Lot size calculation constants."""
    MIN_LOT: Final[float] = 0.01
    MAX_LOT: Final[float] = 100.0
    LOT_PER_1K_EQUITY: Final[float] = 0.01
    HIGH_CONFIDENCE_MULTIPLIER: Final[float] = 1.2  # >0.8 confidence
    LOW_CONFIDENCE_MULTIPLIER: Final[float] = 0.8   # <0.5 confidence
    CONFIDENCE_THRESHOLD_HIGH: Final[float] = 0.8
    CONFIDENCE_THRESHOLD_LOW: Final[float] = 0.5
    LOT_STEP: Final[float] = 0.01
    BASE_LOT_SIZE: Final[float] = 0.01
    HEDGE1_MULTIPLIER: Final[float] = 0.3
    HEDGE2_MULTIPLIER: Final[float] = 0.5
    HEDGE3_MULTIPLIER: Final[float] = 0.7


# ============================================================================
# AI & Model Constants
# ============================================================================

class AIConfig:
    """AI model configuration."""
    NEXUS_SEQUENCE_LENGTH: Final[int] = 64  # Transformer input
    NEXUS_CONFIDENCE_THRESHOLD: Final[float] = 0.30  # Minimum to trade
    NEXUS_CLASSES: Final[int] = 3  # Sell, Neutral, Buy
    PPO_OBSERVATION_DIM: Final[int] = 4  # [drawdown, ATR, trend, pred]
    PPO_ACTION_DIM: Final[int] = 2  # [hedge_mult, zone_mod]
    PPO_LEARNING_RATE: Final[float] = 0.0003
    PPO_GAMMA: Final[float] = 0.99
    PPO_CLIP_EPSILON: Final[float] = 0.2
    NEXUS_EMBED_DIM: Final[int] = 128
    PPO_N_STEPS: Final[int] = 2048
    NEXUS_NUM_HEADS: Final[int] = 8
    NEXUS_NUM_LAYERS: Final[int] = 6
    PPO_BATCH_SIZE: Final[int] = 64


class CouncilThresholds:
    """Council decision thresholds."""
    MIN_CONFIDENCE: Final[float] = 0.30
    CAUTION_MIN: Final[float] = 0.8
    CAUTION_MAX: Final[float] = 1.5
    WIN_RATE_HISTORY_SIZE: Final[int] = 20
    SENTIMENT_EXTREME_THRESHOLD: Final[float] = 0.5
    MACRO_CONFLICT_THRESHOLD: Final[float] = 0.0005
    NEXUS_MIN_CONFIDENCE: Final[float] = 0.30


class ScalpingConfig:
    """Scalping strategy configuration constants."""
    TP_ATR_MULTIPLIER: Final[float] = 0.3  # 30% of ATR for tight scalping TP
    SL_ATR_MULTIPLIER: Final[float] = 1.0  # 100% of ATR for protective SL
    HEDGE1_ATR_MULTIPLIER: Final[float] = 0.5  # 50% of ATR for first hedge trigger
    HEDGE2_ATR_MULTIPLIER: Final[float] = 1.0  # 100% of ATR for second hedge trigger
    HEDGE3_ATR_MULTIPLIER: Final[float] = 1.5  # 150% of ATR for third hedge trigger
    PENDING_CLOSE_COOLDOWN_SECONDS: Final[float] = 5.0  # Wait time after close orders sent
    MAX_HEDGE_LEVELS: Final[int] = 3  # Maximum number of hedge positions
    SCALPING_AGE_THRESHOLD_SECONDS: Final[float] = 600.0  # 10 minutes for time-based exit


class ZoneRecoveryConfig:
    """Zone recovery configuration constants."""
    MIN_AGE_SECONDS: Final[float] = 3.0  # Minimum position age before hedging
    HEDGE_COOLDOWN_SECONDS: Final[float] = 15.0  # Cooldown between hedge attempts
    BUCKET_CLOSE_COOLDOWN_SECONDS: Final[float] = 15.0  # Cooldown after bucket close
    EMERGENCY_HEDGE_THRESHOLD: Final[int] = 4  # Emergency mode if >= 4 positions
    ZONE_PIPS_DEFAULT: Final[float] = 25.0  # Default zone width in pips
    TP_PIPS_DEFAULT: Final[float] = 25.0  # Default TP in pips
    RETRY_MAX_ATTEMPTS: Final[int] = 3  # Maximum retry attempts
    RETRY_INITIAL_DELAY: Final[float] = 0.5  # Initial retry delay in seconds
    RETRY_BACKOFF_MULTIPLIER: Final[float] = 2.0  # Exponential backoff multiplier
    RETRY_MAX_TOTAL_TIME: Final[float] = 10.0  # Maximum total retry time in seconds
    SENTIMENT_WEIGHT: Final[float] = 0.2
    MACRO_WEIGHT: Final[float] = 0.3
    NEXUS_WEIGHT: Final[float] = 0.5
    IRON_SHIELD_WEIGHT: Final[float] = 0.5


# ============================================================================
# Market Data Constants
# ============================================================================

class MarketData:
    """Market data requirements."""
    MIN_HISTORY_CANDLES: Final[int] = 64
    ATR_PERIOD: Final[int] = 14
    MACRO_H1_CANDLES: Final[int] = 20
    TICK_HISTORY_SIZE: Final[int] = 100
    VOLATILITY_WINDOW_SECONDS: Final[int] = 30
    MIN_TICK_HISTORY: Final[int] = 100
    TICK_BUFFER_SIZE: Final[int] = 200
    CANDLE_AGGREGATION_PERIOD: Final[int] = 60


class SpreadLimits:
    """Maximum allowed spreads by market regime."""
    NORMAL: Final[float] = 30.0  # pips
    HIGH_VOL: Final[float] = 50.0
    PANIC: Final[float] = 100.0  # Allow wider for exits
    NORMAL_MAX_SPREAD_PIPS: Final[float] = 30.0
    HIGH_VOL_MAX_SPREAD_PIPS: Final[float] = 50.0
    PANIC_MAX_SPREAD_PIPS: Final[float] = 100.0
    MIN_SPREAD_PIPS: Final[float] = 0.1


# ============================================================================
# Timing & Cooldown Constants
# ============================================================================

class Timing:
    """Timing and cooldown periods."""
    HEDGE_COOLDOWN_SECONDS: Final[int] = 15
    STATE_SAVE_INTERVAL: Final[int] = 60
    CONFIG_REFRESH_INTERVAL: Final[int] = 60
    REGIME_LOG_INTERVAL: Final[int] = 30
    MIN_POSITION_AGE_SECONDS: Final[int] = 10  # Before hedge
    ZONE_RECOVERY_COOLDOWN: Final[int] = 15
    DASHBOARD_UPDATE_INTERVAL: Final[int] = 60
    TRADE_EXECUTION_TIMEOUT: Final[int] = 30
    MARKET_STATE_UPDATE_INTERVAL: Final[int] = 30


class AdaptiveSleep:
    """Sleep durations by market regime."""
    PANIC: Final[float] = 0.05
    HIGH_VOL: Final[float] = 0.1
    NORMAL: Final[float] = 0.01
    QUIET: Final[float] = 0.5


# ============================================================================
# Circuit Breaker Constants
# ============================================================================

class CircuitBreakerConfig:
    """Circuit breaker thresholds."""
    FAILURE_THRESHOLD: Final[int] = 5
    INITIAL_TIMEOUT_SECONDS: Final[int] = 60
    RESET_TIMEOUT_SECONDS: Final[int] = 300
    MAX_TIMEOUT_SECONDS: Final[int] = 300
    HALF_OPEN_SUCCESS_REQUIRED: Final[int] = 3
    BACKOFF_MULTIPLIER: Final[int] = 2
    BACKOFF_FAILURES_PER_LEVEL: Final[int] = 5
    RESET_TIMEOUT: Final[int] = 300
    INITIAL_BACKOFF_SECONDS: Final[int] = 60
    HALF_OPEN_MAX_CALLS: Final[int] = 10
    MAX_BACKOFF_SECONDS: Final[int] = 3600


# ============================================================================
# Market Regime Definitions
# ============================================================================

class MarketRegime(Enum):
    """Market volatility regimes."""
    QUIET = "QUIET"     # <0.05% in 30s
    NORMAL = "NORMAL"    # 0.05-0.2%
    HIGH_VOL = "HIGH_VOL"  # 0.2-0.5%
    PANIC = "PANIC"     # >0.5%


class VolatilityThresholds:
    """Volatility percentage thresholds for regime detection."""
    PANIC: Final[float] = 0.005  # 0.5%
    HIGH_VOL: Final[float] = 0.002  # 0.2%
    QUIET: Final[float] = 0.0005  # 0.05%


# ============================================================================
# Asset Class Profiles
# ============================================================================

class AssetProfiles:
    """Recommended zone pips by asset class."""
    FOREX_MAJOR: Final[int] = 10
    FOREX_GBP: Final[int] = 15
    FOREX_JPY: Final[int] = 20
    GOLD: Final[int] = 25
    SILVER: Final[int] = 30
    INDEX: Final[int] = 50
    BTC: Final[int] = 100
    ETH: Final[int] = 50
    SOL: Final[int] = 40
    BNB: Final[int] = 30
    XRP: Final[int] = 15
    DOGE: Final[int] = 10


# ============================================================================
# Logging Constants
# ============================================================================

class LogConfig:
    """Logging configuration."""
    FORMAT: Final[str] = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT: Final[str] = '%Y-%m-%d %H:%M:%S'
    DEFAULT_LEVEL: Final[str] = 'INFO'
    NOISY_LIBRARIES: Final[list] = ['httpx', 'httpcore', 'urllib3']
    NOISY_LEVEL: Final[str] = 'WARNING'


# ============================================================================
# Database Constants
# ============================================================================

class DatabaseConfig:
    """Database configuration."""
    SQLITE_PATH: Final[str] = "data/market_memory.db"
    SUPABASE_TIMEOUT: Final[int] = 10
    MAX_RECONNECT_ATTEMPTS: Final[int] = 3
    RECONNECT_DELAY_SECONDS: Final[int] = 5


# ============================================================================
# Trading Signals
# ============================================================================

class TradingSignals(Enum):
    """Valid trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"


class OrderTypes(Enum):
    """Order type constants."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    BUY = "BUY"
    SELL = "SELL"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"


class OrderActions(Enum):
    """Order action constants."""
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    MODIFY = "MODIFY"
    PENDING = "PENDING"


# ============================================================================
# Error Codes
# ============================================================================

class ErrorCodes:
    """Centralized error code definitions."""
    # Broker errors
    BROKER_CONNECTION = "BROKER_001"
    BROKER_EXECUTION = "BROKER_002"
    INSUFFICIENT_MARGIN = "BROKER_003"
    
    # AI errors
    MODEL_LOAD = "AI_001"
    PREDICTION_FAILED = "AI_002"
    INVALID_CONFIDENCE = "AI_003"
    
    # Data errors
    INSUFFICIENT_DATA = "DATA_001"
    INVALID_DATA = "DATA_002"
    
    # Config errors
    CONFIG_ERROR = "CONFIG_001"
    
    # Risk errors
    RISK_LIMIT_EXCEEDED = "RISK_001"
    DRAWDOWN_EXCEEDED = "RISK_002"
    
    # Logic errors
    INVALID_DECISION = "LOGIC_001"
    POSITION_NOT_FOUND = "LOGIC_002"
    
    # Circuit breaker
    CIRCUIT_OPEN = "CIRCUIT_001"
    
    # Market errors
    MARKET_PANIC = "MARKET_001"
    HIGH_SPREAD = "MARKET_002"


# ============================================================================
# MT5 Constants
# ============================================================================

class MT5Config:
    """MetaTrader 5 specific constants."""
    DEFAULT_MAGIC: Final[int] = 888888
    DEFAULT_SLIPPAGE: Final[int] = 10
    DEFAULT_DEVIATION: Final[int] = 10
    COMMENT_PREFIX: Final[str] = "AETHER"
    ALGO_DISABLED_RETCODE: Final[int] = 10027
    SUCCESS_RETCODE: Final[int] = 10009
