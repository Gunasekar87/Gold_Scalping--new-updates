import os
from dataclasses import dataclass

def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

@dataclass(frozen=True)
class FeatureFlags:
    # [AI INTELLIGENCE] ENABLED BY DEFAULT FOR HIGHEST PERFORMANCE
    ENABLE_POLICY: bool = _get_bool("AETHER_ENABLE_POLICY", True)
    ENABLE_GOVERNOR: bool = _get_bool("AETHER_ENABLE_GOVERNOR", True)
    ENABLE_TELEMETRY: bool = _get_bool("AETHER_ENABLE_TELEMETRY", True)

@dataclass(frozen=True)
class PolicyTunables:
    MIN_CONFIDENCE: float = float(os.getenv("AETHER_MIN_CONFIDENCE", "0.55"))
    MAX_SPREAD_ATR: float = float(os.getenv("AETHER_MAX_SPREAD_ATR", "0.25"))
    TP_ATR_MULT: float = float(os.getenv("AETHER_TP_ATR_MULT", "0.8"))
    SL_ATR_MULT: float = float(os.getenv("AETHER_SL_ATR_MULT", "1.2"))
    MAX_HEDGE_STACK: int = int(os.getenv("AETHER_MAX_HEDGE_STACK", "3"))

@dataclass(frozen=True)
class RiskLimits:
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("AETHER_MAX_DAILY_LOSS_PCT", "3.0"))
    MAX_TOTAL_EXPOSURE: float = float(os.getenv("AETHER_MAX_TOTAL_EXPOSURE", "3.0"))
    MAX_SPREAD_POINTS: float = float(os.getenv("AETHER_MAX_SPREAD_POINTS", "350"))
    NEWS_LOCKOUT: bool = _get_bool("AETHER_NEWS_LOCKOUT", True)
    # [TREASURY] Capital Reserve (Shadow Balance)
    # 0.5 = Trade with 50% of funds. 50% is "Shadow" (Safety Buffer).
    CAPITAL_RESERVE_PCT: float = float(os.getenv("AETHER_CAPITAL_RESERVE_PCT", "0.5"))

FLAGS = FeatureFlags()
POLICY = PolicyTunables()
RISK = RiskLimits()
