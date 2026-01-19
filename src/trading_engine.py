"""
Trading Engine - Core trading logic and orchestration.

This module provides the main trading engine including:
- Trade entry and exit orchestration
- AI decision integration
- Market condition validation
- Rate limiting and safety controls

Author: AETHER Development Team
License: MIT
Version: 1.0.0
"""

import time
import os
import logging
import asyncio
import math
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum

# Import constants
from .constants import ScalpingConfig, ZoneRecoveryConfig

# Import async database components
from .infrastructure.async_database import (
    AsyncDatabaseManager, AsyncDatabaseQueue, TickData, CandleData, TradeData
)
from .utils.trading_logger import TradingLogger, DecisionTracker, format_pips
from .utils.news_filter import NewsFilter
from .utils.news_calendar import NewsCalendar
from .ai_core.tick_pressure import TickPressureAnalyzer
from .utils.trader_dashboard import get_dashboard
from .ai_core.trap_hunter import TrapHunter

# [AI INTELLIGENCE] New Policy & Governance Modules
from src.config.settings import FLAGS, POLICY as _PTUNE, RISK as _RLIM
from src.policy.hedge_policy import HedgePolicy, HedgeConfig
from src.policy.risk_governor import RiskGovernor, RiskLimits
from src.core.trade_authority import TradeAuthority # [PHASE 5] Supreme Court
from src.utils.telemetry import TelemetryWriter, DecisionRecord
from src.utils.data_normalization import normalize_positions, normalize_position
from src.features.market_features import (
    spread_atr,
    zscore,
    simple_breakout_quality,
    simple_regime_trend,
    simple_structure_break,
    linear_regression_slope,
)

logger = logging.getLogger("TradingEngine")
# [CRITICAL] Get the specific UI logger that run_bot.py listens to
ui_logger = logging.getLogger("AETHER_UI")


class TradeAction(Enum):
    """Enumeration of possible trade actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Represents a trading signal with metadata."""
    action: TradeAction
    symbol: str
    confidence: float
    reason: str
    metadata: Dict[str, Any]
    timestamp: float

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class TradingConfig:
    """Configuration for trading parameters."""
    symbol: str
    initial_lot: float
    max_lot: float = 100.0
    min_lot: float = 0.01
    global_trade_cooldown: float = 5.0  # Reduced from 15.0s for Continuous Scalping
    max_spread_pips: float = 30.0
    risk_multiplier_range: Tuple[float, float] = (0.1, 2.0)
    timeframe: str = "M1"



import logging
logger = logging.getLogger("TradingEngine")

class TradingEngine:
    """
    Core trading engine that orchestrates all trading activities.

    This class handles:
    - AI decision processing and validation
    - Trade execution with safety checks
    - Rate limiting and cooldown management
    - Integration with all trading components
    """

    def __init__(self, config: TradingConfig, broker_adapter, market_data, position_manager, risk_manager, db_manager: Optional[AsyncDatabaseManager] = None, ppo_guardian=None, global_brain=None, tick_analyzer=None):
        self.config = config
        self.broker = broker_adapter
        self.market_data = market_data
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.ppo_guardian = ppo_guardian
        self.global_brain = global_brain # Layer 9: Inter-Market Correlation

        # [PHASE 3] Initialize Supervisor and Workers
        from .ai_core.supervisor import Supervisor
        from .ai_core.workers import RangeWorker, TrendWorker
        from .ai.hedge_intelligence import HedgeIntelligence # [NEW] The Oracle
        
        self.supervisor = Supervisor()
        self.range_worker = RangeWorker()
        self.trend_worker = TrendWorker()
        self.hedge_intel = HedgeIntelligence(self.config) # [NEW] Initialize Oracle
        self.news_filter = NewsFilter() # [AI INTELLIGENCE] Initialize News Filter
        self.news_calendar = NewsCalendar() # [HIGHEST INTELLIGENCE] Event Horizon
        self.tick_analyzer = tick_analyzer if tick_analyzer else TickPressureAnalyzer() # [HIGHEST INTELLIGENCE] Tick Pressure
        self.trap_hunter = TrapHunter(self.tick_analyzer) # [FAKE-OUT DEFENSE] Initialize Trap Hunter

        # [AI INTELLIGENCE] Initialize Policy & Governance
        self._telemetry = TelemetryWriter()
        self._hedge_policy = HedgePolicy(HedgeConfig(
            min_confidence=_PTUNE.MIN_CONFIDENCE,
            max_spread_atr=_PTUNE.MAX_SPREAD_ATR,
            tp_atr_mult=_PTUNE.TP_ATR_MULT,
            sl_atr_mult=_PTUNE.SL_ATR_MULT,
            max_stack=_PTUNE.MAX_HEDGE_STACK,
        ))
        self._governor = RiskGovernor(RiskLimits(
            max_total_exposure_pct=0.15,  # 15% max margin usage
            max_drawdown_pct=0.20,  # 20% emergency mode trigger
            position_limit_per_1k=2,  # 2 positions per $1000 balance
            news_lockout=False,  # News lockout disabled by default
        ))
        
        # [PHASE 5] The Treasury & Supreme Court
        self.authority = TradeAuthority()
        logger.info("[INIT] Trade Authority & Risk Governor Online")
        
        # [DASHBOARD] Initialize Trader Dashboard link
        from .utils.trader_dashboard import get_dashboard
        self.dashboard = get_dashboard()
        logger.info(f"TradeAuthority initialized (Global Cap: {self.authority.current_global_cap})")

        # Async database components
        self.db_manager = db_manager
        
        # Decision Tracker
        self.decision_tracker = DecisionTracker()
        self.db_queue: Optional[AsyncDatabaseQueue] = None

        # Rate limiting
        self.last_trade_time = 0.0
        self.trade_cooldown_active = False
        self.entry_cooldowns = {}  # [FIX] Per-symbol entry cooldowns to prevent double entries
        
        # Status Tracking (For UI Feedback)
        self.last_pause_reason = None

        # Strict entry gating (user-selected mode A: strict entries / soft exits)
        self._strict_entry = str(os.getenv("AETHER_STRICT_ENTRY", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._strict_entry_min_candles = int(os.getenv("AETHER_STRICT_ENTRY_MIN_CANDLES", "60"))
        self._strict_tick_max_age_s = float(os.getenv("AETHER_STRICT_TICK_MAX_AGE_S", "5.0"))

        # Freshness gate for ANY NEW order (entry/hedge/recovery)
        self._freshness_gate = str(os.getenv("AETHER_ENABLE_FRESHNESS_GATE", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._fresh_tick_max_age_s = float(
            os.getenv("AETHER_FRESH_TICK_MAX_AGE_S", "10.0")  # Increased from 5.0s to 10.0s for recovery trades
        )
        # 0/negative = auto based on timeframe (2x TF + 10s)
        self._fresh_candle_close_max_age_s = float(os.getenv("AETHER_FRESH_CANDLE_CLOSE_MAX_AGE_S", "0"))
        self._freshness_trace = str(os.getenv("AETHER_FRESHNESS_TRACE", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._last_freshness_ok: Optional[bool] = None
        self._last_freshness_trace_ts: float = 0.0
        self._time_offset: Optional[float] = None  # Auto-detected timezone offset
        self._last_offset_calc: float = 0.0  # Track when offset was last calculated

        # Optional data provenance trace (tick/candles/OBI availability) for live verification
        self._data_provenance_trace = str(os.getenv("AETHER_DATA_PROVENANCE_TRACE", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            self._data_provenance_trace_interval_s = float(os.getenv("AETHER_DATA_PROVENANCE_TRACE_EVERY_S", "30"))
        except Exception:
            self._data_provenance_trace_interval_s = 30.0
        self._last_data_prov_trace_ts: float = 0.0

        self._last_entry_gate_reason: Optional[str] = None
        self._last_entry_gate_ts: float = 0.0

        # Decision tracking for smart logging (only log changes)
        self.decision_tracker = DecisionTracker()
        self._last_signal_logged = False  # Track if we just logged a signal
        self._last_position_status_time = 0.0  # Track last position status log time

        # Track last cooldown log time
        self._last_cooldown_log_time = 0.0

        # ENHANCEMENT 7: Comprehensive Performance Tracking
        # Statistics
        self.session_stats = {
            # Basic counters
            "trades_opened": 0,
            "trades_closed": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "start_time": time.time(),
            
            # Learning/optimization metrics
            "equity_peak": None,
            "max_drawdown_pct": 0.0,
            
            # ENHANCEMENT 7: Detailed Performance Metrics
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            
            # Streak tracking
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            
            # Trade duration tracking
            "trade_durations": [],
            "avg_trade_duration_seconds": 0.0,
            "min_trade_duration": 0.0,
            "max_trade_duration": 0.0,
            
            # Returns for Sharpe calculation
            "returns": [],
            
            # Hourly performance
            "hourly_pnl": {},
        }

        # End-of-session optimization metrics (profit + stability)
        self._equity_peak: Optional[float] = None
        self._max_drawdown_pct: float = 0.0
        self._last_equity_check_ts: float = 0.0

        logger.info(f"TradingEngine initialized for {config.symbol}")
        logger.info("[ENHANCEMENT 7] Advanced performance tracking enabled")
        
        # INTEGRATION FIX: Initialize Model Monitor (Enhancement #8)
        try:
            from .utils.model_monitor import ModelMonitor
            self.model_monitor = ModelMonitor()
            logger.info("[ENHANCEMENT 8] Model monitoring initialized")
        except Exception as e:
            logger.warning(f"Model monitor not available: {e}")
            self.model_monitor = None
        
        # INTEGRATION FIX: Setup callbacks for position_manager
        if hasattr(self.position_manager, 'callbacks'):
            self.position_manager.callbacks['on_trade_close'] = self._update_performance_metrics
            if self.model_monitor:
                self.position_manager.callbacks['on_prediction_outcome'] = self._record_prediction_outcome
            # Add strategist callback
            self.position_manager.callbacks['on_strategist_update'] = self._update_strategist_stats
            logger.info("[INTEGRATION] Position manager callbacks configured")

    def _log_entry_gate(self, reason: str) -> None:
        """Throttled log for strict entry gating (avoids spam)."""
        now = time.time()
        if reason != self._last_entry_gate_reason or (now - self._last_entry_gate_ts) > 10.0:
            logger.info(f"[ENTRY_GATED] {reason}")
            self._last_entry_gate_reason = reason
            self._last_entry_gate_ts = now

    def _get_fresh_candle_close_max_age_s(self) -> float:
        """Auto candle-close freshness threshold based on configured timeframe."""
        try:
            configured = float(self._fresh_candle_close_max_age_s)
        except Exception:
            configured = 0.0

        if configured and configured > 0:
            return float(configured)

        tf_s = 60.0
        try:
            tf_s = float(self.market_data.candles._timeframe_seconds())
        except Exception:
            tf_s = 60.0

        # Default: allow up to ~2 bars behind + a small buffer
        return float(max((2.0 * tf_s) + 10.0, 30.0))

    def _freshness_ok_for_new_orders(self, tick: Dict) -> Tuple[bool, str]:
        """Return (ok, reason) based on tick age and last completed candle close age."""
        if not self._freshness_gate:
            return True, "disabled"

        max_tick = float(self._fresh_tick_max_age_s) if self._fresh_tick_max_age_s else 0.0
        max_candle = self._get_fresh_candle_close_max_age_s()

        now = time.time()
        tick_ts = float(tick.get('time', 0.0) or 0.0)

        # [TIMEZONE AUTO-CORRECTION]
        # Recalculate offset every hour to handle server time drift
        should_recalc = (
            self._time_offset is None or 
            (now - self._last_offset_calc) > 3600  # Recalculate every hour
        )
        
        if should_recalc and tick_ts > 0:
            raw_diff = now - tick_ts
            # If diff is > 10 mins (600s), assume timezone/clock skew and compensate
            if abs(raw_diff) > 600:
                old_offset = self._time_offset
                self._time_offset = raw_diff
                self._last_offset_calc = now
                if old_offset is None:
                    logger.debug(f"[FRESHNESS] Detected Timezone Offset: {self._time_offset:.2f}s ({self._time_offset/3600:.1f} hours). Auto-adjusting freshness checks.")
                elif abs(old_offset - self._time_offset) > 60:  # Log if drift > 1 minute
                    logger.warning(f"[FRESHNESS] Updated Timezone Offset: {old_offset:.2f}s → {self._time_offset:.2f}s")
            else:
                self._time_offset = 0.0
                self._last_offset_calc = now
        
        offset = self._time_offset if self._time_offset is not None else 0.0
        
        try:
            # FIX: Don't adjust timestamp - use raw_diff directly when offset exists
            # The issue was double-adjustment: tick_ts might already be in broker time
            if tick_ts > 0:
                if abs(offset) > 600:  # Large offset detected (timezone issue)
                    # Use the pre-calculated raw_diff which is already correct
                    tick_age = abs(now - tick_ts) - abs(offset)  # Subtract offset from age
                    tick_age = max(0, tick_age)  # Ensure non-negative
                else:
                    # Normal case: no timezone issue
                    tick_age = abs(now - tick_ts)
            else:
                tick_age = float(tick.get('tick_age_s', float('inf')))
        except Exception as e:
            logger.error(f"[FRESHNESS_ERROR] {e}")
            tick_age = float('inf')

        try:
            candle_close_age = float(tick.get('candle_close_age_s', float('inf')))
        except Exception:
            candle_close_age = float('inf')

        if max_tick > 0 and tick_age > max_tick:
            return False, f"stale_tick age={tick_age:.2f}s max={max_tick:.2f}s (offset={offset:.2f}s)"
        if max_candle > 0 and candle_close_age > max_candle:
            return False, f"stale_candle_close age={candle_close_age:.2f}s max={max_candle:.2f}s"

        return True, f"ok tick_age={tick_age:.2f}s candle_close_age={candle_close_age:.2f}s"

    def _maybe_log_freshness_trace(self, tick: Dict) -> None:
        """Optional freshness telemetry for live verification (throttled).

        Logs only when freshness state changes (ok<->blocked) or every ~30s.
        """
        if not self._freshness_trace:
            return

        ok, reason = self._freshness_ok_for_new_orders(tick)
        now = time.time()

        should_log = False
        if self._last_freshness_ok is None or bool(ok) != bool(self._last_freshness_ok):
            should_log = True
        elif (now - float(self._last_freshness_trace_ts)) > 30.0:
            should_log = True

        if not should_log:
            return

        try:
            tick_age = float(tick.get('tick_age_s', float('inf')))
        except Exception:
            tick_age = float('inf')
        try:
            candle_age = float(tick.get('candle_close_age_s', float('inf')))
        except Exception:
            candle_age = float('inf')

        max_tick = float(self._fresh_tick_max_age_s) if self._fresh_tick_max_age_s else 0.0
        max_candle = self._get_fresh_candle_close_max_age_s()

        logger.info(
            f"[FRESHNESS_TRACE] ok={bool(ok)} reason={reason} "
            f"tick_age_s={tick_age:.2f} max_tick_s={max_tick:.2f} "
            f"candle_close_age_s={candle_age:.2f} max_candle_s={max_candle:.2f}"
        )

        self._last_freshness_ok = bool(ok)
        self._last_freshness_trace_ts = now

    def _maybe_log_data_provenance_trace(self, symbol: str, tick: Dict) -> None:
        """Optional trace proving which inputs are live vs missing.

        This does not change decisions; it only logs a compact snapshot periodically.
        """
        if not self._data_provenance_trace:
            return

        now = time.time()
        interval = float(self._data_provenance_trace_interval_s) if self._data_provenance_trace_interval_s else 30.0
        if interval <= 0:
            interval = 30.0
        if (now - float(self._last_data_prov_trace_ts)) < interval:
            return
        self._last_data_prov_trace_ts = now

        # Tick timing
        try:
            tick_ts = float(tick.get('time', 0.0) or 0.0)
        except Exception:
            tick_ts = 0.0
        try:
            tick_age = float(tick.get('tick_age_s', float('inf')))
        except Exception:
            tick_age = float('inf')

        # Candle timing (latest completed candle close)
        try:
            candle_close_ts = float(tick.get('candle_close_ts', 0.0) or 0.0)
        except Exception:
            candle_close_ts = 0.0
        try:
            candle_close_age = float(tick.get('candle_close_age_s', float('inf')))
        except Exception:
            candle_close_age = float('inf')

        # Candle history snapshot (count + latest bar open)
        candle_count = None
        last_bar_open_ts = None
        try:
            history = self.market_data.candles.get_history(symbol)
            candle_count = int(len(history)) if history else 0
            if history:
                last_bar_open_ts = float(history[-1].get('time', 0) or 0)
        except Exception:
            candle_count = None
            last_bar_open_ts = None

        # Order book / OBI availability (from MarketDataManager tick enrichment)
        obi = tick.get('obi', None)
        obi_ok = bool(tick.get('obi_ok', False))
        obi_applicable = bool(tick.get('obi_applicable', False))

        logger.debug(
            "[DATA_TRACE] "
            f"symbol={symbol} broker={type(self.broker).__name__ if self.broker else None} "
            f"tick_ts={tick_ts:.0f} tick_age_s={tick_age:.2f} "
            f"candle_close_ts={candle_close_ts:.0f} candle_close_age_s={candle_close_age:.2f} "
            f"candles_n={candle_count} last_bar_open_ts={last_bar_open_ts} "
            f"obi={obi} obi_ok={obi_ok} obi_applicable={obi_applicable}"
        )

    def _update_equity_metrics_throttled(self) -> None:
        """Track equity peak and max drawdown, throttled for HFT loop safety."""
        now = time.time()
        # Default: check once every 2 seconds to avoid broker/API spam.
        try:
            every_s = float(os.getenv("AETHER_EQUITY_TRACK_EVERY_S", "2.0"))
        except Exception:
            every_s = 2.0

        if every_s <= 0:
            every_s = 2.0

        if (now - self._last_equity_check_ts) < every_s:
            return
        self._last_equity_check_ts = now

        try:
            acct = self.broker.get_account_info() if self.broker else None
            if not acct:
                return
            equity = float(acct.get('equity', 0.0) or 0.0)
            if equity <= 0:
                return

            if self._equity_peak is None or equity > float(self._equity_peak):
                self._equity_peak = float(equity)

            peak = float(self._equity_peak) if self._equity_peak else float(equity)
            if peak > 0:
                dd_pct = max(0.0, (peak - equity) / peak)
                if dd_pct > self._max_drawdown_pct:
                    self._max_drawdown_pct = float(dd_pct)

            # Expose for dashboards / end-of-session tuning
            self.session_stats["equity_peak"] = float(self._equity_peak) if self._equity_peak is not None else None
            self.session_stats["max_drawdown_pct"] = float(self._max_drawdown_pct)
        except Exception:
            return
    
    # ============================================================================
    # ENHANCEMENT 7: Performance Metrics Update
    # Added: January 4, 2026
    # Purpose: Track detailed performance statistics for each closed trade
    # ============================================================================
    
    def _update_performance_metrics(self, profit: float, duration_seconds: float = 0.0):
        """
        Update detailed performance metrics when a trade closes.
        
        ENHANCEMENT 7: Comprehensive performance tracking
        
        Args:
            profit: Profit/loss from the trade
            duration_seconds: How long the trade was open
        """
        try:
            # Update basic counters
            self.session_stats["trades_closed"] += 1
            self.session_stats["total_profit"] += profit
            
            # Categorize trade outcome
            if profit > 0.01:  # Win (threshold to avoid rounding errors)
                self.session_stats["wins"] += 1
                self.session_stats["consecutive_wins"] += 1
                self.session_stats["consecutive_losses"] = 0
                
                # Track largest win
                if profit > self.session_stats["largest_win"]:
                    self.session_stats["largest_win"] = profit
                
                # Update max consecutive wins
                if self.session_stats["consecutive_wins"] > self.session_stats["max_consecutive_wins"]:
                    self.session_stats["max_consecutive_wins"] = self.session_stats["consecutive_wins"]
                    
            elif profit < -0.01:  # Loss
                self.session_stats["losses"] += 1
                self.session_stats["consecutive_losses"] += 1
                self.session_stats["consecutive_wins"] = 0
                
                # Track largest loss
                if profit < self.session_stats["largest_loss"]:
                    self.session_stats["largest_loss"] = profit
                
                # Update max consecutive losses
                if self.session_stats["consecutive_losses"] > self.session_stats["max_consecutive_losses"]:
                    self.session_stats["max_consecutive_losses"] = self.session_stats["consecutive_losses"]
                    
            else:  # Breakeven
                self.session_stats["breakeven"] += 1
                self.session_stats["consecutive_wins"] = 0
                self.session_stats["consecutive_losses"] = 0
            
            # Calculate win rate
            total_trades = self.session_stats["wins"] + self.session_stats["losses"] + self.session_stats["breakeven"]
            if total_trades > 0:
                self.session_stats["win_rate"] = self.session_stats["wins"] / total_trades
            
            # Calculate average win/loss
            if self.session_stats["wins"] > 0:
                # Calculate from all profitable trades
                wins_total = sum([p for p in self.session_stats.get("returns", []) if p > 0.01])
                self.session_stats["avg_win"] = wins_total / self.session_stats["wins"] if wins_total else 0.0
            
            if self.session_stats["losses"] > 0:
                # Calculate from all losing trades
                losses_total = abs(sum([p for p in self.session_stats.get("returns", []) if p < -0.01]))
                self.session_stats["avg_loss"] = losses_total / self.session_stats["losses"] if losses_total else 0.0
            
            # Calculate profit factor
            gross_profit = sum([p for p in self.session_stats.get("returns", []) if p > 0])
            gross_loss = abs(sum([p for p in self.session_stats.get("returns", []) if p < 0]))
            
            if gross_loss > 0:
                self.session_stats["profit_factor"] = gross_profit / gross_loss
            else:
                self.session_stats["profit_factor"] = 999.0 if gross_profit > 0 else 0.0
            
            # Track returns for Sharpe ratio
            self.session_stats["returns"].append(profit)
            
            # Calculate Sharpe ratio (simplified: returns / std dev of returns)
            if len(self.session_stats["returns"]) >= 10:
                import numpy as np
                returns_array = np.array(self.session_stats["returns"])
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                if std_return > 0:
                    # Annualized Sharpe (assuming ~250 trading days, ~50 trades/day)
                    self.session_stats["sharpe_ratio"] = (mean_return / std_return) * np.sqrt(250 * 50)
                else:
                    self.session_stats["sharpe_ratio"] = 0.0
            
            # Track trade duration
            if duration_seconds > 0:
                self.session_stats["trade_durations"].append(duration_seconds)
                
                # Calculate duration stats
                durations = self.session_stats["trade_durations"]
                self.session_stats["avg_trade_duration_seconds"] = sum(durations) / len(durations)
                self.session_stats["min_trade_duration"] = min(durations)
                self.session_stats["max_trade_duration"] = max(durations)
            
            # Track hourly P&L
            current_hour = time.strftime("%H:00")
            if current_hour not in self.session_stats["hourly_pnl"]:
                self.session_stats["hourly_pnl"][current_hour] = 0.0
            self.session_stats["hourly_pnl"][current_hour] += profit
            
            # Log summary every 10 trades
            if self.session_stats["trades_closed"] % 10 == 0:
                logger.info(
                    f"[PERFORMANCE] Trades: {self.session_stats['trades_closed']} | "
                    f"WinRate: {self.session_stats['win_rate']:.1%} | "
                    f"PF: {self.session_stats['profit_factor']:.2f} | "
                    f"Sharpe: {self.session_stats['sharpe_ratio']:.2f}"
                )
                
        except Exception as e:
            logger.warning(f"[PERFORMANCE] Error updating metrics: {e}")
    
    # ============================================================================
    # INTEGRATION FIX: Model Monitor Integration
    # Added: January 4, 2026
    # Purpose: Record prediction outcomes for AI accuracy tracking
    # ============================================================================
    
    def _record_prediction_outcome(self, timestamp: float, actual_direction: str, profit: float):
        """
        Record actual trade outcome for model monitoring.
        
        INTEGRATION FIX: Connects Enhancement #8 with position manager
        
        Args:
            timestamp: Timestamp when prediction was made
            actual_direction: Actual outcome ('UP', 'DOWN', 'NEUTRAL')
            profit: Actual profit/loss
        """
        if not self.model_monitor:
            return
        
        try:
            self.model_monitor.record_outcome(
                timestamp=timestamp,
                actual_direction=actual_direction,
                profit=profit
            )
            
            # Log accuracy summary every 50 trades
            matched_count = sum(1 for p in self.model_monitor.predictions if p['actual'] is not None)
            if matched_count > 0 and matched_count % 50 == 0:
                accuracy = self.model_monitor.get_accuracy()
                should_retrain, reason = self.model_monitor.should_retrain()
                
                logger.info(
                    f"[MODEL MONITOR] Accuracy: {accuracy:.2%} | "
                    f"Predictions: {matched_count} | "
                    f"Status: {reason}"
                )
                
                if should_retrain:
                    logger.warning(f"[MODEL MONITOR] ALERT: {reason}")
                    
        except Exception as e:
            logger.warning(f"[MODEL MONITOR] Error recording outcome: {e}")
    
    # ============================================================================
    # INTEGRATION FIX: Strategist Win Rate Integration
    # Added: January 5, 2026
    # Purpose: Update strategist win rate tracking for Kelly Criterion
    # ============================================================================
    
    def _update_strategist_stats(self, profit: float, win: bool):
        """
        Update strategist win rate statistics.
        
        INTEGRATION FIX: Connects Enhancement #4 with position manager
        
        Args:
            profit: Trade profit/loss
            win: Whether trade was profitable
        """
        try:
            # Strategist is passed in run_trading_cycle, store reference if needed
            # For now, log the update - full integration requires strategist reference
            logger.debug(f"[STRATEGIST] Trade result: {'WIN' if win else 'LOSS'} | Profit: ${profit:.2f}")
            
            # TODO: When strategist reference is available:
            # if hasattr(self, '_strategist_ref') and self._strategist_ref:
            #     self._strategist_ref.update_stats(profit=profit, win=win)
            
        except Exception as e:
            logger.warning(f"[STRATEGIST] Error updating stats: {e}")


    async def initialize_database(self) -> None:
        """Initialize async database components."""
        if self.db_manager:
            try:
                await self.db_manager.connect()
                await self.db_manager.initialize_schema()
                self.db_queue = AsyncDatabaseQueue(self.db_manager)
                await self.db_queue.start()
                logger.info("Database components initialized")
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                self.db_manager = None

    async def shutdown_database(self) -> None:
        """Shutdown async database components."""
        if self.db_queue:
            await self.db_queue.stop()
        if self.db_manager:
            await self.db_manager.disconnect()

    def validate_market_conditions(self, symbol: str, tick: Dict) -> Tuple[bool, str]:
        """
        Validate if market conditions are suitable for trading.

        Args:
            symbol: Trading symbol
            tick: Current tick data

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check basic market data validity
        if not tick or 'bid' not in tick or 'ask' not in tick:
            return False, "Invalid tick data"

        # [HIGHEST INTELLIGENCE] Check Event Horizon (News Blackout)
        if self.news_calendar.is_blackout_period(symbol):
            return False, "News Blackout (Event Horizon)"

        # Check spread
        # Adjust multiplier for JPY and XAU pairs
        multiplier = 10000
        if "JPY" in symbol or "XAU" in symbol:
            multiplier = 100
            
        spread_pips = abs(tick['ask'] - tick['bid']) * multiplier
        if spread_pips > self.config.max_spread_pips:
            return False, f"Spread too wide: {spread_pips:.1f} pips"

        # Check market data manager conditions
        return self.market_data.validate_market_conditions(symbol, tick)

    def _validate_physics_conditions(self, action: str, regime, pressure_metrics: Dict, tick: Dict) -> Tuple[bool, str]:
        """
        Validates trade against 'Unified Field Theory' physics:
        1. Reynolds Number (Turbulence) -> Block if extreme turbulence
        2. VPIN (Toxicity) -> Block if toxic flow detected
        3. Entropy -> Already handled by regime detection
        
        Args:
            action: Trade action ("BUY" or "SELL")
            regime: Market regime from RegimeDetector
            pressure_metrics: Physics/chemistry metrics from TickPressureAnalyzer
            tick: Current tick data
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not pressure_metrics:
            return True, "No Physics Data"

        try:
            # 1. REYNOLDS NUMBER (Turbulence)
            physics = pressure_metrics.get('physics', {})
            reynolds = physics.get('reynolds_number', 0.0)
            
            # If Re > 5000 (Extreme Turbulence), standard directional logic fails
            if reynolds > 5000:
                return False, f"Extreme Turbulence (Re={reynolds:.0f})"

            # 2. VPIN (Toxicity)
            chemistry = pressure_metrics.get('chemistry', {})
            vpin = chemistry.get('vpin', 0.0)
            
            # If VPIN > 0.7 (Very Toxic Flow), avoid providing liquidity
            if vpin > 0.7:
                return False, f"Toxic Flow Detected (VPIN={vpin:.2f})"

            return True, "Physics OK"

        except Exception as e:
            logger.error(f"[PHYSICS] Validation error: {e}")
            return True, "Physics Error (Default Allow)"

    def calculate_position_size(self, signal: TradeSignal, account_info: Dict,
                              strategist, shield, ppo_guardian=None, 
                              atr_value=0.001, trend_strength=0.0) -> Tuple[float, str]:
        """
        Calculate appropriate position size for a trade signal with PPO optimization.

        Args:
            signal: Trade signal
            account_info: Account information
            strategist: Strategist instance
            shield: IronShield instance
            ppo_guardian: PPO Guardian instance for AI-driven sizing (optional)
            atr_value: Current ATR for volatility assessment
            trend_strength: Trend strength for PPO decision

        Returns:
            Tuple of (lot_size, reason)
        """
        try:
            # Get base lot size from shield
            raw_lot = shield.calculate_entry_lot(
                account_info.get('equity', 1000),
                confidence=signal.confidence,
                atr_value=atr_value,
                trend_strength=trend_strength
            )

            if raw_lot is None or raw_lot <= 0:
                return 0.0, "Invalid base lot calculation"

            # Apply strategist risk multiplier
            risk_mult = strategist.get_risk_multiplier()

            # IMPORTANT:
            # IronShield.calculate_entry_lot() already applies equity/confidence/ATR/trend scaling.
            # Re-applying balance/conf scalers here causes quadratic growth in size (high-risk).
            # If you want the legacy extra scalers, enable AETHER_ENGINE_EXTRA_SCALING=1.
            enable_extra_scaling = str(os.getenv("AETHER_ENGINE_EXTRA_SCALING", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

            win_rate_scale = 1.0
            if hasattr(strategist, 'recent_win_rate'):
                try:
                    if strategist.recent_win_rate > 0.6:
                        win_rate_scale = 1.1
                    elif strategist.recent_win_rate < 0.4:
                        win_rate_scale = 0.8
                except Exception:
                    win_rate_scale = 1.0

            if enable_extra_scaling:
                equity = account_info.get('equity', 1000)
                balance_scale = max(1.0, equity / 1000.0)

                conf_scale = 1.0
                if signal.confidence > 0.8:
                    conf_scale = 1.2
                elif signal.confidence < 0.5:
                    conf_scale = 0.8

                dynamic_lot = raw_lot * balance_scale * conf_scale * win_rate_scale * risk_mult
            else:
                dynamic_lot = raw_lot * win_rate_scale * risk_mult
            
            # NEW: Apply PPO Guardian AI optimization
            ppo_mult = 1.0  # Default if PPO not available
            if ppo_guardian:
                try:
                    # Calculate pip multiplier for logging
                    pip_multiplier = 100 if "XAU" in signal.symbol or "GOLD" in signal.symbol else 10000
                    
                    ppo_mult = ppo_guardian.get_position_size_multiplier(
                        atr=atr_value,
                        trend_strength=trend_strength,
                        confidence=signal.confidence,
                        current_equity=account_info.get('equity', 1000)
                    )
                    atr_pips = atr_value * pip_multiplier
                    # Changed to DEBUG to prevent log spam
                    logger.debug(f"[PPO_SIZING] Position Multiplier: {ppo_mult:.2f}x | ATR: {atr_pips:.1f}pips | Trend: {trend_strength:.2f} | Conf: {signal.confidence:.2f}")
                except Exception as ppo_error:
                    logger.warning(f"[PPO_SIZING] Failed: {ppo_error}, using 1.0x default")
                    ppo_mult = 1.0
            
            # Calculate final lot size with ALL multipliers
            # Use 4 decimals to allow for micro-lots (0.001) if broker supports it.
            # MT5 Adapter will normalize to exact step (e.g. 0.01 or 0.001).
            entry_lot = round(dynamic_lot * ppo_mult, 4)

            # Validate lot size bounds
            if entry_lot < self.config.min_lot:
                # If calculated is too small but we have equity, default to min_lot
                entry_lot = self.config.min_lot

            if entry_lot > self.config.max_lot:
                entry_lot = self.config.max_lot

            reason_parts = [
                f"Lot: {entry_lot}",
                f"Base: {raw_lot:.2f}",
                f"WinScale: {win_rate_scale:.1f}"
            ]

            if enable_extra_scaling:
                reason_parts.extend([
                    f"BalScale: {balance_scale:.1f}",
                    f"ConfScale: {conf_scale:.1f}",
                ])
            
            if ppo_guardian and ppo_mult != 1.0:
                reason_parts.append(f"ppo_mult: {ppo_mult:.2f}")
            
            return entry_lot, " | ".join(reason_parts)

        except Exception as e:
            logger.exception(f"Error calculating position size: {e}")
            return 0.0, f"Calculation error: {e}"

    def validate_trade_entry(self, signal: TradeSignal, lot_size: float,
                           account_info: Dict, tick: Dict, is_recovery_trade: bool = False) -> Tuple[bool, str]:
        """
        Perform final validation before executing a trade.

        Args:
            signal: Trade signal
            lot_size: Calculated lot size
            account_info: Account information
            tick: Current tick data
            is_recovery_trade: If True, bypass position direction check (for hedges/DCA/zone recovery)

        Returns:
            Tuple of (can_enter, reason)
        """
        # Check global trade cooldown
        # [CRITICAL FIX] Bypass cooldown for Recovery Trades (Hedges/DCA).
        # We cannot wait 30s when the house is on fire.
        if not is_recovery_trade:
             time_since_last_trade = time.time() - self.last_trade_time
             if time_since_last_trade < self.config.global_trade_cooldown:
                 return False, f"Global cooldown: {time_since_last_trade:.1f}s < {self.config.global_trade_cooldown}s"

        # Check account equity
        equity = account_info.get('equity', 0)
        if equity <= 0:
            return False, "Invalid account equity"

        # Check if algo trading is allowed
        if not self.broker.is_trade_allowed():
            return False, "Algo trading disabled"

        # [AI INTELLIGENCE] Check News/Time Filter
        # Prevent entries during high-impact news events or low-liquidity zones
        if self.news_filter:
            is_safe, news_reason = self.news_filter.check_status()
            if not is_safe:
                return False, f"Entry Blocked: {news_reason}"

        # Check for recent bucket closes
        symbol = signal.symbol
        if self.position_manager.is_bucket_closed_recently(f"{symbol}_recent", 15.0):
            return False, "Recent bucket close - cooldown active"

        # CRITICAL: Prevent adding to existing positions in same direction
        # (unless it's a recovery trade for hedging/DCA/zone recovery)
        if not is_recovery_trade:
            existing_positions = self.broker.get_positions(symbol)
            if existing_positions is None: # [CRITICAL FIX] Broker Error Check
                logger.warning(f"[SAFETY] Failed to check existing positions for {symbol} - Blocking Entry")
                return False, "Broker Error: Could not check positions"
                
            if existing_positions:
                    if pos.type == signal_direction:
                        return False, f"Position already exists in {signal.action.value} direction - awaiting TP/hedge trigger"

        # [PHASE 5] The Treasury: Shadow Balance & Valkyrie Check
        if not is_recovery_trade:
             # Calculate Risk Metrics
             risk_metrics = {
                 "balance": account_info.get('balance', 0.0),
                 "equity": account_info.get('equity', 0.0),
                 "margin": account_info.get('margin', 0.0),
                 "total_positions": self.position_manager.get_total_positions(),
                 "total_exposure_pct": (account_info.get('margin', 0.0) / account_info.get('balance', 1.0)) if account_info.get('balance', 1.0) > 0 else 0.0,
                 "account_drawdown_pct": self.position_manager._calculate_drawdown(account_info.get('balance', 0.0), account_info.get('equity', 0.0))
             }
             
             veto, veto_reason = self._governor.veto(risk_metrics)
             if veto:
                 return False, f"Risk Governor Veto: {veto_reason}"

             # [PHASE 5] Supreme Court: Global Cap & Dynamic Layers
             # Note: validate_trade_entry is for NEW ENTRIES ("OPEN")
             approved, reason = self.authority.check_constitution(
                 self.broker, signal.symbol, lot_size, "OPEN"
             )
             if not approved:
                 return False, f"Unconstitutional: {reason}"

        # --- HFT LAYER 7: ORDER BOOK IMBALANCE (OBI) FILTER (INTELLIGENCE ONLY) ---
        # OBI and Direction Validator are used for GUIDANCE, NOT BLOCKING
        # They help choose direction and adjust lot size, but NEVER prevent trades
        
        obi_applicable = bool(tick.get('obi_applicable', False))
        obi_ok = bool(tick.get('obi_ok', False))
        
        # Get Direction Validator confidence for logging
        validation_score = signal.metadata.get('validation_score', 0.5)
        
        # Log OBI status for information only (NO BLOCKING)
        if obi_applicable and not obi_ok:
            logger.debug(f"[OBI_INFO] OBI unavailable, using Direction Validator ({validation_score:.0%}) for guidance")
        
        # If OBI is available, log it but DON'T block (only inform)
        if obi_ok:
            obi = float(tick.get('obi', 0.0) or 0.0)
            if abs(obi) > 0.3:  # Significant imbalance detected
                if signal.action == TradeAction.BUY and obi < -0.3:
                    logger.warning(f"[OBI_INFO] Sell pressure detected (OBI: {obi:.2f}) but proceeding with BUY (Validation: {validation_score:.0%})")
                elif signal.action == TradeAction.SELL and obi > 0.3:
                    logger.warning(f"[OBI_INFO] Buy pressure detected (OBI: {obi:.2f}) but proceeding with SELL (Validation: {validation_score:.0%})")
                else:
                    logger.debug(f"[OBI_INFO] OBI aligned with signal (OBI: {obi:.2f}, Validation: {validation_score:.0%})")

        # ALWAYS return True - NEVER block trades
        # Direction and lot size are already adjusted by Direction Validator
        return True, "Trade entry validated (intelligence-guided)"

    async def execute_trade_entry(self, signal: TradeSignal, lot_size: float,
                          tick: Dict, strategist, shield) -> Optional[Dict]:
        """
        Execute a trade entry order.

        Args:
            signal: Trade signal
            lot_size: Position size
            tick: Current tick data
            strategist: Strategist instance
            shield: IronShield instance

        Returns:
            Order result dict or None if failed
        """
        global logger  # Ensure logger is accessible in exception handler
        try:
            strict_entry = bool(self._strict_entry)

            # Extract signal metadata for logging and logic
            nexus_conf = signal.metadata.get('nexus_signal_confidence', 0.0)
            trend_strength = signal.metadata.get('trend_strength', 0.0)
            market_status = signal.metadata.get('market_status', 'Unknown')
            caution_factor = signal.metadata.get('caution_factor', 1.0)
            sentiment = signal.metadata.get('sentiment_score', 0.0)
            regime_name = signal.metadata.get('regime', 'UNKNOWN')

            # Determine entry price and order type
            if signal.action == TradeAction.BUY:
                entry_price = tick['ask']
                order_type = "BUY"
            else:
                entry_price = tick['bid']
                order_type = "SELL"

            # Get dynamic TP parameters - USE ACTUAL ATR
            symbol = signal.symbol
            atr_value = 0.0
            try:
                atr_value = float(signal.metadata.get('atr', 0.0) or 0.0)
            except Exception:
                atr_value = 0.0

            if atr_value <= 0.0:
                if strict_entry and hasattr(self.market_data, 'calculate_atr_checked'):
                    atr_checked, ok, areason = self.market_data.calculate_atr_checked(symbol, 14)
                    if not ok or atr_checked is None:
                        logger.warning(f"[STRICT] Blocking entry: ATR unavailable ({areason})")
                        return None
                    atr_value = float(atr_checked)
                else:
                    atr_value = self.market_data.calculate_atr(symbol, 14) if hasattr(self.market_data, 'calculate_atr') else 0.0010

            rsi_value = None
            try:
                rsi_value = signal.metadata.get('rsi', None) if hasattr(signal, 'metadata') else None
            except Exception:
                rsi_value = None
            if rsi_value is None and strict_entry and hasattr(self.market_data, 'calculate_rsi_checked'):
                rsi_checked, ok, rreason = self.market_data.calculate_rsi_checked(symbol, 14)
                if not ok or rsi_checked is None:
                    logger.warning(f"[STRICT] Blocking entry: RSI unavailable ({rreason})")
                    return None
                rsi_value = float(rsi_checked)
                try:
                    signal.metadata['rsi'] = rsi_value
                except Exception:
                    pass

            obi_applicable = bool(tick.get('obi_applicable', False))
            obi_ok = bool(tick.get('obi_ok', False))
            if hasattr(signal, 'metadata'):
                try:
                    obi_ok = bool(signal.metadata.get('obi_ok', obi_ok))
                except Exception:
                    pass
            
            # [SMART OVERRIDE] Do not block on OBI if we have a signal
            if strict_entry and obi_applicable and not obi_ok:
                logger.info("[STRICT] OBI unavailable (applicable but not ok) - Proceeding with trade (AI Decision)")
                # return None  # DISABLED BLOCKING

            atr_ok = bool(atr_value is not None and float(atr_value) > 0.0)
            rsi_ok = bool(rsi_value is not None)
            
            # Modified strict check to be lenient on OBI
            strict_ok = (not strict_entry) or (atr_ok and rsi_ok)
            
            if strict_entry and not strict_ok:
                logger.warning(
                    f"[STRICT] Blocking entry: strict_ok=False (atr_ok={atr_ok} rsi_ok={rsi_ok})"
                )
                return None
            
            # Convert ATR to points based on symbol type
            if 'JPY' in symbol or 'XAU' in symbol:
                atr_points = atr_value * 100  # JPY and Gold pairs use 2 decimal places
                pip_multiplier = 100
            else:
                atr_points = atr_value * 10000  # Major forex pairs use 4 decimal places
                pip_multiplier = 10000
                
            zone_points, tp_points = shield.get_dynamic_params(atr_points)  # Convert to points
            tp_pips_entry = tp_points # Define for logging compatibility

            # Calculate broker-side TP price for instant execution
            # Use Shield's dynamic TP (configurable via settings.yaml)
            # NO SL - hedging strategy manages risk through zone recovery
            
            if order_type == "BUY":
                tp_price_broker = entry_price + (tp_points / pip_multiplier)
            else:  # SELL
                tp_price_broker = entry_price - (tp_points / pip_multiplier)
            
            # Round to correct precision
            digits = 2 if "JPY" in symbol or "XAU" in symbol else 5
            tp_price_broker = round(tp_price_broker, digits)

            # Execute order WITHOUT broker TP/SL for hedge strategy
            # CRITICAL: No SL on broker because hedges ARE the risk management!
            # Setting SL would close position before hedges trigger, defeating the strategy.
            # Bucket logic (Python-side 100ms monitoring) manages ALL exits including break-even.
            
            # === EQUITY CHECK BEFORE EXECUTION ===
            account_info = self.broker.get_account_info()
            logger.debug(f"[ACCOUNT_INFO] Retrieved: {account_info}")
            if not account_info:
                logger.error("[ACCOUNT_INFO] Failed to retrieve account information")
                return None
            
            if account_info.get('equity', 0) < 100:
                logger.error(f"[TRADE] Insufficient equity: {account_info.get('equity', 0)} < 100")
                return None
            
            # Estimate required margin
            contract_size = 100 if "XAU" in signal.symbol else 100000
            leverage = account_info.get('leverage', 100)
            required_margin = lot_size * contract_size * entry_price / leverage
            available_margin = account_info.get('margin_free', 0)
            
            if required_margin > available_margin:
                logger.error(f"[TRADE] Insufficient margin: Required {required_margin:.2f} > Available {available_margin:.2f}")
                return None
            
            # [SUPREME COURT] Constitution Check
            # Before we execute, we MUST ask the Trade Authority.
            # We access it via PositionManager because that's where the Authority lives.
            approved, reason = self.position_manager.trade_authority.check_constitution(self.broker, signal.symbol, lot_size, "OPEN")
            if not approved:
                logger.warning(f"⚖️ [TRADE ENTRY] BLOCKED by Supreme Court: {reason}")
                return None

            result = self.broker.execute_order(
                action="OPEN",
                symbol=signal.symbol,
                order_type=order_type,
                price=entry_price,
                volume=lot_size,
                sl=0.0,  # NO SL - hedging strategy manages risk
                tp=0.0,  # NO BROKER TP - Virtual TP managed by Python for nil latency
                strict_entry=bool(strict_entry),
                strict_ok=bool(strict_ok),
                atr_ok=bool(atr_ok),
                rsi_ok=bool(rsi_ok),
                obi_ok=bool(obi_ok) if obi_applicable else None,
                trace_reason=f"OPEN_ENTRY:{signal.reason[:24]}",
                # MT5 comment limit: 31 chars
                comment=f"{signal.reason[:20]}"
            )
            
            # Log broker targets (actually set on broker for single positions)
            if result and result.get('ticket'):
                logger.info(f"[ORDER] VIRTUAL TP: {tp_price_broker:.5f} (Dynamic) | NO SL - Hedging Strategy")
                logger.info(f"[ORDER] Risk Management: Virtual TP for instant exits | Hedges manage downside risk")

            if result and result.get('ticket'):
                # Update statistics
                self.session_stats["trades_opened"] += 1

                # Record trade in database
                if self.db_queue:
                    trade_data = TradeData(
                        ticket=result['ticket'],
                        symbol=signal.symbol,
                        trade_type=order_type,
                        volume=lot_size,
                        open_price=entry_price,
                        close_price=None,
                        profit=None,
                        open_time=time.time(),
                        close_time=None,
                        strategy_reason=signal.reason
                    )
                    await self.db_queue.add_trade(trade_data)

                # Record learning data with entry TP/SL metadata
                ticket = result['ticket']

                # Risk intelligence calculations (AI-Adjusted)
                # OPTIMIZED GAPS: Linear expansion to prevent tight whipsaws in Hedge 2
                # Ask PPO for dynamic zone modifier to align Plan with Execution
                zone_mod = 1.0
                if self.ppo_guardian:
                    try:
                        # Use 0 drawdown for initial plan
                        _, zone_mod = self.ppo_guardian.get_dynamic_zone(0.0, atr_value, trend_strength, nexus_conf)
                    except Exception as e:
                        logger.warning(f"[PPO] Failed to get dynamic zone: {e}")
                        zone_mod = 1.0

                self.position_manager.record_learning_trade(ticket, signal.symbol, {
                    "symbol": signal.symbol,
                    "type": order_type,
                    "entry_price": entry_price,
                    "obs": [
                        0.0,  # drawdown_pips at entry
                        float(atr_value) if atr_value is not None else 0.0,  # ATR (price units)
                        float(trend_strength) if trend_strength is not None else 0.0,  # trend strength
                        float(signal.metadata.get("nexus_signal_confidence", 0.0) or 0.0),  # nexus/conf
                    ],
                    "action": [1.0, float(zone_mod)],  # hedge_mult, zone_mod (zone_mod may be PPO-derived)
                    "open_time": time.time(),
                    # Store entry targets for Python-side monitoring
                    "entry_tp_pips": tp_pips_entry,
                    "entry_sl_pips": 0.0,  # NO SL - hedging strategy only
                    "entry_atr": atr_value
                })

                # === ENHANCED AI INTELLIGENCE TRADING PLAN LOGGING ===
                # Comprehensive AI decision factors and execution intelligence
                symbol = signal.symbol
                # Reuse ATR/regime already computed for this entry.
                
                # Convert ATR to pips based on symbol type
                if "JPY" in symbol or "XAU" in symbol:
                    atr_pips = atr_value * 100  # JPY and Gold pairs use 2 decimal places
                else:
                    atr_pips = atr_value * 10000  # Major forex pairs use 4 decimal places

                # Define pip divisor based on symbol for correct price calculation
                pip_divisor = 100 if "JPY" in symbol or "XAU" in symbol else 10000
                point_approx = 1.0 / (pip_divisor * 10) # Approx point size for spread calc

                tp_pips = atr_pips * ScalpingConfig.TP_ATR_MULTIPLIER  # 30% of ATR for tight scalping
                tp_price_dist = tp_pips / pip_divisor
                tp_price = entry_price + tp_price_dist if order_type == "BUY" else entry_price - tp_price_dist

                # === UNIFIED ZONE RECOVERY PLAN ===
                # Calculate Zone Width (Fixed for the sequence)
                # Based on Hedge 1 distance (0.5 ATR)
                zone_pips = atr_pips * ScalpingConfig.HEDGE1_ATR_MULTIPLIER * zone_mod
                zone_price_dist = zone_pips / pip_divisor
                
                # Get Spread for Lot Calculation (Use the live tick that triggered this entry)
                props = self._get_symbol_properties(symbol)
                point_size = float(props.get('point_size', point_approx) or point_approx)
                spread_points = float((tick['ask'] - tick['bid']) / max(1e-12, point_size))
                
                # Calculate Lots using IronShield (Same as Execution)
                # Convert pips to points for IronShield
                zone_points = zone_pips * 10
                tp_points = tp_pips * 10
                
                # === AI BOOST: FUTURISTIC PREDICTION ===
                # If Oracle predicts a crash/rally, boost the hedge lots in that direction
                # to capitalize on the move and exit faster.
                ai_bias = 0.0
                oracle_prediction = signal.metadata.get('oracle_prediction', 'NEUTRAL')
                
                # Determine H1 direction (Opposite to Entry)
                h1_dir_check = "SELL" if order_type == "BUY" else "BUY"
                
                if h1_dir_check == "SELL" and oracle_prediction == "BEARISH":
                    ai_bias = 0.10 # +10% Aggression on Sell Hedges
                elif h1_dir_check == "BUY" and oracle_prediction == "BULLISH":
                    ai_bias = 0.10 # +10% Aggression on Buy Hedges

                # Calculate Hedge Lots Iteratively
                # Hedge 1 (Opposite)
                raw_h1 = self.risk_manager.shield.calculate_defense(
                    lot_size, spread_points, fixed_zone_points=zone_points, fixed_tp_points=tp_points,
                    oracle_prediction=oracle_prediction, hedge_level=1
                )
                hedge1_lot = round(raw_h1 * (1.0 + ai_bias), 4)
                
                # Hedge 2 (Recovery) - Recovers H1
                hedge2_lot = self.risk_manager.shield.calculate_defense(
                    hedge1_lot, spread_points, fixed_zone_points=zone_points, fixed_tp_points=tp_points,
                    oracle_prediction=oracle_prediction, hedge_level=2
                )
                
                # === SMART EXPANSION LOGIC ===
                # Determine Expansion Factor based on AI Prediction
                ai_expansion_factor = 0.3 # Default: Hedge 3 is 30% further out
                
                # Check Oracle Prediction (if available in metadata)
                oracle_prediction = signal.metadata.get('oracle_prediction', 'NEUTRAL')
                
                # If we are Buying (H1 is Sell), and AI says Bullish (Reversal), delay the Sell Hedge 3
                if order_type == "BUY" and oracle_prediction == "BULLISH":
                    ai_expansion_factor = 0.6 # Push it 60% away (Give it room to breathe)
                # If we are Selling (H1 is Buy), and AI says Bearish (Reversal), delay the Buy Hedge 3
                elif order_type == "SELL" and oracle_prediction == "BEARISH":
                    ai_expansion_factor = 0.6

                # Calculate Smart Buffer
                smart_buffer_pips = (zone_pips * ai_expansion_factor)
                smart_buffer_dist = smart_buffer_pips / pip_divisor
                
                # Hedge 3 (Opposite) - Same direction as H1
                # We boost the zone size for H3 calculation to account for the extra distance
                # Effective Zone for H3 = Zone + Buffer
                expanded_zone_points = zone_points * (1.0 + ai_expansion_factor)
                
                raw_h3 = self.risk_manager.shield.calculate_defense(
                    hedge2_lot, spread_points, fixed_zone_points=expanded_zone_points, fixed_tp_points=tp_points,
                    oracle_prediction=oracle_prediction, hedge_level=3
                )
                hedge3_lot = round(raw_h3 * (1.0 + ai_bias), 4)
                
                # Hedge 4 (Recovery) - Recovers H3
                # H4 recovers H3 over the expanded zone distance
                hedge4_lot = self.risk_manager.shield.calculate_defense(
                    hedge3_lot, spread_points, fixed_zone_points=expanded_zone_points, fixed_tp_points=tp_points,
                    oracle_prediction=oracle_prediction, hedge_level=4
                )

                # Calculate Hedge Prices (Smart Expansion Logic)
                if order_type == "BUY":
                    # Initial Buy
                    # Hedge 1 (Sell): Entry - Zone
                    hedge1_price = entry_price - zone_price_dist
                    h1_dir = "SELL"
                    
                    # Hedge 2 (Buy): Entry (Recovery)
                    hedge2_price = entry_price
                    h2_dir = "BUY"
                    
                    # Hedge 3 (Sell): H1 - Smart Buffer (Expansion)
                    # For BUY Initial: H1=SELL, H2=BUY. H3 should be SELL (Recover H2)
                    hedge3_price = hedge1_price - smart_buffer_dist
                    h3_dir = "SELL"
                    
                    # Hedge 4 (Buy): Matches H2 (Entry)
                    hedge4_price = hedge2_price
                    h4_dir = "BUY"
                    
                else: # SELL
                    # Initial Sell
                    # Hedge 1 (Buy): Entry + Zone
                    hedge1_price = entry_price + zone_price_dist
                    h1_dir = "BUY"
                    
                    # Hedge 2 (Sell): Entry (Recovery)
                    hedge2_price = entry_price
                    h2_dir = "SELL"
                    
                    # Hedge 3 (Buy): H1 + Smart Buffer (Expansion)
                    hedge3_price = hedge1_price + smart_buffer_dist
                    h3_dir = "BUY"
                    
                    # Hedge 4 (Sell): Matches H2 (Entry)
                    hedge4_price = hedge2_price
                    h4_dir = "SELL"

                # Log Smart Expansion Plan
                logger.info(f"[SMART EXPANSION] H3 Buffer: {ai_expansion_factor*100:.0f}% ({smart_buffer_pips:.1f} pips) | AI: {oracle_prediction}")

                # Calculate projected hedge levels for display AND execution
                projected_hedges = []
                
                # Hedge 1
                projected_hedges.append({
                    'direction': h1_dir,
                    'trigger_price': round(hedge1_price, 5),
                    'lots': hedge1_lot,
                    'tp_price': "Break-even"
                })
                
                # Hedge 2
                projected_hedges.append({
                    'direction': h2_dir,
                    'trigger_price': round(hedge2_price, 5),
                    'lots': hedge2_lot,
                    'tp_price': "Break-even"
                })

                # Hedge 3
                projected_hedges.append({
                    'direction': h3_dir,
                    'trigger_price': round(hedge3_price, 5),
                    'lots': hedge3_lot,
                    'tp_price': "Break-even"
                })
                
                # Hedge 4
                projected_hedges.append({
                    'direction': h4_dir,
                    'trigger_price': round(hedge4_price, 5),
                    'lots': hedge4_lot,
                    'tp_price': "Break-even"
                })

                # Log using the new UI Logger
                TradingLogger.log_initial_trade(f"{symbol}_{0 if order_type=='BUY' else 1}", {
                    'action': order_type,
                    'symbol': symbol,
                    'lots': lot_size,
                    'entry_price': entry_price,
                    'tp_price': tp_price_broker,
                    'tp_atr': 0.3,
                    'tp_pips': tp_pips_entry,
                    'hedges': projected_hedges,
                    'atr_pips': atr_pips,
                    'reasoning': f"[{regime_name}] {signal.reason}"
                })

                # === STRUCTURED TRADING PLAN LOG ===
                # Use TradingLogger for clean, structured output
                
                # Prepare data for TradingLogger
                bucket_id = f"{symbol}_{int(time.time())}"
                
                # Store metadata for this trade in position manager (for AI learning and exit logic)
                if result and result.get('ticket'):
                    ticket = result['ticket']
                    atr_pips = atr_value * (100 if "XAU" in symbol or "GOLD" in symbol else 10000)
                    
                    # Use thread-safe persistence method
                    self.position_manager.record_trade_metadata(ticket, {
                        "symbol": symbol,
                        "type": order_type,
                        "entry_price": entry_price,
                        "entry_tp_pips": tp_pips,  # Store fixed TP target
                        "entry_sl_pips": 0.0,  # NO SL - hedging strategy only
                        "entry_atr": atr_value,  # Store ATR at entry
                        "obs": [0.0, atr_value, trend_strength, nexus_conf],  # AI observation
                        "action": [1.0, 0.8],  # Default hedge params
                        "open_time": time.time(),
                        "hedge_plan": {
                            "hedge1_trigger_price": hedge1_price,
                            "hedge2_trigger_price": hedge2_price,
                            "hedge3_trigger_price": hedge3_price,
                            "hedge4_trigger_price": hedge4_price,
                            "hedge1_lots": hedge1_lot,
                            "hedge2_lots": hedge2_lot,
                            "hedge3_lots": hedge3_lot,
                            "hedge4_lots": hedge4_lot,
                            "zone_width_pips": format_pips(zone_pips, symbol)
                        }
                    })
                    logger.debug(f"[METADATA] Stored entry data for ticket #{ticket}: TP={tp_pips:.1f}pips, ATR={atr_pips:.1f}pips (No SL - Hedge Strategy)")
                
                return result
            else:
                logger.error("Trade execution failed")
                return None

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None

    async def process_position_management(self, symbol: str, positions: List[Dict],
                                  tick: Dict, point: float, shield, ppo_guardian,
                                  rsi_value: float = 50.0, oracle=None, pressure_metrics=None, trap_hunter=None) -> bool:
        """
        Process position management for a symbol.

        Args:
            symbol: Trading symbol
            positions: List of positions
            tick: Current tick data
            point: Point value
            shield: IronShield instance
            ppo_guardian: PPO Guardian instance

            oracle: Optional Oracle instance (Layer 4)
            pressure_metrics: Optional Tick Pressure Metrics
            trap_hunter: Optional TrapHunter instance for fakeout detection

        Returns:
            True if positions were managed (closed/opened)
        """
        # [ROBUSTNESS] Ensure all positions are dicts before processing
        # This prevents TypeError: 'Position' object is not subscriptable in RiskManager
        # [ROBUSTNESS] Ensure all positions are dicts before processing
        # This prevents TypeError: 'Position' object is not subscriptable in RiskManager
        positions = normalize_positions(positions)

        logger.debug(f"[POS_MGMT] Called for {symbol} with {len(positions)} positions")
        
        if not positions:
            return False
        
        # OPTIMIZATION: Removed redundant broker.get_positions() call.
        # The caller (_process_existing_positions) already synced with broker.
        # We just need to filter out any known ghost tickets.
        
        # Filter out ghost tickets
        positions = [p for p in positions if p['ticket'] not in self.position_manager._ghost_tickets]
        
        if not positions:
            return False

        # [CRITICAL FIX] Update PositionManager with latest broker data (Profit/Price)
        # This ensures 'should_close_bucket' uses real-time PnL, not stale data.
        self.position_manager.update_positions(positions)

        # Only log when we have multiple positions (actual bucket)
        if len(positions) > 1:
            logger.debug(f"[BUCKET] Checking exit for {symbol}: {len(positions)} positions")

        # Check for bucket exits first
        bucket_closed = False

        # Find or create bucket for positions
        position_tickets = [p['ticket'] for p in positions]
        bucket_id = self.position_manager.find_bucket_by_tickets(position_tickets)
        
        if not bucket_id:
            # No existing bucket - create new one
            position_objects = [
                self.position_manager.active_positions.get(p['ticket'])
                for p in positions
                if p['ticket'] in self.position_manager.active_positions
            ]
            
            if position_objects:
                bucket_id = self.position_manager.create_bucket(position_objects)
                if len(positions) > 1:
                    logger.debug(f"[BUCKET] CREATED NEW: {bucket_id} with {len(positions)} positions")
                else:
                    logger.debug(f"[BUCKET] CREATED NEW: {bucket_id} for single position (enables TP/Zone tracking)")
        else:
            logger.debug(f"[BUCKET] REUSING EXISTING: {bucket_id} for {len(positions)} positions")

        # [VALKYRIE PROTOCOL] Check for Frozen State
        # If frozen, we do NOT touch it. No TP, no SL moves, no hedges.
        if bucket_id and bucket_id in self.position_manager.bucket_stats:
             # Access String Value safely or Enum
             state = self.position_manager.bucket_stats[bucket_id].state
             # Check against Enum value (bucket_frozen)
             if str(state.value) == "bucket_frozen": 
                 # Log only once every minute to reduce spam
                 if time.time() - getattr(self, f"_last_frozen_log_{bucket_id}", 0) > 60:
                     logger.info(f"❄️ [VALKYRIE] Skipping management for FROZEN bucket {bucket_id}. Waiting for cleanup.")
                     setattr(self, f"_last_frozen_log_{bucket_id}", time.time())
                 return False

        # Prepare market data for intelligent scalping analysis (MOVED UP)
        atr_value = self.market_data.calculate_atr(symbol, 14)
        trend_strength = self.market_data.calculate_trend_strength(symbol, 20)
        trend_direction = 0.0
        rsi_val = rsi_value

        atr_ok = True
        trend_ok = True
        trend_dir_ok = True
        rsi_ok = True

        if self._strict_entry:
            # For any logic that can open NEW orders (DCA/calculated recovery/zone recovery),
            # ensure indicators are computed from real candles.
            if hasattr(self.market_data, 'calculate_atr_checked'):
                a, ok, _ = self.market_data.calculate_atr_checked(symbol, 14)
                atr_ok = bool(ok and a is not None)
                if atr_ok:
                    atr_value = float(a)
            if hasattr(self.market_data, 'calculate_trend_strength_checked'):
                t, ok, _ = self.market_data.calculate_trend_strength_checked(symbol, 20)
                trend_ok = bool(ok and t is not None)
                if trend_ok:
                    trend_strength = float(t)
            if hasattr(self.market_data, 'calculate_trend_direction_checked'):
                td, ok, _ = self.market_data.calculate_trend_direction_checked(symbol, 20)
                trend_dir_ok = bool(ok and td is not None)
                if trend_dir_ok:
                    trend_direction = float(td)
            if hasattr(self.market_data, 'calculate_rsi_checked'):
                r, ok, _ = self.market_data.calculate_rsi_checked(symbol, 14)
                rsi_ok = bool(ok and r is not None)
                if rsi_ok:
                    rsi_val = float(r)
        else:
            if hasattr(self.market_data, 'calculate_trend_direction'):
                try:
                    trend_direction = float(self.market_data.calculate_trend_direction(symbol))
                except Exception:
                    trend_direction = 0.0

        # [GOD MODE] Fetch candles for Structure Analysis
        candles = self.market_data.candles.get_history(symbol)

        # [GOD MODE] Fetch Equity for Emergency Trigger
        # Essential to prevent false triggers on default 1000.0 equity
        account_equity = 1000.0
        try:
             acct = self.broker.get_account_info()
             if acct:
                 account_equity = acct.get('equity', 1000.0)
        except Exception as e:
             logger.warning(f"[GOD MODE] Failed to fetch equity: {e}")

        ok_fresh, fresh_reason = self._freshness_ok_for_new_orders(tick)
        
        market_data = {
            'equity': account_equity, # [CRITICAL FIX] Pass real equity
            'atr': atr_value,  # ATR in points
            'spread': abs(tick['ask'] - tick['bid']),  # Spread in points
            'trend_strength': trend_strength,  # Trend strength 0-1
            'trend_direction': trend_direction,  # Signed trend direction (-1..+1)
            'current_price': (tick['ask'] + tick['bid']) / 2,
            'bid': tick['bid'], # Explicit Bid
            'ask': tick['ask'], # Explicit Ask
            'time': tick.get('time', 0.0),
            'tick_age_s': float(tick.get('tick_age_s', float('inf'))),
            'candle_close_age_s': float(tick.get('candle_close_age_s', float('inf'))),
            'timeframe_s': int(tick.get('timeframe_s', 60) or 60),
            'fresh_ok': bool(ok_fresh),
            'fresh_reason': str(fresh_reason),
            'point': point,  # Point value for pip calculations
            'candles': candles, # Pass full history for structure analysis
            'rsi': rsi_val, # Add RSI here
            'strict_entry': bool(self._strict_entry),
            'atr_ok': bool(atr_ok),
            'trend_ok': bool(trend_ok),
            'trend_dir_ok': bool(trend_dir_ok),
            'rsi_ok': bool(rsi_ok),
        }
        
        ai_context = {
            'pressure_metrics': pressure_metrics,
            'pressure_dominance': pressure_metrics.get('pressure_dominance', 0.0) if pressure_metrics else 0.0
        }

        # [AI SNIPER] Check for Smart Unwind Opportunity
        if oracle and len(positions) > 1 and bucket_id:
             # Prepare market data history for Oracle
             history = self.market_data.candles.get_history(symbol)
             # Ensure we have enough history
             if len(history) >= 60: # Increased to 60 for Transformer
                 # Pass full history (candles) to PositionManager for v5.5.0 Oracle Logic
                 unwound = await self.position_manager.execute_ai_sniper_logic(
                     bucket_id, oracle, self.broker, history, symbol
                 )
                 if unwound:
                     logger.info(f"[SNIPER] Smart Unwind executed for {bucket_id}")
                     return True # Positions changed

        # [HIGHEST INTELLIGENCE] LAYER 2: THE ERASER (Tactical De-Risking)
        if len(positions) > 1 and bucket_id:
            # Pass market_data and ai_context to Eraser
            erased = await self.position_manager.execute_eraser_logic(bucket_id, self.broker, market_data, ai_context)
            if erased:
                logger.info(f"[ERASER] Tactical De-Risking executed for {bucket_id}")
                return True # Positions changed

        # Check if bucket should be closed
        # (market_data is already prepared)

        # Periodic position status update (every 5 seconds) to show AI is monitoring
        current_time = time.time()
        if current_time - self._last_position_status_time >= 5.0:
            self._last_position_status_time = current_time
            first_pos = positions[0]
            entry_price = first_pos.get('price_open', 0)
            current_price = market_data['current_price']
            
            # Calculate pips correctly: XAUUSD uses 100 (1 pip = 0.01), forex uses 10000 (1 pip = 0.0001)
            pip_multiplier = 100 if "XAU" in symbol or "GOLD" in symbol else 10000
            atr_pips = atr_value * pip_multiplier  # ATR in pips
            
            # Check for stored TP in metadata
            tp_pips = atr_pips * 0.3  # Default dynamic TP
            tp_source = "Dynamic 0.3 ATR"
            next_hedge_info = None
            
            try:
                ticket = first_pos.get('ticket') if isinstance(first_pos, dict) else first_pos.ticket
                if ticket in self.position_manager.active_learning_trades:
                    metadata = self.position_manager.active_learning_trades[ticket]
                    stored_tp = metadata.get('entry_tp_pips', 0.0)
                    if stored_tp > 0:
                        tp_pips = stored_tp
                        tp_source = "Stored"
                        
                    # Extract Next Hedge Info
                    hedge_plan = metadata.get('hedge_plan', {})
                    pos_count = len(positions)
                    if pos_count < 5: # Max 4 hedges
                        next_hedge_key = f"hedge{pos_count}" # e.g. hedge1 if 1 pos exists
                        trigger_price = hedge_plan.get(f"{next_hedge_key}_trigger_price")
                        lots = hedge_plan.get(f"{next_hedge_key}_lots")
                        if trigger_price:
                             next_hedge_info = {'price': trigger_price, 'lots': lots, 'type': 'PENDING'}
            except Exception as e:
                logger.debug(f"[STATUS] Failed reading stored TP/hedge plan metadata: {e}", exc_info=True)
            
            # Calculate current P&L in pips
            if first_pos.get('type') == 0:  # BUY
                pnl_pips = (current_price - entry_price) * pip_multiplier
            else:  # SELL
                pnl_pips = (entry_price - current_price) * pip_multiplier
            
            # Use new Dashboard Logger
            ai_notes = f"RSI {rsi_value:.1f}"
            
            # Get zone boundaries from bucket
            upper_level = 0
            lower_level = 0
            try:
                if hasattr(self.position_manager, 'get_bucket_metadata'):
                    bucket_meta = self.position_manager.get_bucket_metadata(bucket_id)
                    if bucket_meta:
                        upper_level = bucket_meta.get('upper_level', 0)
                        lower_level = bucket_meta.get('lower_level', 0)
            except Exception as e:
                logger.debug(f"Could not get zone boundaries: {e}")
            
            # [INTELLIGENT AI COMMUNICATION] Context-aware, grounded reasoning
            # Track last logged state to avoid spam
            if not hasattr(self, '_last_ai_state'):
                self._last_ai_state = {}
            
            # Create state signature for this position
            state_key = f"{symbol}_{bucket_id}"
            current_state = {
                'pnl_pips': round(pnl_pips, 1),
                'rsi': round(rsi_value, 1) if rsi_value else 0,
                'price_zone': 'ABOVE' if current_price > upper_level else 'BELOW' if current_price < lower_level else 'INSIDE'
            }
            
            last_state = self._last_ai_state.get(state_key, {})
            
            # === INTELLIGENT DECISION: WHAT TO COMMUNICATE ===
            # Determine what's most important RIGHT NOW based on context
            
            # 1. Check for critical situations (always communicate)
            zone_status = current_state['price_zone']
            distance_to_upper = (upper_level - current_price) * pip_multiplier if upper_level else 0
            distance_to_lower = (current_price - lower_level) * pip_multiplier if lower_level else 0
            
            is_critical = (
                abs(distance_to_upper) < 10 or abs(distance_to_lower) < 10 or  # Near zone boundary
                zone_status != last_state.get('price_zone', '') or  # Zone change
                abs(current_state['pnl_pips'] - last_state.get('pnl_pips', 0)) > 50 or  # Large P&L swing
                rsi_value > 80 or rsi_value < 20  # Extreme RSI
            )
            
            # 2. Check for significant changes (communicate periodically)
            # [DAMPENING] Only log significant non-critical changes every 60 seconds
            now = time.time()
            last_log_ts = last_state.get('last_log_ts', 0)
            
            is_significant = (
                (abs(current_state['pnl_pips'] - last_state.get('pnl_pips', 0)) > 20 or
                 abs(current_state['rsi'] - last_state.get('rsi', 0)) > 15) and
                (now - last_log_ts > 60.0) # Dampener
            ) or (not last_state) # First time
            
            should_log = is_critical or is_significant
            
            if should_log:
                # === GROUNDED ANALYSIS: Build message based on REAL market context ===
                
                # Determine PRIMARY focus (what matters most right now)
                if zone_status == 'ABOVE' and abs(distance_to_upper) > 50:
                    # CRITICAL: Price way outside zone
                    focus = "HEDGE_IMMINENT"
                    message = (
                        f"🚨 CRITICAL ALERT - {symbol}\n"
                        f"Price has broken {abs(distance_to_upper):.1f} pips ABOVE our zone boundary.\n"
                        f"Current: {current_price:.5f} | Zone Top: {upper_level:.5f}\n\n"
                        f"🧠 AI Analysis:\n"
                        f"Market showing strong upward momentum. RSI at {rsi_value:.1f} "
                        f"{'- OVERBOUGHT, reversal expected soon' if rsi_value > 70 else '- momentum still building'}.\n\n"
                    )
                    
                    # Get Oracle prediction for grounding
                    try:
                        if hasattr(self, 'oracle') and self.oracle:
                            pred = self.oracle.predict(symbol)
                            if pred:
                                oracle_says = pred.get('prediction', 'NEUTRAL')
                                conf = pred.get('confidence', 0)
                                if oracle_says == 'UP':
                                    message += f"🔮 Oracle confirms: Expects FURTHER upside ({conf:.0%} confidence)\n"
                                    message += f"⚡ Decision: HEDGE NOW to protect against continued rise\n"
                                elif oracle_says == 'DOWN':
                                    message += f"🔮 Oracle predicts: Reversal coming ({conf:.0%} confidence)\n"
                                    message += f"⚡ Decision: WAIT for reversal - hedge only if continues higher\n"
                                else:
                                    message += f"🔮 Oracle sees: Consolidation likely\n"
                                    message += f"⚡ Decision: Monitor closely - hedge if breaks higher\n"
                    except:
                        message += f"⚡ Decision: Hedge trigger at {abs(distance_to_upper) + 10:.0f} pips\n"
                    
                elif zone_status == 'BELOW' and abs(distance_to_lower) > 50:
                    # CRITICAL: Price way outside zone (downside)
                    focus = "HEDGE_IMMINENT"
                    message = (
                        f"🚨 CRITICAL ALERT - {symbol}\n"
                        f"Price has broken {abs(distance_to_lower):.1f} pips BELOW our zone boundary.\n"
                        f"Current: {current_price:.5f} | Zone Bottom: {lower_level:.5f}\n\n"
                        f"🧠 AI Analysis:\n"
                        f"Market showing strong downward pressure. RSI at {rsi_value:.1f} "
                        f"{'- OVERSOLD, bounce expected' if rsi_value < 30 else '- selling continues'}.\n\n"
                    )
                    
                    try:
                        if hasattr(self, 'oracle') and self.oracle:
                            pred = self.oracle.predict(symbol)
                            if pred:
                                oracle_says = pred.get('prediction', 'NEUTRAL')
                                conf = pred.get('confidence', 0)
                                if oracle_says == 'DOWN':
                                    message += f"🔮 Oracle confirms: Expects FURTHER downside ({conf:.0%} confidence)\n"
                                    message += f"⚡ Decision: HEDGE NOW to protect against continued fall\n"
                                elif oracle_says == 'UP':
                                    message += f"🔮 Oracle predicts: Bounce coming ({conf:.0%} confidence)\n"
                                    message += f"⚡ Decision: WAIT for bounce - hedge only if continues lower\n"
                    except:
                        message += f"⚡ Decision: Hedge trigger at {abs(distance_to_lower) + 10:.0f} pips\n"
                
                elif pnl_pips > 20:
                    # POSITIVE: In profit
                    focus = "PROFIT_MANAGEMENT"
                    message = (
                        f"✅ PROFIT UPDATE - {symbol}\n"
                        f"Position now +{pnl_pips:.1f} pips (${sum(p.get('profit', 0) for p in positions):+.2f})\n"
                        f"Target: +{tp_pips:.1f} pips | Remaining: {tp_pips - pnl_pips:.1f} pips\n\n"
                        f"🧠 AI Analysis:\n"
                        f"Price at {current_price:.5f}, safely inside zone. "
                        f"RSI {rsi_value:.1f} shows {'bullish momentum continuing' if rsi_value > 50 else 'momentum weakening'}.\n\n"
                    )
                    
                    try:
                        if hasattr(self, 'oracle') and self.oracle:
                            pred = self.oracle.predict(symbol)
                            if pred:
                                oracle_says = pred.get('prediction', 'NEUTRAL')
                                if oracle_says == 'UP':
                                    message += f"🔮 Oracle: Predicts continued upside - HOLD for full TP\n"
                                elif oracle_says == 'DOWN':
                                    message += f"🔮 Oracle: Reversal warning - Consider taking profit early\n"
                                else:
                                    message += f"🔮 Oracle: Consolidation expected - HOLD current position\n"
                    except:
                        message += f"⚡ Plan: Hold for TP target\n"
                
                elif pnl_pips < -100:
                    # NEGATIVE: Significant drawdown
                    focus = "RECOVERY_MODE"
                    message = (
                        f"🔄 RECOVERY STATUS - {symbol}\n"
                        f"Drawdown: {pnl_pips:.1f} pips (${sum(p.get('profit', 0) for p in positions):.2f})\n"
                        f"Price: {current_price:.5f} | Zone: [{lower_level:.5f} - {upper_level:.5f}]\n\n"
                        f"🧠 AI Analysis:\n"
                        f"Position underwater but price {'inside' if zone_status == 'INSIDE' else 'outside'} recovery zone. "
                        f"RSI {rsi_value:.1f} {'suggests reversal coming' if (rsi_value < 30 or rsi_value > 70) else 'neutral'}.\n\n"
                    )
                    
                    try:
                        if hasattr(self, 'oracle') and self.oracle:
                            pred = self.oracle.predict(symbol)
                            if pred:
                                oracle_says = pred.get('prediction', 'NEUTRAL')
                                trajectory = pred.get('trajectory', [])
                                if oracle_says == 'UP' and first_pos.get('type') == 0:  # BUY position
                                    message += f"🔮 Oracle: Predicts recovery ({trajectory[:2] if trajectory else 'N/A'} pips)\n"
                                    message += f"⚡ Plan: WAIT for price recovery - no hedge needed yet\n"
                                elif oracle_says == 'DOWN' and first_pos.get('type') == 1:  # SELL position
                                    message += f"🔮 Oracle: Predicts recovery\n"
                                    message += f"⚡ Plan: WAIT for price recovery\n"
                                else:
                                    message += f"🔮 Oracle: No immediate recovery signal\n"
                                    message += f"⚡ Plan: Monitor for zone breach - hedge if necessary\n"
                    except:
                        message += f"⚡ Plan: Monitoring for recovery or hedge trigger\n"
                
                else:
                    # NEUTRAL: Normal monitoring
                    focus = "MONITORING"
                    message = (
                        f"📊 Market Update - {symbol}\n"
                        f"P&L: {pnl_pips:+.1f} pips | RSI: {rsi_value:.1f} | Price: {current_price:.5f}\n"
                        f"Status: {'Inside zone - normal monitoring' if zone_status == 'INSIDE' else 'Outside zone - watching closely'}\n"
                    )
                
                # [LEGACY] Demoted to DEBUG - rely on Entry/Hedge/Exit reports only
                logger.debug(f"\n{message}")
                
                current_state['last_log_ts'] = time.time()
                self._last_ai_state[state_key] = current_state

        # Initialize bucket_closed to False
        bucket_closed = False
        
        # [AI PLAN B] Reversion Escape Check
        # If Plan B is active (Fakeout Detected) OR Risk Veto is active (Defensive Hold),
        # we exit as soon as we are profitable (Break-Even + small profit) to reduce risk.
        if getattr(self, f"_plan_b_active_{symbol}", False) or getattr(self, f"_plan_b_veto_{symbol}", False):
            # Calculate Net PnL (Profit + Swap + Commission)
            net_profit = sum(
                (p.get('profit', 0.0) if isinstance(p, dict) else p.profit) + 
                (p.get('swap', 0.0) if isinstance(p, dict) else p.swap) + 
                (p.get('commission', 0.0) if isinstance(p, dict) else p.commission) 
                for p in positions
            )
            
            # [INTELLIGENCE FIX] Dynamic Slippage Buffer
            # Fixed $0.50 is risky for larger lots. We need a buffer proportional to volume.
            # Aim for $10.00 per lot (approx 10 pips on Gold) to cover execution slippage.
            total_vol = sum(p.get('volume', 0.0) if isinstance(p, dict) else p.volume for p in positions)
            slippage_buffer = max(0.50, total_vol * 10.0)
            
            if net_profit > slippage_buffer: 
                logger.info(f"[PLAN B] Reversion Escape Successful! Net Profit: ${net_profit:.2f} (Buffer: ${slippage_buffer:.2f}). Closing all positions.")
                bucket_closed = await self.position_manager.close_bucket_positions(
                    self.broker,
                    bucket_id,
                    symbol,
                    trace={
                        'strict_entry': bool(self._strict_entry),
                        'strict_ok': True,
                        'atr_ok': bool(market_data.get('atr_ok', True)),
                        'rsi_ok': bool(market_data.get('rsi_ok', True)),
                        'obi_ok': bool(market_data.get('obi_ok', False)),
                        'reason': 'PLAN_B_REVERSION_ESCAPE',
                    },
                    ppo_guardian=ppo_guardian,
                )
                if bucket_closed:
                    setattr(self, f"_plan_b_active_{symbol}", False)
                    setattr(self, f"_plan_b_veto_{symbol}", False)
                    return True
        
        logger.debug(f"[TP_CHECK] About to call should_close_bucket for {bucket_id}")
        should_close, confidence = self.position_manager.should_close_bucket(
            bucket_id, ppo_guardian, market_data
        )
        logger.debug(f"[TP_CHECK] should_close_bucket returned: {should_close}, confidence: {confidence}")

        if should_close:
            logger.debug(f"[BUCKET] EXIT TRIGGERED: Confidence {confidence:.3f} - Closing all positions")
            bucket_closed = await self.position_manager.close_bucket_positions(
                self.broker,
                bucket_id,
                symbol,
                trace={
                    'strict_entry': bool(self._strict_entry),
                    'strict_ok': True,
                    'atr_ok': bool(market_data.get('atr_ok', True)),
                    'rsi_ok': bool(market_data.get('rsi_ok', True)),
                    'obi_ok': bool(market_data.get('obi_ok', False)),
                    'reason': 'TP_EXIT',
                },
                ppo_guardian=ppo_guardian,
            )
            if bucket_closed:
                logger.debug(f"[BUCKET] CLOSED SUCCESSFULLY: {bucket_id}")
            else:
                logger.warning(f"[BUCKET] CLOSE FAILED: {bucket_id}")
        # Only log if exit was checked and failed (for debugging)
        # Silent operation when no exit needed - reduces log spam

        # If bucket not closed, check for zone recovery (only log if executed)
        if not bucket_closed:
            # Freshness gate: allow exits, but block ANY NEW order when feed is stale
            if self._freshness_gate:
                ok_fresh, fresh_reason = self._freshness_ok_for_new_orders(tick)
                if not ok_fresh:
                    logger.warning(f"[FRESHNESS] NEW orders blocked (recovery/hedge): {fresh_reason}")
                    return bucket_closed

            # [PHASE 2] THE STABILIZER: Check for Hedging Trigger
            # If Drawdown > 1.5 * ATR, trigger hedge immediately
            stabilizer_triggered = self.position_manager.check_stabilizer_trigger(bucket_id, market_data)
            
            if stabilizer_triggered:
                logger.warning(f"[STABILIZER] Emergency Hedge Triggered for {bucket_id}")
                # Force Zone Recovery to execute a hedge
                # We do this by passing a flag or just letting the risk manager handle it
                # For now, we rely on the standard zone recovery logic but with heightened awareness
                # In future, we can call a specific self.risk_manager.execute_stabilizer_hedge()
            
            # [HIGHEST INTELLIGENCE] LAYER 2: CALCULATED RECOVERY
            # Check if bucket is stuck > 30 mins and needs a muscle move
            # Now includes IronShield for Trend Veto
            # Ensure any recovery NEW order respects the same per-symbol cap as zone recovery.
            try:
                market_data['max_positions_per_symbol'] = int(self.risk_manager.config.max_hedges)
            except Exception:
                # Fail-safe: if cap is unavailable, do not inject anything and let PositionManager fallback.
                pass
            calculated_recovery_executed = await self.position_manager.execute_calculated_recovery(
                self.broker, bucket_id, market_data, self.risk_manager.shield
            )
            if calculated_recovery_executed:
                logger.info(f"[MUSCLE] Calculated Recovery Executed for {bucket_id}")
                return True

            logger.debug(f"[ZONE_CHECK] Bucket {bucket_id} not closed, checking zone recovery")
            
            # Convert Position objects to dictionaries for zone recovery
            positions_dict = [
                {
                    'ticket': p.get('ticket') if isinstance(p, dict) else p.ticket,
                    'symbol': p.get('symbol') if isinstance(p, dict) else p.symbol,
                    'type': p.get('type') if isinstance(p, dict) else p.type,
                    'volume': p.get('volume') if isinstance(p, dict) else p.volume,
                    'price_open': p.get('price_open') if isinstance(p, dict) else p.price_open,
                    'price_current': p.get('price_current') if isinstance(p, dict) else p.price_current,
                    'profit': p.get('profit') if isinstance(p, dict) else p.profit,
                    'sl': p.get('sl') if isinstance(p, dict) else p.sl,
                    'tp': p.get('tp') if isinstance(p, dict) else p.tp,
                    'time': p.get('time') if isinstance(p, dict) else p.time,
                    'comment': p.get('comment', '') if isinstance(p, dict) else getattr(p, 'comment', '')
                }
                for p in positions
            ]
            
            # Calculate point value for symbol (XAUUSD uses different pip size)
            if "XAU" in symbol or "GOLD" in symbol:
                point_value = 0.01  # XAUUSD: 1 pip = 0.01
            elif "JPY" in symbol:
                point_value = 0.01  # JPY pairs: 1 pip = 0.01
            else:
                point_value = 0.0001  # Standard forex: 1 pip = 0.0001
            
            # Calculate current ATR for dynamic zone sizing
            atr_value = None
            if hasattr(self.market_data, 'calculate_atr_checked'):
                atr_v, ok, areason = self.market_data.calculate_atr_checked(symbol, 14)
                if ok and atr_v is not None:
                    atr_value = atr_v
                else:
                    logger.warning(f"[ZONE_CHECK] Skipping zone recovery: ATR unavailable ({areason})")
                    return bucket_closed
            else:
                atr_value = self.market_data.calculate_atr(symbol, 14) if hasattr(self.market_data, 'calculate_atr') else 0.0010

            if self._strict_entry and (rsi_value is None):
                logger.warning("[STRICT] Skipping zone recovery: RSI unavailable")
                return bucket_closed
            
            # Get volatility ratio
            volatility_ratio = self.market_data.get_volatility_ratio() if hasattr(self.market_data, 'get_volatility_ratio') else 1.0

            # Safety check for logging
            safe_atr = atr_value if atr_value is not None else 0.0
            safe_vol = volatility_ratio if volatility_ratio is not None else 1.0

            # [PHASE 5] Supreme Court: Dynamic Hedge Limits
            # Before we recover, check if a new hedge is Constitutional.
            # "HEDGE" action checks against current_hedge_cap (4-6) depending on volatility.
            hedge_approved, hedge_reason = self.authority.check_constitution(
                self.broker, symbol, 0.01, "HEDGE" # Volume estimate for check
            )
            
            if not hedge_approved:
                 # [PLAN B] If Hedge is blocked by Constitution (e.g. Volatility Cap hit),
                 # we enforce "Reversion Escape" (Plan B) to exit at break-even instead of stacking risk.
                 logger.debug(f"⚖️ [ZONE_CHECK] Hedge BLOCKED by Supreme Court: {hedge_reason}")
                 logger.debug(f"🛡️ [PLAN B] Activating Reversion Escape Protocol for {symbol}")
                 setattr(self, f"_plan_b_veto_{symbol}", True)
                 return bucket_closed

            logger.debug(f"[ZONE_CHECK] Calling execute_zone_recovery for {symbol} with {len(positions_dict)} positions | ATR: {safe_atr:.5f} | VolRatio: {safe_vol:.2f}")
            zone_recovery_executed = self.risk_manager.execute_zone_recovery(
                self.broker, symbol, positions_dict, tick, point_value,
                shield, ppo_guardian, self.position_manager, bool(self._strict_entry), oracle=oracle, atr_val=atr_value,
                volatility_ratio=volatility_ratio, rsi_value=rsi_value, trap_hunter=trap_hunter, pressure_metrics=pressure_metrics,
                max_hedges_override=self.authority.current_hedge_cap # [PHASE 5] Dynamic Cap
            )
            if zone_recovery_executed:
                logger.info(f"[ZONE] RECOVERY EXECUTED for {symbol}")
                # Force immediate update of position cache to reflect new hedge
                # This prevents "Ghost Trades" where the bot doesn't know about the new hedge
                
                # [CRITICAL UPDATE v7.0.9] Pipeline Fusion: Lock Signal Wire
                # Prevent "Hunting Mode" from firing fresh entries while Hedge is settling
                if self.entry_cooldowns is not None:
                     self.entry_cooldowns[symbol] = time.time()
                     logger.info(f"[PIPELINE COMMIT] Hedge executed -> Locked Initial Entry Cooldown for {symbol}")

                await asyncio.sleep(0.2) # Give broker a moment
                all_positions = self.broker.get_positions()
                if all_positions:
                    self.position_manager.update_positions(all_positions)
                    logger.info(f"[SYNC] Positions updated after hedge. Total: {len(all_positions)}")
            else:
                logger.debug(f"[ZONE_CHECK] No recovery needed for {symbol}")
            return zone_recovery_executed

        return bucket_closed

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        stats = self.session_stats.copy()
        stats["runtime_seconds"] = time.time() - stats["start_time"]
        return stats

    def reset_session_stats(self) -> None:
        """Reset session statistics."""
        self.session_stats = {
            "trades_opened": 0,
            "trades_closed": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "start_time": time.time()
        }
        logger.info("Session statistics reset")

    async def _check_global_safety(self):
        """
        [DOOMSDAY PROTOCOL] Global Equity Stop Loss.
        If Drawdown > 75%, CLOSE EVERYTHING immediately.
        
        [VALKYRIE PROTOCOL] The Freeze.
        If Drawdown > 15%, FREEZE ACCOUNT.
        """
        # 1. Check Valkyrie Status (15% Drawdown Freeze)
        if self._governor.valkyrie_active:
            if not getattr(self, '_valkyrie_executed', False):
                logger.critical("❄️ [VALKYRIE] GOVERNOR REQUESTED FREEZE! EXECUTING...")
                
                # Freeze all active buckets
                market_data = {
                     'ask': self.market_data.get_tick_data(self.config.symbol)['ask'],
                     'bid': self.market_data.get_tick_data(self.config.symbol)['bid']
                }
                
                # Iterate all buckets
                for bucket_id in list(self.position_manager.bucket_stats.keys()):
                     await self.position_manager.execute_perfect_hedge(self.broker, bucket_id, market_data)
                
                self._valkyrie_executed = True
                print(">>> ❄️ [VALKYRIE] ACCOUNT FROZEN. TRADING STOPPED.", flush=True)

        enable_doomsday = str(os.getenv("AETHER_ENABLE_DOOMSDAY", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not enable_doomsday:
            return

        if getattr(self, '_safety_lock', False):
            return # Already locked

        account_info = self.broker.get_account_info()
        if not account_info:
            return

        balance = account_info.get('balance', 0.0)
        equity = account_info.get('equity', 0.0)
        
        if balance <= 0: return

        drawdown_pct = (balance - equity) / balance
        
        # Safety limit (defaults to 75%, override via env)
        try:
            limit = float(os.getenv("AETHER_DOOMSDAY_DRAWDOWN_PCT", "0.75"))
        except Exception:
            limit = 0.75

        if drawdown_pct > limit:
            logger.critical(f"[DOOMSDAY] GLOBAL EQUITY STOP TRIGGERED! Drawdown: {drawdown_pct*100:.1f}%")
            print(f">>> [CRITICAL] DOOMSDAY PROTOCOL ACTIVATED. CLOSING ALL TRADES.", flush=True)
            
            self._safety_lock = True
            
            # Close all positions
            positions = self.broker.get_positions()
            if positions:
                if hasattr(self.broker, 'close_positions'):
                    try:
                        close_results = await self.broker.close_positions(
                            positions,
                            trace={
                                'strict_entry': bool(self._strict_entry),
                                'strict_ok': True,
                                'atr_ok': None,
                                'rsi_ok': None,
                                'obi_ok': None,
                                'reason': 'DOOMSDAY_GLOBAL_EQUITY_STOP',
                            },
                        )
                    except TypeError:
                        close_results = await self.broker.close_positions(positions)
                    try:
                        for ticket, res in (close_results or {}).items():
                            r = res or {}
                            ok = r.get('retcode') == mt5.TRADE_RETCODE_DONE
                            logger.critical(
                                f"[DOOMSDAY] Close ticket={ticket} ok={ok} retcode={r.get('retcode')} comment={r.get('comment', '')}"
                            )
                    except Exception:
                        logger.critical("[DOOMSDAY] Close batch completed (result parse failed)")
                else:
                    for pos in positions:
                        ticket = pos.get('ticket') if isinstance(pos, dict) else pos.ticket
                        try:
                            await self.broker.close_position(
                                ticket,
                                trace={
                                    'strict_entry': bool(self._strict_entry),
                                    'strict_ok': True,
                                    'atr_ok': None,
                                    'rsi_ok': None,
                                    'obi_ok': None,
                                    'reason': 'DOOMSDAY_GLOBAL_EQUITY_STOP',
                                },
                            )
                        except TypeError:
                            await self.broker.close_position(ticket)
                        logger.critical(f"[DOOMSDAY] Closed ticket {ticket}")
            
            # Raise flag to stop bot
            raise Exception("Global Equity Stop Loss Triggered")

    async def run_trading_cycle(self, strategist, shield, ppo_guardian,
                               nexus=None, oracle=None) -> None:
        """
        Run a complete trading cycle.

        Args:
            strategist: Strategist instance
            shield: IronShield instance
            ppo_guardian: PPO Guardian instance
            nexus: Optional NexusBrain instance
            oracle: Optional Oracle instance (Layer 4)
        """
        symbol = self.config.symbol

        try:
            # [SAFETY FIRST] Check Global Equity Stop
            await self._check_global_safety()

            # Maintain end-of-session risk metrics (profit vs drawdown)
            self._update_equity_metrics_throttled()

            # Get market data
            # print(f">>> [DEBUG] Fetching tick for {symbol}...", flush=True)
            tick = self.market_data.get_tick_data(symbol)
            if not tick:
                print(f">>> [DEBUG] No tick data for {symbol} (Check MT5 Connection/Market Watch)", flush=True)
                return

            # Optional telemetry: prove tick/candle freshness live (throttled)
            self._maybe_log_freshness_trace(tick)
            self._maybe_log_data_provenance_trace(symbol, tick)

            # [HIGHEST INTELLIGENCE] Update Tick Pressure Analyzer
            self.tick_analyzer.add_tick(tick)
            pressure_metrics = self.tick_analyzer.get_pressure_metrics()

            # Record market data to database
            await self._record_market_data(symbol, tick)

            # Validate market conditions
            market_ok, reason = self.validate_market_conditions(symbol, tick)
            
            if not market_ok:
                # [UI FEEDBACK] Log pause reason only when it changes to avoid spam
                # Special handling for "Spread too wide" to avoid spamming due to small pip changes
                is_spread_issue = "Spread too wide" in reason
                
                should_log = False
                if is_spread_issue:
                    # Only log if we weren't already paused for spread
                    if self.last_pause_reason is None or "Spread too wide" not in self.last_pause_reason:
                        should_log = True
                elif reason != self.last_pause_reason:
                    should_log = True

                if should_log:
                    logger.info(f"[PAUSED] TRADING PAUSED: {reason}")
                    # User requested to disable this from terminal log
                    if not is_spread_issue:
                        print(f">>> [PAUSED] {reason}", flush=True)
                    self.last_pause_reason = reason
                return
            
            # [UI FEEDBACK] Log resumption if we were previously paused
            if self.last_pause_reason is not None:
                logger.info(f"[RESUMED] TRADING RESUMED: Market conditions normalized.")

                self.last_pause_reason = None

            # Get account info
            account_info = self.broker.get_account_info()
            if not account_info:
                print(">>> [WARN] Could not fetch account info", flush=True)
                return

            # Process management for existing positions
            # Check for positions first to determine mode (Management vs Hunting)
            # [ROBUSTNESS] Fetch all positions and filter manually to handle case-sensitivity
            # This prevents "Ghost Entries" where bot misses existing trades due to 'xauusd' vs 'XAUUSD'
            all_pos = self.broker.get_all_positions()
            if all_pos is None:
                positions = None
            else:
                # FIX: Handle both Object (dot notation) and Dict (subscript) just in case
                positions = []
                # DEBUG: Log all visible symbols to help user debug matching issues
                if all_pos:
                    visible_symbols = set()
                    for p in all_pos:
                        s = p.symbol if hasattr(p, 'symbol') else (p.get('symbol') if isinstance(p, dict) else 'Unknown')
                        visible_symbols.add(s)


                positions = []
                for p in all_pos:
                     # Adapt to Object or Dict
                     if hasattr(p, 'symbol'):
                         sym = p.symbol
                     else:
                         sym = p.get('symbol') if isinstance(p, dict) else None
                         
                     # [ROBUSTNESS] Fuzzy Match (Handle suffixes like .m, .pro, +, etc.)
                     # If configured "XAUUSD" matches "XAUUSD.m" or "XAUUSD+"
                     match = False
                     if sym:
                         sym_clean = sym.lower()
                         cfg_clean = symbol.lower()
                         if sym_clean == cfg_clean:
                             match = True
                         elif sym_clean.startswith(cfg_clean) and len(sym_clean) <= len(cfg_clean) + 4:
                             match = True
                     
                     if match:
                         # FIX: Convert namedtuple/dataclass to dict for downstream compatibility
                         positions.append(normalize_position(p))
            
            # [CRITICAL FIX] Handle Broker API Failure
            if positions is None:
                logger.warning(f"[SAFETY] Failed to fetch positions for {symbol} - Skipping cycle to prevent ghost trades.")
                return

            if positions:
                # MANAGEMENT MODE
                await self._process_existing_positions(symbol, tick, shield, ppo_guardian, oracle, pressure_metrics)
                return # STRICTLY RETURN - No new entries while positions exist

            # HUNTING MODE
            # Proceed to AI analysis for new entries

            # [STRICT ENTRIES] Require real, sufficient, fresh inputs before opening new positions.
            macro_context = self.market_data.get_macro_context()
            atr_value, trend_strength = self._calculate_indicators(symbol)
            atr_value, trend_strength = self._calculate_indicators(symbol)
            rsi_value = self.market_data.calculate_rsi(symbol)
            
            # [PHASE 5] Update Constitution (Dynamic Layers)
            # Fetch equity for scaling
            current_equity = self.broker.get_equity()
            self.authority.update_constitution(atr_value, current_equity)

            # Freshness gate applies to any NEW entry attempt (even if strict mode is off)
            if self._freshness_gate:
                ok, freason = self._freshness_ok_for_new_orders(tick)
                if not ok:
                    self._log_entry_gate(f"Freshness gate: {freason}")
                    return

            if self._strict_entry:
                # Candle sufficiency check (prevents ATR/RSI/trend falling back to neutral defaults)
                # [FRESHNESS] Force refresh candles to ensure strict entry checks use latest data
                history = self.market_data.candles.get_history(symbol, force_refresh=True)
                if not history or len(history) < self._strict_entry_min_candles:
                    self._log_entry_gate(
                        f"Insufficient candles: have={len(history) if history else 0} need={self._strict_entry_min_candles}"
                    )
                    return

                # Indicators must be computed from real candle history (no neutral fallbacks)
                try:
                    if hasattr(self.market_data, 'calculate_atr_checked'):
                        atr_v, ok, areason = self.market_data.calculate_atr_checked(symbol, 14)
                        if not ok or atr_v is None:
                            self._log_entry_gate(f"ATR unavailable: {areason}")
                            return
                        atr_value = atr_v

                    if hasattr(self.market_data, 'calculate_trend_strength_checked'):
                        ts_v, ok, treason = self.market_data.calculate_trend_strength_checked(symbol, 20)
                        if not ok or ts_v is None:
                            self._log_entry_gate(f"Trend unavailable: {treason}")
                            return
                        trend_strength = ts_v

                    if hasattr(self.market_data, 'calculate_rsi_checked'):
                        rsi_v, ok, rreason = self.market_data.calculate_rsi_checked(symbol, 14)
                        if not ok or rsi_v is None:
                            self._log_entry_gate(f"RSI unavailable: {rreason}")
                            return
                        rsi_value = rsi_v
                except Exception as e:
                    self._log_entry_gate(f"Indicator check error: {e}")
                    return

                # Macro proxies: if correlations are enabled, require proxy ticks to be available
                try:
                    if getattr(self.market_data, 'macro_eye', None) is not None and hasattr(self.market_data, 'get_macro_context_checked'):
                        macro_vec, ok, mreason = self.market_data.get_macro_context_checked()
                        if not ok:
                            self._log_entry_gate(f"Macro data unavailable: {mreason}")
                            return
                        if ok and macro_vec is not None:
                            macro_context = macro_vec
                except Exception as e:
                    self._log_entry_gate(f"Macro check error: {e}")
                    return

            # --- PHASE 4: PREDATOR VISION (Trap Trading) ---
            # "Stop Hunting the Stop Hunters"
            # If we detect a Bull Trap, we SELL immediately (fading the breakout).
            trap_signal = self.trap_hunter.scan(symbol, history, tick)
            
            if trap_signal.is_trap and trap_signal.suggested_action in ("BUY", "SELL"):
                 # Check confidence
                 if trap_signal.counter_confidence >= 0.8:
                     logger.critical(f"🦅 [PREDATOR] SIGNAL DETECTED: {trap_signal.trap_type} -> {trap_signal.suggested_action}")
                     logger.info(f"    Reason: {trap_signal.details}")
                     
                     # Execute Counter-Strike
                     price = tick['ask'] if trap_signal.suggested_action == "BUY" else tick['bid']
                     
                     # Use standard sizing (or aggressive?) - Stick to standard for now
                     # Volume is calculated by Risk Manager inside check_entry_conditions usually,
                     # but here we are bypassing. We need to calculate volume.
                     # Quick fix: Use min volume (0.01) * 5 or configured base.
                     base_vol = 0.1 # Default Predator size
                     
                     # [SAFETY] Validate with Risk Governor (Shadow Balance & Valkyrie Check)
                     # Fetch fresh metrics for accurate risk assessment
                     acct_info = self.broker.get_account_info()
                     risk_metrics = {
                         "balance": acct_info.get('balance', 0.0),
                         "equity": acct_info.get('equity', 0.0),
                         "margin": acct_info.get('margin', 0.0),
                         "total_positions": self.position_manager.get_total_positions(),
                         # Approximate exposure/drawdown if not easily available, or fetch properly
                         "total_exposure_pct": (acct_info.get('margin', 0.0) / acct_info.get('balance', 1.0)) if acct_info.get('balance', 1.0) > 0 else 0.0,
                         "account_drawdown_pct": self.position_manager._calculate_drawdown(acct_info.get('balance', 0.0), acct_info.get('equity', 0.0))
                     }

                     # [SAFETY] Validate with Risk Governor (Shadow Balance & Valkyrie Check)
                     # Fetch fresh metrics for accurate risk assessment
                     acct_info = self.broker.get_account_info()
                     risk_metrics = {
                         "balance": acct_info.get('balance', 0.0),
                         "equity": acct_info.get('equity', 0.0),
                         "margin": acct_info.get('margin', 0.0),
                         "total_positions": self.position_manager.get_total_positions(),
                         # Approximate exposure/drawdown if not easily available, or fetch properly
                         "total_exposure_pct": (acct_info.get('margin', 0.0) / acct_info.get('balance', 1.0)) if acct_info.get('balance', 1.0) > 0 else 0.0,
                         "account_drawdown_pct": self.position_manager._calculate_drawdown(acct_info.get('balance', 0.0), acct_info.get('equity', 0.0))
                     }

                     veto, veto_reason = self._governor.veto(risk_metrics)
                     
                     # [PHASE 5] Validate with Supreme Court (Global Cap & Dynamic Layers)
                     approved, reason = self.authority.check_constitution(
                         self.broker, symbol, base_vol, "OPEN"
                     )
                     
                     if not veto and approved:
                          self.broker.execute_order(
                                symbol=symbol,
                                action="OPEN",
                                order_type=trap_signal.suggested_action,
                                price=price,
                                volume=base_vol,
                                sl=0.0, tp=0.0, # Managed by bucket logic
                                comment="PREDATOR_TRAP",
                                trace_reason=f"PREDATOR_{trap_signal.trap_type}"
                          )
                          return # Skip standard AI logic
                     
                     if veto:
                          logger.warning(f"[PREDATOR] Signal valid but Risk Governor Veto: {veto_reason}")
                     if not approved:
                          logger.warning(f"[PREDATOR] Signal valid but Unconstitutional: {reason}")

            # --- LAYER 4: ORACLE ENGINE ---
            oracle_prediction = "NEUTRAL"
            oracle_confidence = 0.0
            history = self.market_data.candles.get_history(symbol) # Get history once
            
            if oracle:
                # Get last 60 candles
                if len(history) >= 60:
                    # [UPGRADE] Use V2 Logic (AI + Macro + Fusion)
                    oracle_result = await oracle.get_sniper_signal_v2(symbol, history[-60:])
                    
                    # Map result back to prediction/confidence for compatibility
                    sig = oracle_result['signal']
                    if sig == 1: oracle_prediction = "UP"
                    elif sig == -1: oracle_prediction = "DOWN"
                    else: oracle_prediction = "NEUTRAL"
                    
                    oracle_confidence = oracle_result['confidence']
                    
                    if oracle_confidence > 0.5: # Lower threshold as Fusion is stricter
                        logger.info(f"[ORACLE] Prediction: {oracle_prediction} ({oracle_confidence:.2f}) | {oracle_result['reason']}")

                    # [HIGHEST INTELLIGENCE] LAYER 1: REGIME DETECTION
                    # Use Oracle's advanced math to double-check regime
                    oracle_regime, oracle_signal = oracle.get_regime_and_signal(history[-60:])
                    logger.info(f"[ORACLE] Regime: {oracle_regime} | Signal: {oracle_signal}")
                    
                    # [USER REQUEST] DISABLED CIRCUIT BREAKER FOR INITIAL ENTRIES
                    # The user requested that enhancements apply ONLY to hedging/recovery.
                    # We log the Oracle's opinion but do NOT block the trade.

            # --- HIERARCHICAL AI DECISION LOGIC (v5.0) ---
            # 1. Supervisor: Detect Regime
            # Prepare data for Supervisor
            # Note: atr_value/trend_strength/rsi_value/macro_context are precomputed above,
            # and in strict mode are guaranteed to be based on real data.
            
            # 1. Supervisor: Detect Regime
            # Prepare data for Supervisor
            # Note: atr_value/trend_strength/rsi_value/macro_context are precomputed above
            
            supervisor_data = {
                'atr': atr_value,
                'trend_strength': trend_strength,
                'volatility_ratio': self.market_data.get_volatility_ratio(),
                'macro_context': macro_context,
                'pressure_metrics': pressure_metrics
            }
            
            # [UPGRADE] Pass candle history to Supervisor for Geometrician (Entropy/Hurst) Analysis
            regime = self.supervisor.detect_regime(supervisor_data, candles=history)
            logger.info(f"[SUPERVISOR] Market Regime: {regime.name} ({regime.confidence:.2f}) | {regime.description}")
            
            # 2. Supervisor: Select Worker
            worker_name = self.supervisor.get_active_worker(regime)
            
            # [AI STATUS MONITOR] Store Analysis State for Real-Time Dashboard
            self.latest_analysis = {
                'regime': regime.name,
                'oracle_pred': oracle_prediction,
                'oracle_conf': oracle_confidence,
                'pressure': pressure_metrics,
                'worker': worker_name,
                'rsi': rsi_value,
                'atr': atr_value,
                'timestamp': time.time()
            }
            
            # 3. Worker: Generate Signal
            # Prepare context for worker
            market_context = {
                'symbol': symbol,
                'current_price': tick['bid'],
                'tick': tick,
                'history': history,
                'macro_context': macro_context,
                'rsi': rsi_value,
                'trend_strength': trend_strength,
                'atr': atr_value,
                'pressure_metrics': pressure_metrics # [HIGHEST INTELLIGENCE]
            }
            
            action, confidence, reason = "HOLD", 0.0, "No Worker"
            
            if worker_name == "RANGE_WORKER":
                action, confidence, reason = self.range_worker.get_signal(market_context)
            elif worker_name == "TREND_WORKER":
                action, confidence, reason = self.trend_worker.get_signal(market_context)
            else:
                # Fallback for CHAOS or DEFENSIVE regimes
                action, confidence, reason = self.range_worker.get_signal(market_context)
                reason = f"[DEFENSIVE] {reason}"

            # --- ORACLE ALIGNMENT FILTER (Anti-Overlap) ---
            # Prevents entries when the worker fights the Oracle with high confidence.
            if action != "HOLD" and oracle_prediction in {"UP", "DOWN"} and oracle_confidence > 0.0:
                aligns = (oracle_prediction == "UP" and action == "BUY") or (oracle_prediction == "DOWN" and action == "SELL")
                if aligns:
                    confidence = min(1.0, confidence + (0.05 * oracle_confidence))
                    reason = f"{reason} | Oracle Align ({oracle_prediction} {oracle_confidence:.2f})"
                else:
                    confidence = max(0.0, confidence - (0.40 * oracle_confidence))
                    reason = f"{reason} | Oracle Veto ({oracle_prediction} {oracle_confidence:.2f})"

            # Final clamp
            confidence = max(0.0, min(1.0, confidence))
                
            # [TRAP HUNTER] Check for institutional traps (Fakeouts/Icebergs)
            self.trap_hunter.scan(symbol, history, tick)
            is_trap = self.trap_hunter.is_trap(action)
            if action != "HOLD" and is_trap:
                log_msg = f"[TRAP DETECTED] {action} signal blocked by Trap Hunter."
                logger.warning(log_msg)
                action = "HOLD"
                reason = f"{reason} | [TRAP-VETO] Institutional Trap Detected"
                confidence = 0.0

            # Create Signal Object
            signal = None
            # ====================================================================
            # [MASTER ALGORITHM] THE UNIFIED FIELD THEORY VALIDATION LOOP
            # Integrated: January 8, 2026
            # ====================================================================
            if action != "HOLD" and confidence > 0.5:
                physics_ok, physics_reason = self._validate_physics_conditions(
                    action, regime, pressure_metrics, tick
                )
                
                if not physics_ok:
                    logger.warning(f"[PHYSICS BLOCKED] {action} blocked: {physics_reason}")
                    action = "HOLD"
                    reason = f"{reason} | [PHYSICS] {physics_reason}"
                    confidence = 0.0

            if action != "HOLD" and confidence > 0.5:
                # [DIRECTION VALIDATOR] - 7-Factor Intelligence Check (with error handling)
                try:
                    from src.ai_core.direction_validator import get_direction_validator
                    
                    validator = get_direction_validator()
                    
                    # Prepare market data for validation
                    # Handle macro_context being either dict or list
                    macro_dict = macro_context if isinstance(macro_context, dict) else {}
                    
                    # Fetch Multi-Timeframe Trends (Factor 5)
                    mtf_trends = self.market_data.calculate_multi_timeframe_trends(symbol)
                    
                    # [PREDICTIVE INTELLIGENCE] Calculate next 5 candles trajectory
                    oracle_trajectory = []
                    if oracle:
                        # Use same history cache
                        oracle_trajectory = oracle.predict_trajectory(history[-60:], horizon=5)

                    validation_data = {
                        'trend': regime.name if hasattr(regime, 'name') else str(regime),
                        'regime': regime,
                        'trajectory': oracle_trajectory, # [NEW] Passed to Analyst
                        'rsi': rsi_value,
                        'macd': self.market_data.calculate_macd(symbol),
                        'volume': history[-1].get('tick_volume', 0) if history else 0,
                        'avg_volume': sum(h.get('tick_volume', 0) for h in history[-20:]) / 20 if len(history) >= 20 else 0,
                        'current_price': tick['bid'],
                        'support': macro_dict.get('support', tick['bid'] - 10),
                        'resistance': macro_dict.get('resistance', tick['bid'] + 10),
                        'tick': tick,
                        'avg_spread': tick.get('spread', tick['ask'] - tick['bid']),
                        'oracle_prediction': oracle_prediction,
                        'oracle_confidence': oracle_confidence,
                        'nexus_signal_confidence': confidence,
                        'm1_trend': mtf_trends.get('m1_trend', 'NEUTRAL'),
                        'm5_trend': mtf_trends.get('m5_trend', 'NEUTRAL'),
                        'm15_trend': mtf_trends.get('m15_trend', 'NEUTRAL'),
                        'pressure': pressure_metrics # [COUNCIL] Pass Pressure to Validator
                    }
                    
                    # Validate direction (protected by validator's internal error handling)
                    validation = validator.validate_direction(action, validation_data, confidence)
                    
                    # Apply validation results
                    original_confidence = confidence
                    confidence = confidence * validation.confidence_multiplier
                    confidence = max(0.0, min(1.0, confidence))  # Clamp again
                    
                    # Check if we should invert the signal
                    if validation.should_invert and validation.score < 0.20:
                        # Very poor validation - flip direction
                        action = "SELL" if action == "BUY" else "BUY"
                        logger.warning(f"[SIGNAL] ⚠️ INVERTED {action}: {validation.reasoning}")
                        reason = f"{reason} | INVERTED ({validation.score:.0%})"
                    elif validation.score >= 0.80:
                        # Strong validation - only log in debug
                        logger.debug(f"[SIGNAL] ✅ Strong validation ({validation.score:.0%})")
                        reason = f"{reason} | Validated ({validation.score:.0%})"
                    elif validation.score < 0.25:
                        # CRITICAL: Block very weak signals to prevent bad trades
                        # [ADJUSTMENT] Lowered to 25% to allow defensive entries in CHAOS
                        logger.warning(
                            f"[SIGNAL] 🚫 BLOCKED: Validation too weak ({validation.score:.0%}). "
                            f"Failed factors: {', '.join(validation.failed_factors[:3])}"
                        )
                        logger.info(f"[SIGNAL] Waiting for stronger signal (need ≥40% validation)")
                        return  # Block the trade
                    # Moderate validation (40-80%) - no logging, just proceed
                    

                    # Store validation score in metadata
                    validation_score = validation.score
                    validation_factors = f"{validation.passed_factors}/{validation.total_factors}"
                    
                    # [WICK INTELLIGENCE] Check for wick rejection zones
                    try:
                        from src.ai_core.wick_intelligence import get_wick_intelligence
                        
                        wick_intel = get_wick_intelligence()
                        should_block, wick_reason = wick_intel.should_block_trade(
                            direction=action,
                            current_price=tick['bid'],
                            recent_candles=history[-10:] if history else []
                        )
                        
                        if should_block:
                            # Use Dashboard for deduped blocking logs
                            self.dashboard.log_block(f"WICK INTELLIGENCE: {wick_reason}")
                            
                            # Suggest better entry
                            suggested_price = wick_intel.get_safe_entry_suggestion(
                                direction=action,
                                current_price=tick['bid'],
                                recent_candles=history[-10:] if history else []
                            )
                            
                            if suggested_price:
                                logger.debug(
                                    f"[WICK INTELLIGENCE] 💡 Suggested entry: "
                                    f"{suggested_price:.2f} (current: {tick['bid']:.2f})"
                                )
                            
                            # Skip this trade
                            return
                        else:
                            logger.debug(f"[WICK INTELLIGENCE] ✅ {wick_reason}")
                    
                    except Exception as wick_e:
                        logger.debug(f"[WICK INTELLIGENCE] Check failed: {wick_e}")
                        # Continue without wick check if it fails
                    
                except Exception as e:
                    # Direction Validator failed - continue with original signal
                    logger.error(f"[DIRECTION_VALIDATOR] Validation failed: {e}")
                    logger.error("[DIRECTION_VALIDATOR] Continuing with original signal (no validation)")
                    validation_score = 0.5  # Neutral score on error
                    validation_factors = "0/7"
                
                signal = TradeSignal(
                    action=TradeAction.BUY if action == "BUY" else TradeAction.SELL,
                    symbol=symbol,
                    confidence=confidence,
                    reason=f"[{worker_name}] {reason}",
                    metadata={
                        'regime': regime.name,
                        'worker': worker_name,
                        'macro_data': macro_context,
                        'atr': atr_value,
                        'trend_strength': trend_strength,
                        'rsi': rsi_value,
                        'oracle_prediction': oracle_prediction, # Keep Oracle for penalties
                        'nexus_signal_confidence': confidence, # Map for compatibility
                        'validation_score': validation_score,
                        'validation_factors': validation_factors
                    },
                    timestamp=time.time()
                )

            
            # Use market_regime variable for compatibility with downstream logic
            market_regime = regime
            if not signal:
                # [UI FEEDBACK] Log "Thinking" status periodically to reassure user
                if self.decision_tracker:
                    # Fix: Provide default confidence for HOLD decisions
                    should_log, _ = self.decision_tracker.should_log_decision(
                        symbol, {'action': 'HOLD', 'confidence': 0.0, 'reasoning': reason}
                    )
                    if should_log:
                        # Use Dashboard for deduped AI thinking logs
                        self.dashboard.ai_decision(
                            prediction="HOLD",
                            confidence=0.0,
                            reason=f"{regime.name} | {reason}"
                        )
                return
            
            # [CRITICAL FIX] Double Entry Prevention (Thread-Safe)
            # Use atomic check-and-set to prevent race conditions
            import threading
            if not hasattr(self, '_entry_lock'):
                self._entry_lock = threading.Lock()
            
            # Atomic cooldown check and set
            with self._entry_lock:
                last_entry_ts = self.entry_cooldowns.get(symbol, 0)
                time_since_last = time.time() - last_entry_ts
                
                # [FIX] Use Configurable Cooldown (was hardcoded 60s)
                # For scalping, 5.0s is usually sufficient to prevent machine-gunning
                cooldown_limit = self.config.global_trade_cooldown
                
                if time_since_last < cooldown_limit:
                    # Cooldown active - block this entry
                    # Only log if it's not super spammy (e.g. every 5s)
                    if time_since_last > 1.0: 
                         logger.debug(f"[COOLDOWN] Entry blocked for {symbol}. Last entry was {time_since_last:.1f}s ago (limit: {cooldown_limit}s).")
                    return
                
                # Reserve this entry slot immediately (atomic)
                self.entry_cooldowns[symbol] = time.time()
                logger.debug(f"[COOLDOWN] Entry slot reserved for {symbol}. Cooldown set.")

            # Show AI decision on dashboard (event-driven, no spam)
            dashboard = get_dashboard()
            dashboard.ai_decision(
                prediction=signal.action.value,
                confidence=signal.confidence,
                reason=signal.reason
            )

            # --- CHAMELEON FILTER (SOFT PENALTY) ---
            # Instead of blocking, reduce lot size for Counter-Trend trades
            regime_penalty = 1.0
            if market_regime.name == "TRENDING_UP" and signal.action == TradeAction.SELL:
                regime_penalty = 0.5 # Cut lots in half
                logger.info(f"[CHAMELEON] CAUTION: Counter-Trend SELL. Reducing lots by 50%.")
            elif market_regime.name == "TRENDING_DOWN" and signal.action == TradeAction.BUY:
                regime_penalty = 0.5 # Cut lots in half
                logger.info(f"[CHAMELEON] CAUTION: Counter-Trend BUY. Reducing lots by 50%.")
                
            # --- ORACLE FILTER (SOFT PENALTY) ---
            # If Oracle strongly disagrees, reduce lot size further
            oracle_penalty = 1.0
            if oracle_prediction != "NEUTRAL" and oracle_confidence > 0.7:
                if (signal.action == TradeAction.BUY and oracle_prediction == "DOWN") or \
                   (signal.action == TradeAction.SELL and oracle_prediction == "UP"):
                    oracle_penalty = 0.5
                    logger.info(f"[ORACLE] CAUTION: Signal contradicts Oracle. Reducing lots by 50%.")

            # --- HOLOGRAPHIC FILTER (TICK PRESSURE) ---
            # If Order Flow Pressure strongly disagrees, reduce lot size
            pressure_penalty = 1.0
            if pressure_metrics['intensity'] == 'HIGH':
                if (signal.action == TradeAction.BUY and pressure_metrics['dominance'] == 'SELL') or \
                   (signal.action == TradeAction.SELL and pressure_metrics['dominance'] == 'BUY'):
                    pressure_penalty = 0.5
                    logger.info(f"[HOLOGRAPHIC] CAUTION: High Intensity Counter-Pressure. Reducing lots by 50%.")

            # --- LAYER 9: GLOBAL BRAIN (SOFT PENALTY) ---
            # Check DXY/US10Y impact on Gold
            global_bias = 0.0
            brain_penalty = 1.0
            if self.global_brain and ("XAU" in symbol or "GOLD" in symbol):
                correlation_signal = None
                try:
                    # Live proxy feed (USDJPY/US500 etc) with confidence gating
                    if hasattr(self.global_brain, 'get_bias_signal'):
                        correlation_signal = self.global_brain.get_bias_signal()
                    else:
                        # Backward-compat: fall back to score-only
                        global_bias = float(self.global_brain.get_bias())
                except Exception:
                    correlation_signal = None

                if correlation_signal is not None and getattr(correlation_signal, 'confidence', 0.0) > 0.0:
                    global_bias = float(correlation_signal.score)
                    logger.info(
                        f"[GLOBAL_BRAIN] Correlation Bias: {global_bias:.2f} (Driver: {correlation_signal.driver}, Conf: {correlation_signal.confidence:.2f})"
                    )

                    # PENALTY LOGIC: If Global Brain disagrees, reduce size
                    if global_bias < -0.5 and signal.action == TradeAction.BUY:
                        brain_penalty = 0.5
                        logger.info(f"[GLOBAL_BRAIN] CAUTION: Bearish Correlation ({global_bias:.2f}). Reducing lots by 50%.")
                    elif global_bias > 0.5 and signal.action == TradeAction.SELL:
                        brain_penalty = 0.5
                        logger.info(f"[GLOBAL_BRAIN] CAUTION: Bullish Correlation ({global_bias:.2f}). Reducing lots by 50%.")

            # ATR/trend already computed for this cycle; reuse to avoid introducing fallbacks.

            # Calculate position size with PPO optimization
            lot_size, lot_reason = self.calculate_position_size(
                signal, account_info, strategist, shield,
                ppo_guardian=ppo_guardian,
                atr_value=atr_value,
                trend_strength=trend_strength
            )
            
            # [FORENSIC LOGGING] Track initial lot calculation
            initial_lot_size = lot_size
            logger.debug(f"[DECISION_PATH] ═══════════════════════════════════════════")
            logger.debug(f"[DECISION_PATH] TRADE DECISION ANALYSIS")
            logger.debug(f"[DECISION_PATH] ═══════════════════════════════════════════")
            logger.debug(f"[DECISION_PATH] 1. SIGNAL GENERATION:")
            logger.debug(f"[DECISION_PATH]    Worker: {signal.metadata.get('worker', 'Unknown')}")
            logger.debug(f"[DECISION_PATH]    Action: {signal.action.value}")
            logger.debug(f"[DECISION_PATH]    Confidence: {signal.confidence:.2%}")
            logger.debug(f"[DECISION_PATH]    Reason: {signal.reason}")
            logger.debug(f"[DECISION_PATH] 2. MARKET CONTEXT:")
            logger.debug(f"[DECISION_PATH]    Regime: {market_regime.name}")
            logger.debug(f"[DECISION_PATH]    ATR: {atr_value:.4f}")
            logger.debug(f"[DECISION_PATH]    Trend Strength: {trend_strength:.2f}")
            logger.debug(f"[DECISION_PATH]    RSI: {rsi_value:.1f}")
            logger.debug(f"[DECISION_PATH]    Validation Score: {validation_score:.0%}")
            logger.debug(f"[DECISION_PATH] 3. INITIAL LOT CALCULATION:")
            logger.debug(f"[DECISION_PATH]    Base Lot: {initial_lot_size:.4f}")
            logger.debug(f"[DECISION_PATH]    Reason: {lot_reason}")
            logger.debug(f"[DECISION_PATH] 4. PENALTIES TO APPLY:")
            logger.debug(f"[DECISION_PATH]    Regime Penalty: {regime_penalty:.2f}x")
            logger.debug(f"[DECISION_PATH]    Oracle Penalty: {oracle_penalty:.2f}x")
            logger.debug(f"[DECISION_PATH]    Brain Penalty: {brain_penalty:.2f}x")
            logger.debug(f"[DECISION_PATH]    Pressure Penalty: {pressure_penalty:.2f}x")
            

            # === ADAPTIVE VALIDATION MULTIPLIER ===
            # Instead of blocking trades, scale size/TP based on validation score
            validation_multiplier = 1.0
            tp_multiplier = 1.0
            sl_multiplier = 1.0
            adaptive_strategy = "STANDARD"
            
            if validation_score >= 0.70:
                # Strong validation - Full confidence
                validation_multiplier = 1.0
                tp_multiplier = 1.0
                sl_multiplier = 1.0
                adaptive_strategy = "TREND_FOLLOWING"
                logger.debug(f"[ADAPTIVE] Strong signal ({validation_score:.0%}) → Full size, standard TP/SL")
                
            elif validation_score >= 0.50:
                # Moderate validation - Cautious scalping
                validation_multiplier = 0.6  # 60% size
                tp_multiplier = 0.6  # Tighter TP (60% of normal)
                sl_multiplier = 0.7  # Tighter SL (70% of normal)
                adaptive_strategy = "SCALPING"
                logger.debug(f"[ADAPTIVE] Moderate signal ({validation_score:.0%}) → Reduced size (60%), tight TP/SL (Scalp mode)")
                
            elif validation_score >= 0.40:
                # Weak validation - Very cautious
                validation_multiplier = 0.4  # 40% size
                tp_multiplier = 0.4  # Very tight TP
                sl_multiplier = 0.5  # Very tight SL
                adaptive_strategy = "MICRO_SCALP"
                logger.debug(f"[ADAPTIVE] Weak signal ({validation_score:.0%}) → Minimal size (40%), micro-scalp TP/SL")
                
            elif validation_score >= 0.25:
                # Very weak - Range trading only
                validation_multiplier = 0.25  # 25% size
                tp_multiplier = 0.3  # Extremely tight TP
                sl_multiplier = 0.4  # Extremely tight SL
                adaptive_strategy = "RANGE_FADE"
                logger.debug(f"[ADAPTIVE] Very weak signal ({validation_score:.0%}) → Range fade (25% size, tight exits)")
            else:
                # Critical - Block trade
                logger.warning(f"[ADAPTIVE] Critical divergence ({validation_score:.0%}) → Trade BLOCKED")
                return
            
            # Store multipliers in signal metadata for TP/SL calculation
            signal.metadata['tp_multiplier'] = tp_multiplier
            signal.metadata['sl_multiplier'] = sl_multiplier
            signal.metadata['adaptive_strategy'] = adaptive_strategy
            
            # === APPLY INTELLIGENCE PENALTIES ===
            # Combine all penalties (Validation * Regime * Oracle * Brain * Pressure)
            # Example: 0.6 * 0.5 * 0.5 * 1.0 * 1.0 = 0.15 (15% size)
            total_penalty = validation_multiplier * regime_penalty * oracle_penalty * brain_penalty * pressure_penalty
            
            if total_penalty < 1.0:
                old_lot = lot_size
                lot_size = round(lot_size * total_penalty, 4)
                # [ADAPTIVE FIX] Allow smaller lots for weak signals (min 0.001 instead of 0.01)
                # This enables MICRO_SCALP and RANGE_FADE modes to actually reduce size
                lot_size = max(lot_size, 0.001)
                lot_reason += f" (AI Penalty: {total_penalty:.2f}x)"
                logger.debug(f"[AI_COUNCIL] Consensus Weak. Reducing Size: {old_lot} -> {lot_size} lots")

            # --- CHAMELEON VOLATILITY SCALING ---
            # If Volatile, cut lot size in half for safety
            if market_regime.name == "VOLATILE":
                lot_size = lot_size * 0.5
                lot_reason += " (Reduced 50% due to VOLATILE regime)"
                logger.info(f"[CHAMELEON] Volatility Detected! Reducing lot size to {lot_size:.2f}")

            if lot_size <= 0:
                print(f">>> [DEBUG] Lot Size Rejected: {lot_reason}", flush=True)
                logger.info(f"[POSITION] SIZING REJECTED: {lot_reason}")
                logger.debug(f"[DECISION_PATH] ═══════════════════════════════════════════")
                logger.debug(f"[DECISION_PATH] TRADE REJECTED: Lot size {lot_size:.4f} <= 0")
                logger.debug(f"[DECISION_PATH] ═══════════════════════════════════════════")
                return
            
            # [FORENSIC LOGGING] Track final lot size after all multipliers
            logger.debug(f"[DECISION_PATH] 5. ADAPTIVE MULTIPLIER:")
            logger.debug(f"[DECISION_PATH]    Validation Multiplier: {validation_multiplier:.2f}x")
            logger.debug(f"[DECISION_PATH]    TP Multiplier: {tp_multiplier:.2f}x")
            logger.debug(f"[DECISION_PATH]    SL Multiplier: {sl_multiplier:.2f}x")
            logger.debug(f"[DECISION_PATH]    Strategy: {adaptive_strategy}")
            logger.debug(f"[DECISION_PATH] 6. FINAL LOT CALCULATION:")
            logger.debug(f"[DECISION_PATH]    Initial: {initial_lot_size:.4f}")
            logger.debug(f"[DECISION_PATH]    After Penalties: {lot_size:.4f}")
            logger.debug(f"[DECISION_PATH]    Total Multiplier: {(lot_size / initial_lot_size if initial_lot_size > 0 else 0):.4f}x")


            # Normalize lot size to broker requirements BEFORE logging or executing
            # This ensures the logs match the actual execution
            if hasattr(self.broker, 'normalize_lot_size'):
                raw_lot = lot_size
                lot_size = self.broker.normalize_lot_size(symbol, lot_size)
                if raw_lot != lot_size:
                     # Only log if significant change
                     if abs(raw_lot - lot_size) > 0.000001:
                        logger.debug(f"[LOT ADJUST] Normalized {raw_lot} -> {lot_size} to match broker requirements")
                        logger.debug(f"[DECISION_PATH]    Broker Normalized: {raw_lot:.4f} -> {lot_size:.4f}")

            # Only log lot size if the signal was just logged (i.e., it changed)
            if self._last_signal_logged:
                logger.info(f"[POSITION] SIZE CALCULATED: {lot_size} lots | Reason: {lot_reason}")
                logger.debug(f"[DECISION_PATH] 7. FINAL EXECUTION:")
                logger.debug(f"[DECISION_PATH]    Final Lot Size: {lot_size:.4f}")
                logger.debug(f"[DECISION_PATH]    Entry Price: ~{tick.get('ask' if signal.action == TradeAction.BUY else 'bid', 0):.2f}")
                logger.debug(f"[DECISION_PATH] ═══════════════════════════════════════════")


            # Final validation (is_recovery_trade=False for normal entries)
            # [OPTIMIZATION] For Continuous Scalping, we relax the cooldown check if the signal is strong
            # But we must respect the global cooldown to prevent API bans
            can_enter, entry_reason = self.validate_trade_entry(signal, lot_size, account_info, tick, is_recovery_trade=False)
            if not can_enter:
                # [DEBUG] Print rejection reason to console
                print(f">>> [DEBUG] Entry Blocked: {entry_reason}", flush=True)
                
                # [TELEMETRY] Log blocked entry for analysis
                if FLAGS.ENABLE_TELEMETRY:
                    self._telemetry.write(DecisionRecord(
                        ts=time.time(),
                        symbol=symbol,
                        action="entry_blocked",
                        side="buy" if signal.action == TradeAction.BUY else "sell",
                        price=tick['ask'] if signal.action == TradeAction.BUY else tick['bid'],
                        lots=lot_size,
                        features={"reason": entry_reason, "obi": tick.get('obi', 0.0)},
                        context={"regime": regime.name, "worker": worker_name},
                        decision={"blocked": True, "reason": entry_reason}
                    ))

                # Handle Cooldown Logs specifically (Throttled)
                if "cooldown" in entry_reason.lower():
                    current_time = time.time()
                    if current_time - self._last_cooldown_log_time > 5.0:
                        remaining_msg = ""
                        # Try to extract time from "Global cooldown: X.Xs < Y.Ys"
                        if "Global cooldown" in entry_reason and "<" in entry_reason:
                            try:
                                parts = entry_reason.split('<')
                                limit = float(parts[1].replace('s', '').strip())
                                current = float(parts[0].split(':')[1].replace('s', '').strip())
                                remaining = limit - current
                                remaining_msg = f" Resuming in {remaining:.1f}s"
                            except Exception as e:
                                logger.debug(f"[PAUSED] Failed parsing cooldown remaining time: {e}")
                        
                        logger.info(f"[PAUSED] Trading Halted: {entry_reason}.{remaining_msg}")
                        self._last_cooldown_log_time = current_time

                # Suppress repetitive blocking logs (cooldown, position exists) - only log once when signal changes
                elif "already exists" not in entry_reason.lower():
                    logger.info(f"[TRADE] ENTRY BLOCKED: {entry_reason}")
                elif "already exists" in entry_reason.lower() and self._last_signal_logged:
                    # Only log position blocking when signal just changed (not every cycle)
                    logger.info(f"[TRADE] ENTRY BLOCKED: {entry_reason}")
                return

            logger.debug(f"[TRADE] VALIDATION PASSED: All entry conditions met")
            
            # Update last_trade_time for global tracking
            self.last_trade_time = time.time()
            # Note: entry_cooldowns[symbol] already set atomically above

            # Execute trade
            # --- GENERATE ENTRY SUMMARY ---
            
            # 1. Calculate Virtual TP using IronShield (Same logic as execution)
            atr_value = 0.0
            try:
                atr_value = float(signal.metadata.get('atr', 0.0) or 0.0)
            except Exception:
                atr_value = 0.0

            if atr_value <= 0.0:
                atr_value = self.market_data.calculate_atr(symbol, 14) if hasattr(self.market_data, 'calculate_atr') else 0.0010
            
            # Convert ATR to points based on symbol type
            if 'JPY' in symbol or 'XAU' in symbol:
                atr_points = atr_value * 100
                pip_multiplier = 100
            else:
                atr_points = atr_value * 10000
                pip_multiplier = 10000
                
            # Get dynamic params from Shield
            zone_points, tp_points = shield.get_dynamic_params(atr_points)
            
            # [ADAPTIVE INTELLIGENCE] Apply TP/SL multipliers based on validation score
            tp_mult = signal.metadata.get('tp_multiplier', 1.0)
            sl_mult = signal.metadata.get('sl_multiplier', 1.0)
            
            original_tp = tp_points
            original_zone = zone_points
            
            tp_points = tp_points * tp_mult
            zone_points = zone_points * sl_mult  # Zone acts as SL distance
            
            # Log adaptive adjustments
            if tp_mult < 1.0 or sl_mult < 1.0:
                logger.debug(f"[ADAPTIVE] TP adjusted: {original_tp:.1f} → {tp_points:.1f} pips (x{tp_mult})")
                logger.debug(f"[ADAPTIVE] SL adjusted: {original_zone:.1f} → {zone_points:.1f} pips (x{sl_mult})")
            
            tp_pips = tp_points # This is in pips (e.g. 25.0 * 0.6 = 15.0 for scalp)
            
            is_buy = signal.action == TradeAction.BUY
            entry_price = tick['ask'] if is_buy else tick['bid']
            
            # Calculate Virtual TP Price
            if is_buy:
                virtual_tp_price = entry_price + (tp_pips / pip_multiplier)
            else:
                virtual_tp_price = entry_price - (tp_pips / pip_multiplier)

            # Generate Clean Entry Summary
            # Generate Clean Entry Summary with Live Data
            adaptive_strat = signal.metadata.get('adaptive_strategy', 'STANDARD')
            
            # Extract Live Metrics
            rsi_val = float(signal.metadata.get('rsi', 50.0))
            trend_val = float(signal.metadata.get('trend_strength', 0.0))
            oracle_conf = float(signal.metadata.get('oracle_confidence', 0.0))
            oracle_pred = signal.metadata.get('oracle_prediction', 'NEUTRAL')
            
            # Smart formatting
            rsi_desc = "Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral"
            trend_desc = "Strong UP" if trend_val > 0.5 else "Strong DOWN" if trend_val < -0.5 else "No Trend"
            
            summary = (
                f"\n>>> [AI ENTRY PLAN] 🧠 <<<\n"
                f"Action:        {signal.action.value} {lot_size} lots @ {entry_price:.5f}\n"
                f"Strategy:      {adaptive_strat} (Confidence: {validation_score:.0%})\n"
                f"Virtual TP:    {virtual_tp_price:.5f} (+{tp_pips:.1f} pips)\n"
                f"----------------------------------------------------\n"
                f"LIVE DATA SNAPSHOT:\n"
                f"• RSI (14):    {rsi_val:.1f} ({rsi_desc})\n"
                f"• Trend:       {trend_val:.3f} ({trend_desc})\n"
                f"• Oracle:      {oracle_pred} ({oracle_conf:.0%})\n"
                f"• Regime:      {signal.metadata.get('regime', 'Unknown')}\n"
                f"----------------------------------------------------\n"
                f"RATIONALE:     {signal.reason}\n"
                f"===================================================="
            )
            
            # [FIX] Windows Console Compatibility - Force clean version on Windows
            clean_summary = summary # Use same detailed summary for Windows now that we have good formatting
            
            import sys
            if sys.platform == 'win32':
                ui_logger.info(clean_summary)
            else:
                try:
                    ui_logger.info(summary)
                except Exception:
                    ui_logger.info(clean_summary)
            
            result = await self.execute_trade_entry(signal, lot_size, tick, strategist, shield)
            if result:
                logger.info(f"[TRADE] EXECUTION SUCCESSFUL: {signal.action.value} {signal.symbol} {lot_size} lots @ {tick['ask'] if signal.action == TradeAction.BUY else tick['bid']:.5f}")
                logger.info(f"   [EXIT] STRATEGY: Virtual TP/SL managed by bucket logic | AI Confidence: {signal.confidence:.3f}")
            else:
                logger.error(f"[TRADE] EXECUTION FAILED for {signal.symbol}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in trading cycle: {e}", flush=True)
            logger.error(f"Error in trading cycle: {e}")

    async def _record_market_data(self, symbol: str, tick: Dict) -> None:
        """Record tick and candle data to database."""
        if not self.db_queue:
            return

        # Record tick data
        tick_data = TickData(
            symbol=symbol,
            bid=tick['bid'],
            ask=tick['ask'],
            timestamp=time.time(),
            flags=0
        )
        await self.db_queue.add_tick(tick_data)

        # Record candle data if available
        if self.market_data.candles._cache:
            latest_candle = self.market_data.candles.get_latest_candle(symbol)
            if latest_candle:
                candle_data = CandleData(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    open_price=latest_candle.get('open', 0),
                    high=latest_candle.get('high', 0),
                    low=latest_candle.get('low', 0),
                    close=latest_candle.get('close', 0),
                    volume=latest_candle.get('volume', 0),
                    timestamp=time.time()
                )
                await self.db_queue.add_candle(candle_data)

    def _get_symbol_properties(self, symbol):
        """
        Returns correct pip value and point size for the symbol.
        CRITICAL: Distinguishes between Forex (10.0) and Gold (1.0).
        """
        symbol_upper = symbol.upper()
        
        if "XAU" in symbol_upper or "GOLD" in symbol_upper:
            return {
                "pip_value": 1.0,      # 1 pip = $1 per lot (approx) on Gold
                "point_size": 0.01,    # Price moves in cents
                "pip_size": 0.10       # Standard Gold Pip is 10 cents
            }
        elif "JPY" in symbol_upper:
            return {
                "pip_value": 9.0,      # Approx for JPY pairs
                "point_size": 0.001,
                "pip_size": 0.01
            }
        else: # Standard Forex (EURUSD, GBPUSD, etc.)
            return {
                "pip_value": 10.0,     # Standard Lot = $10/pip
                "point_size": 0.00001,
                "pip_size": 0.0001
            }

    async def _handle_blocked_hedge_strategy(self, symbol, positions, decision, tick):
        """
        [HIGHEST INTELLIGENCE] Plan B:
        When AI blocks a hedge, we don't just walk away. 
        We optimize the EXISTING positions to escape the danger zone.
        """
        reasons = decision.get("reasons", [])
        
        # STRATEGY 1: FAKEOUT DETECTED (Weak Trend)
        # If the hedge was blocked because the trend is weak (Low Confidence),
        # it means the AI predicts a Reversion (Bounce).
        if any("min_conf" in r for r in reasons):
            # We log this strategic shift.
            # In a fully automated version, this would send a 'ModifyOrder' to move TP to Break-Even.
            # For now, we activate "Reversion Escape Mode".
            if not getattr(self, f"_plan_b_active_{symbol}", False):
                logger.info(f"🧠 [AI PLAN B] Hedge blocked (Fakeout Detected). "
                            f"Strategy shifted to 'Mean Reversion Escape' for {symbol}. "
                            f"Expecting price bounce to exit in profit.")
                setattr(self, f"_plan_b_active_{symbol}", True)

        # STRATEGY 2: RISK LIMIT REACHED
        elif any("veto" in r for r in reasons):
            if not getattr(self, f"_plan_b_veto_{symbol}", False):
                logger.warning(f"🛡️ [AI PLAN B] Risk Governor active. "
                               f"Holding {symbol} positions defensively (No new risk added).")
                setattr(self, f"_plan_b_veto_{symbol}", True)

    async def _process_existing_positions(self, symbol: str, tick: Dict, shield, ppo_guardian, oracle=None, pressure_metrics=None) -> bool:
        """
        Process management for existing positions.
        Returns True if positions were managed (skipping new entries).
        """
        # Update positions from broker
        all_positions = self.broker.get_positions()
        
        # FAIL-SAFE: If broker returns None (error), DO NOT update or cleanup.
        # This prevents wiping state during temporary connection loss.
        if all_positions is None:
            logger.warning("[PROCESS_POS] Failed to fetch positions from broker - skipping update")
            return False

        # Always update positions, even if empty, to ensure closed positions are removed
        self.position_manager.update_positions(all_positions)
        # CRITICAL: Cleanup stale positions immediately after broker update
        self.position_manager.cleanup_stale_positions(all_positions)
        self.position_manager._update_bucket_stats()

        # Get positions for this symbol
        symbol_positions = self.position_manager.get_positions_for_symbol(symbol)

        logger.debug(f"[PROCESS_POS] Found {len(symbol_positions)} positions for {symbol}")

        if symbol_positions:
            # Sort positions by time to ensure correct order for hedging logic
            symbol_positions.sort(key=lambda p: p.time)

            # Get symbol properties for point value
            # Get Symbol Properties for Math
            props = self._get_symbol_properties(symbol)
            point_value = props['point_size']

            logger.debug(f"[PROCESS_POS] Calling process_position_management with {len(symbol_positions)} positions")
            
            # --- ELASTIC DEFENSE PROTOCOL INTEGRATION ---
            # [FRESHNESS] Enforce fresh data for critical position management (Hedging/Exit)
            # This ensures RSI/ATR are calculated on the absolute latest candle state.
            self.market_data.candles.get_history(symbol, force_refresh=True)

            # 1. Get Live Market Intelligence
            current_atr = self.market_data.calculate_atr(symbol)
            current_rsi = self.market_data.calculate_rsi(symbol)
            
            # [AI STATUS MONITOR] Throttled Analysis Update for Dashboard
            current_time = time.time()
            if not hasattr(self, '_last_analysis_update'): self._last_analysis_update = 0
            
            if current_time - self._last_analysis_update > 5.0: # Update every 5s
                self._last_analysis_update = current_time
                
                # Quick Regime Check
                supervisor_data = {
                    'atr': current_atr,
                    'trend_strength': self.market_data.calculate_trend_strength(symbol),
                    'volatility_ratio': self.market_data.get_volatility_ratio(),
                    'macro_context': self.market_data.get_macro_context(),
                    'pressure_metrics': pressure_metrics
                }
                regime = self.supervisor.detect_regime(supervisor_data)
                
                # Update Analysis State
                if not hasattr(self, 'latest_analysis'): self.latest_analysis = {}
                self.latest_analysis.update({
                    'regime': regime.name,
                    'pressure': pressure_metrics,
                    'rsi': current_rsi,
                    'atr': current_atr
                })

            if self._strict_entry:
                try:
                    if hasattr(self.market_data, 'calculate_atr_checked'):
                        atr_v, ok, areason = self.market_data.calculate_atr_checked(symbol)
                        if ok and atr_v is not None:
                            current_atr = atr_v
                        else:
                            logger.warning(f"[STRICT] ATR unavailable for recovery orders: {areason}")
                            current_atr = None
                    if hasattr(self.market_data, 'calculate_rsi_checked'):
                        rsi_v, ok, rreason = self.market_data.calculate_rsi_checked(symbol)
                        if ok and rsi_v is not None:
                            current_rsi = rsi_v
                        else:
                            logger.warning(f"[STRICT] RSI unavailable for recovery orders: {rreason}")
                            current_rsi = None
                except Exception as e:
                    logger.warning(f"[STRICT] Indicator check failed for recovery orders: {e}")
                    current_atr = None
                    current_rsi = None
            
            # 2. Check Dynamic Hedge Trigger (if we have positions)
            # We only intervene if the bucket is losing
            bucket_pnl = sum(p.profit for p in symbol_positions)
            
            # [AI STATUS MONITOR] Periodic Log
            current_time = time.time()
            if current_time - self._last_position_status_time >= 30.0: # Every 30s
                self._last_position_status_time = current_time
                
                # Determine Strategy
                strategy_status = "Monitoring"
                if bucket_pnl > 0:
                    strategy_status = "Profit Protection (Trailing)"
                elif len(symbol_positions) > 1:
                    strategy_status = "Zone Recovery (Hedging)"
                    if oracle:
                        strategy_status += " + Sniper Watch"
                
                # Format PnL
                pnl_str = f"${bucket_pnl:.2f}"
                if bucket_pnl > 0: pnl_str = f"+{pnl_str}"
                
                # Retrieve latest AI analysis
                analysis = getattr(self, 'latest_analysis', {})
                regime = analysis.get('regime', 'ANALYZING')
                oracle_pred = analysis.get('oracle_pred', 'NEUTRAL')
                oracle_conf = analysis.get('oracle_conf', 0.0)
                pressure = analysis.get('pressure', {})
                
                # Format Oracle
                oracle_str = f"{oracle_pred} ({oracle_conf:.2f})" if oracle_pred != "NEUTRAL" else "NEUTRAL"
                
                # Format Pressure
                pressure_str = "BALANCED"
                if pressure:
                    dom = pressure.get('dominance', 'NEUTRAL')
                    intensity = pressure.get('intensity', 'LOW')
                    pressure_str = f"{dom} ({intensity})"

                rsi_disp = "NA" if current_rsi is None else f"{float(current_rsi):.1f}"
                atr_disp = "NA" if current_atr is None else f"{float(current_atr):.4f}"
                
                # [AI STATUS MONITOR] Periodic Log
                total_positions = len(symbol_positions)
                
                status_msg = (
                    f"\n>>> [AI STATUS MONITOR] <<<\n"
                    f"Positions:    {total_positions} Active\n"
                    f"Net PnL:      {pnl_str}\n"
                    f"Strategy:     {strategy_status}\n"
                    f"Regime:       {regime}\n"
                    f"Oracle:       {oracle_str}\n"
                    f"Pressure:     {pressure_str}\n"
                    f"Market:       RSI {rsi_disp} | ATR {atr_disp}\n"
                )

                # Add specific AI focus for active positions
                if symbol_positions:
                    status_msg += f"AI Focus:     Monitoring {total_positions} positions for Exit/Hedge\n"
                    status_msg += f"Action:       Running Wick Intelligence & Profit Checks..."
                else:
                    status_msg += f"Action:       Scanning for Entry Opportunities..."
                
                status_msg += "\n----------------------------------------------------"

                
                # Demoted to DEBUG to rely on TraderDashboard for UI
                logger.debug(status_msg)

            # --- CENTRALIZED MANAGEMENT DELEGATION ---
            # All hedging, recovery, and exit logic must go through 'process_position_management'.
            # Previously, there was duplicate 'Elastic Defense' logic here that caused double-hedging
            # and bypassed the RiskManager's safety checks (cooldowns, freshness, smart timing).
            # We now strictly delegate everything to the core manager.
            
            # FIX: Robust conversion to dict (handling namedtuple vs object)
            pos_dicts = []
            for p in symbol_positions:
                if hasattr(p, '_asdict'):
                    pos_dicts.append(p._asdict())
                elif hasattr(p, '__dict__'):
                    pos_dicts.append(p.__dict__)
                elif isinstance(p, dict):
                    pos_dicts.append(p)
                else:
                    # Fallback (should not happen)
                    logger.warning(f"[POS_CONVERT] Unknown position type: {type(p)}")
                    continue

            positions_managed = await self.process_position_management(
                symbol, pos_dicts, tick,
                point_value, shield, ppo_guardian, rsi_value=current_rsi, oracle=oracle, pressure_metrics=pressure_metrics, trap_hunter=self.trap_hunter
            )
            
            logger.debug(f"[PROCESS_POS] process_position_management returned: {positions_managed}")
            
            if positions_managed:
                logger.info(f"[POSITION] MANAGEMENT EXECUTED for {symbol} - Skipping new entries")
                return True
        
        return False

    def _calculate_indicators(self, symbol: str) -> Tuple[float, float]:
        """Calculate ATR and trend strength."""
        atr_value = self.market_data.calculate_atr(symbol, 14) if hasattr(self.market_data, 'calculate_atr') else 0.0010
        trend_strength = self.market_data.calculate_trend_strength(symbol, 20) if hasattr(self.market_data, 'calculate_trend_strength') else 0.0
        return atr_value, trend_strength