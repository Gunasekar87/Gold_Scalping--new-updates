"""
Position Manager - Handles position tracking, bucket management, and trade state.

This module provides comprehensive position management including:
- Position state tracking and validation
- Bucket-based profit taking logic
- Trade state persistence and recovery
- Position closing coordination

Author: AETHER Development Team
License: MIT
Version: 1.0.0
"""

import time
import logging
import json
import os
import math
import asyncio
from collections import deque
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from threading import Lock, RLock
from enum import Enum
import MetaTrader5 as mt5

# Import enhanced trade explainer for detailed logging
try:
    from .utils.trade_explainer import TradeExplainer
    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False

from .ai_core.architect import Architect

from .core.trade_authority import TradeAuthority
from .core.bad_bank import BadBank

# Import TradingLogger for structured exit summaries
try:
    from .utils.trading_logger import TradingLogger, format_pips
except ImportError:
    # Fallback if import fails
    TradingLogger = None
    format_pips = None

logger = logging.getLogger("PositionManager")
# [CRITICAL] Get the specific UI logger that run_bot.py listens to
ui_logger = logging.getLogger("AETHER_UI")


class PositionState(Enum):
    """Enumeration of possible position states."""
    OPENING = "opening"  # Position order sent, not yet confirmed
    SINGLE_ACTIVE = "single_active"  # Single position with broker TP/SL
    TRANSITIONING = "transitioning"  # Removing broker TP/SL, adding hedge
    BUCKET_ACTIVE = "bucket_active"  # Multiple positions, Python monitoring
    BUCKET_FROZEN = "bucket_frozen"  # [VALKYRIE] Frozen by Perfect Hedge
    PENDING_CLOSE = "pending_close"  # Close order sent
    CLOSED = "closed"  # Position closed
    UNKNOWN = "unknown"  # Unable to determine state
    
    @property
    def uses_broker_tp_sl(self) -> bool:
        """Check if this state uses broker TP/SL."""
        return self == PositionState.SINGLE_ACTIVE
    
    @property
    def uses_python_monitoring(self) -> bool:
        """Check if this state uses Python monitoring."""
        return self in (PositionState.BUCKET_ACTIVE, PositionState.TRANSITIONING)


class BucketMode(Enum):
    """Mode of operation for position buckets."""
    SINGLE = "single"  # Single position, broker TP/SL active
    BUCKET = "bucket"  # Multiple positions, Python monitoring


@dataclass
class Position:
    """Represents a trading position with all relevant data."""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    price_open: float
    price_current: float
    profit: float
    sl: float
    tp: float
    time: float
    swap: float = 0.0  # [FIX] Added swap field
    commission: float = 0.0  # [FIX] Added commission field
    magic: int = 0  # [FIX] Added magic number field
    comment: str = ""
    state: PositionState = PositionState.SINGLE_ACTIVE
    # Store fixed TP/SL targets set at entry (not broker's current TP/SL which may be 0)
    entry_tp_pips: float = 0.0  # TP distance in pips (e.g., 547.5 for 0.3 ATR)
    entry_sl_pips: float = 0.0  # SL distance in pips (e.g., 1825 for 1.0 ATR)
    entry_atr: float = 0.0  # ATR value at entry time

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy position."""
        return self.type == 0

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell position."""
        return self.type == 1


@dataclass
class BucketStats:
    """Statistics for a position bucket."""
    bucket_id: str
    positions: List[int]  # Ticket numbers
    net_profit: float
    open_time: float
    last_update: float
    closed: bool = False
    state: PositionState = PositionState.UNKNOWN  # Current state in state machine
    mode: BucketMode = BucketMode.SINGLE  # Track if single or bucket
    last_state_check: float = 0.0  # Last time state was validated
    exit_reason: str = ""  # Exit criteria that triggered close
    exit_confidence: float = 0.0  # AI confidence at exit trigger
    last_recovery_time: float = 0.0  # Last time calculated recovery was executed


class PositionManager:
    """
    Manages trading positions and bucket-based profit

    This class handles:
    - Position state tracking and validation
    - Bucket creation and management
    - Profit taking logic with AI integration
    - State persistence for crash recovery
    """

    def __init__(self, mt5_adapter=None, state_file: str = "data/position_state.json", callbacks: Dict[str, Callable] = None):
        self.state_file = state_file
        self._lock = RLock()  # Reentrant lock for thread safety
        self._position_locks: Dict[int, Lock] = {}  # Per-position locks for synchronous operations
        self._state_transition_lock = Lock()  # Protects state transitions
        
        # INTEGRATION FIX: Callbacks for trading_engine integration
        self.callbacks = callbacks or {}
        
        # v5.5.0: Initialize Architect
        # v5.5.0: Initialize Architect
        self.architect = Architect(mt5_adapter) if mt5_adapter else None
        
        # [SUPREME COURT] Initialize Trade Authority
        self.trade_authority = TradeAuthority()
        
        # [BAD BANK] Initialize Bad Bank (Debt Management)
        self.bad_bank = BadBank()

        # Core state
        self.active_positions: Dict[int, Position] = {}
        self.bucket_stats: Dict[str, BucketStats] = {}
        self.closed_buckets: Set[str] = set()
        self.pending_closes: Dict[str, float] = {}  # symbol -> timestamp

        # Learning state
        self.active_learning_trades: Dict[int, Dict] = {}
        
        # Ghost tickets (positions that broker reports but don't exist)
        self._ghost_tickets: Set[int] = set()
        
        # [TIMEZONE AUTO-CORRECTION]
        self._time_offset: Optional[float] = None
        
        # [INTELLIGENCE] High Water Mark Memory (Profit Ratchet)
        # Tracks the highest Net PnL seen for each bucket to prevent round-tripping profits
        self.high_water_marks: Dict[str, float] = {}

        # [SLIPPAGE CALIBRATION] Rolling close slippage samples (per-lot USD)
        self._slippage_enabled = str(os.getenv("AETHER_ENABLE_SLIPPAGE_CALIBRATION", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._slippage_window = max(10, int(os.getenv("AETHER_SLIPPAGE_CALIBRATION_WINDOW", "200")))
        self._slippage_p95_multiplier = float(os.getenv("AETHER_SLIPPAGE_P95_MULTIPLIER", "1.25"))
        self._slippage_samples_per_lot_usd: Dict[str, deque] = {}

        # Ensure data directory exists
        os.makedirs(os.path.dirname(state_file), exist_ok=True)

        # Load persisted state
        self._load_state()

    def _p95(self, values: List[float]) -> float:
        """
        Calculate 95th percentile of a list of values.
        """
        if not values:
            return 0.0
        
        try:
            sorted_vals = sorted(values)
            index = int(0.95 * len(sorted_vals))
            # Clamp index
            index = max(0, min(index, len(sorted_vals) - 1))
            return sorted_vals[index]
        except Exception as e:
            logger.error(f"Error calculating p95: {e}")
            return 0.0

    def _calibrated_profit_buffer(self, symbol: str, total_volume: float, base_buffer_usd: float) -> float:
        """
        Calculate calibrated profit buffer based on volume and symbol slippage stats.
        Uses p95 of historical slippage if available.
        """
        if not self._slippage_enabled or not symbol:
             return base_buffer_usd

        with self._lock:
            samples = list(self._slippage_samples_per_lot_usd.get(symbol, []))

        if len(samples) < 10:
            # Not enough data, use default heuristic
            # XAUUSD/Gold typically has higher spread/slippage
            is_gold = "XAU" in symbol or "GOLD" in symbol
            slippage_per_lot = 15.0 if is_gold else 5.0
            volume_component = float(total_volume) * slippage_per_lot
            return max(float(base_buffer_usd), volume_component)

        p95_per_lot = self._p95(samples)
        
        # Calculate expected slippage cost
        calibrated_cost = p95_per_lot * float(total_volume) * float(self._slippage_p95_multiplier)
        
        # Buffer must cover EITHER the base need OR the calibrated slippage, whichever is higher
        return max(float(base_buffer_usd), calibrated_cost)

    def _get_slippage_sample_count(self, symbol: str) -> int:
        if not self._slippage_enabled or not symbol:
            return 0
        with self._lock:
            dq = self._slippage_samples_per_lot_usd.get(symbol)
            return len(dq) if dq is not None else 0

    def _get_slippage_p95_per_lot_usd(self, symbol: str) -> float:
        if not self._slippage_enabled or not symbol:
            return 0.0
        with self._lock:
            samples = list(self._slippage_samples_per_lot_usd.get(symbol, []))
        if len(samples) < 10:
            return 0.0
        try:
            return self._p95(samples)
        except Exception:
            return 0.0

    def _load_state(self) -> None:
        """Load position state from disk."""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Restore bucket stats
            raw_stats = state.get('bucket_stats', {})
            for bucket_id, stats_data in raw_stats.items():
                # Convert state and mode from string values back to enums
                if 'state' in stats_data:
                    try:
                        stats_data['state'] = PositionState(stats_data['state'])
                    except (ValueError, KeyError):
                        stats_data['state'] = PositionState.UNKNOWN
                else:
                    stats_data['state'] = PositionState.UNKNOWN
                
                if 'mode' in stats_data:
                    try:
                        stats_data['mode'] = BucketMode(stats_data['mode'])
                    except (ValueError, KeyError):
                        stats_data['mode'] = BucketMode.SINGLE
                else:
                    stats_data['mode'] = BucketMode.SINGLE
                
                # Provide defaults for new fields
                stats_data.setdefault('last_state_check', time.time())
                stats_data.setdefault('exit_reason', '')
                stats_data.setdefault('last_recovery_time', 0.0)
                
                self.bucket_stats[bucket_id] = BucketStats(**stats_data)

            # Restore closed buckets
            self.closed_buckets = set(state.get('closed_buckets', []))
            
            # Restore learning data (CRITICAL for plan persistence)
            self.active_learning_trades = {
                int(k): v for k, v in state.get('active_learning_trades', {}).items()
            }

            # Restore slippage calibration samples (optional; best-effort)
            if self._slippage_enabled:
                raw_slip = state.get('slippage_samples_per_lot_usd', {})
                if isinstance(raw_slip, dict):
                    for sym, samples in raw_slip.items():
                        if not sym or not isinstance(samples, list):
                            continue
                        try:
                            dq = deque(maxlen=self._slippage_window)
                            for v in samples[-self._slippage_window:]:
                                try:
                                    fv = float(v)
                                except Exception:
                                    continue
                                if fv < 0 or math.isinf(fv) or math.isnan(fv):
                                    continue
                                dq.append(fv)
                            if len(dq) > 0:
                                self._slippage_samples_per_lot_usd[sym] = dq
                        except Exception:
                            continue

            logger.info(f"Position state restored: {len(self.bucket_stats)} buckets, {len(self.active_learning_trades)} learning records")

        except Exception as e:
            logger.error(f"Failed to load position state: {e}")

    def get_total_positions(self) -> int:
        """Returns the total number of active positions tracked."""
        with self._lock:
            return len(self.active_positions)

    def _calculate_drawdown(self, balance: float, equity: float) -> float:
        """Calculate current drawdown percentage."""
        if balance <= 0:
            return 0.0
        drawdown_amount = max(0.0, balance - equity)
        return drawdown_amount / balance

    def _save_state(self) -> None:
        """Persist current state to disk."""
        try:
            state = {
                'bucket_stats': {
                    bid: {
                        'bucket_id': stats.bucket_id,
                        'positions': stats.positions,
                        'net_profit': stats.net_profit,
                        'open_time': stats.open_time,
                        'last_update': stats.last_update,
                        'closed': stats.closed,
                        'state': stats.state.value if hasattr(stats, 'state') else PositionState.UNKNOWN.value,
                        'mode': stats.mode.value if hasattr(stats, 'mode') else BucketMode.SINGLE.value,
                        'last_state_check': getattr(stats, 'last_state_check', time.time()),
                        'exit_reason': getattr(stats, 'exit_reason', ''),
                        'last_recovery_time': getattr(stats, 'last_recovery_time', 0.0)
                    }
                    for bid, stats in self.bucket_stats.items()
                },
                'closed_buckets': list(self.closed_buckets),
                'active_learning_trades': self.active_learning_trades,
                'timestamp': time.time()
            }

            # Persist slippage calibration samples (bounded, optional)
            if self._slippage_enabled:
                with self._lock:
                    state['slippage_samples_per_lot_usd'] = {
                        sym: list(dq)
                        for sym, dq in self._slippage_samples_per_lot_usd.items()
                        if sym and dq
                    }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save position state: {e}")

    def calculate_bucket_drawdown(self, bucket_id: str, market_data: Dict) -> float:
        """
        [PHASE 2] Calculates current drawdown for a bucket in USD.
        Used by 'The Stabilizer' to trigger hedging.
        """
        if bucket_id not in self.bucket_stats:
            return 0.0
            
        stats = self.bucket_stats[bucket_id]
        positions = [self.active_positions[t] for t in stats.positions if t in self.active_positions]
        
        if not positions:
            return 0.0
            
        # Calculate Net PnL
        net_pnl, _, _, _ = self.calculate_net_pnl(positions)
        
        # Drawdown is negative PnL
        if net_pnl < 0:
            return abs(net_pnl)
        return 0.0

    def check_stabilizer_trigger(self, bucket_id: str, market_data: Dict) -> bool:
        """
        [PHASE 2] The Stabilizer: Checks if hedging is needed.
        Trigger: Drawdown > 1.5 * ATR_Value_USD
        """
        with self._lock:
            if bucket_id not in self.bucket_stats:
                return False
                
            stats = self.bucket_stats[bucket_id]
            
            # Only trigger if we are in SINGLE mode (don't hedge a hedge)
            if stats.mode != BucketMode.SINGLE:
                return False
                
            positions = [self.active_positions[t] for t in stats.positions if t in self.active_positions]
            if not positions:
                return False

            # Safety: enforce per-symbol max positions to prevent recovery from bypassing zone limits.
            # This cap is intended to be TOTAL positions on the symbol (base + hedges + recovery).
            max_positions_per_symbol = market_data.get('max_positions_per_symbol')
            if max_positions_per_symbol is None:
                try:
                    max_positions_per_symbol = int(os.getenv('AETHER_MAX_POSITIONS_PER_SYMBOL', '0'))
                except Exception:
                    max_positions_per_symbol = 0
            try:
                max_positions_per_symbol = int(max_positions_per_symbol) if max_positions_per_symbol is not None else 0
            except Exception:
                max_positions_per_symbol = 0
            if max_positions_per_symbol and max_positions_per_symbol > 0:
                if len(positions) >= max_positions_per_symbol:
                    logger.warning(
                        f"[RISK] Calculated recovery blocked: max positions per symbol reached "
                        f"({len(positions)}/{max_positions_per_symbol})"
                    )
                    return False
                
            # Calculate Drawdown
            drawdown = self.calculate_bucket_drawdown(bucket_id, market_data)
            
            # Calculate Threshold (4.0 * ATR in USD) - [TUNED FOR GOLD]
            strict_entry = bool(market_data.get('strict_entry', False))
            atr_ok = bool(market_data.get('atr_ok', True))
            atr_value = market_data.get('atr', None)

            if strict_entry:
                # Do not allow new-risk triggers to be driven by synthetic/default ATR.
                if not atr_ok or atr_value is None or float(atr_value) <= 0:
                    return False

            if atr_value is None:
                # Legacy fallback for non-strict mode
                atr_value = 0.0010

            total_volume = sum(p.volume for p in positions)
            symbol = positions[0].symbol
        
        # Contract Size
        contract_size = 100 if "XAU" in symbol or "GOLD" in symbol else 100000
        
        # ATR in USD = ATR_Price * Contract_Size * Volume
        atr_usd = atr_value * contract_size * total_volume
        
        # [CRITICAL] Dynamic Threshold based on Volatility
        # Base multiplier 4.0x (Gold breathing room)
        # If market is VIOLENT (VolRatio > 1.5), widen to 6.0x to avoid whipping.
        vol_ratio = float(market_data.get('volatility_ratio', 1.0))
        base_mult = 4.0
        if vol_ratio > 1.5:
            base_mult = 6.0
            
        threshold = atr_usd * base_mult
        
        if drawdown > threshold:
            logger.warning(f"[STABILIZER] Triggered! Drawdown ${drawdown:.2f} > Threshold ${threshold:.2f} ({base_mult}x ATR)")
            return True
            
        return False

    async def execute_perfect_hedge(self, broker, bucket_id: str, market_data: Dict) -> bool:
        """
        [VALKYRIE PROTOCOL] THE PERFECT HEDGE (Flash Freeze).
        
        Goal: Neutralize Net Exposure to 0.0.
        Action: Open ONE trade that counter-balances the exact net delta.
        Result: Account PnL is frozen.
        """
        with self._lock:
            if bucket_id not in self.bucket_stats:
                return False
            
            stats = self.bucket_stats[bucket_id]
            # Use get() to handle missing keys safely
            positions = [
                self.active_positions[t] 
                for t in stats.positions 
                if t in self.active_positions
            ]
            
            if not positions:
                return False
                
            # 1. Calculate Net Volume
            net_vol = 0.0
            for p in positions:
                vol = p.volume
                if p.type == 1: # SELL
                    vol = -vol
                net_vol += vol
                
            net_vol = round(net_vol, 2)
            
            # 2. Determine Freeze Action
            if abs(net_vol) < 0.01:
                logger.info(f"[VALKYRIE] Account already neutral (Net: {net_vol}). Freeze successful.")
                return True
                
            action = "SELL" if net_vol > 0 else "BUY"
            freeze_vol = abs(net_vol)
            symbol = positions[0].symbol
            
            logger.critical(f"❄️ [VALKYRIE] EXECUTING FREEZE | Net: {net_vol} | Action: {action} {freeze_vol} lots")
            
            # 3. Execute The Freeze
            # Bypass Supreme Court? NO. The Court should allow this as it REDUCES risk (delta -> 0).
            # But technically it increases 'global positions'.
            # We must force this trade.
            
            # Get Price
            price = market_data.get('bid') if action == "SELL" else market_data.get('ask')
            
            result = broker.execute_order(
                action="OPEN",
                symbol=symbol,
                order_type=action,
                price=price,
                volume=freeze_vol,
                sl=0.0, tp=0.0,
                comment="VALKYRIE_FREEZE",
                trace_reason="VALKYRIE_PROTOCOL_ENGAGED"
            )
            
            if result:
                logger.critical(f"❄️ [VALKYRIE] ACCOUNT FROZEN. WAITING FOR PHASE 3 (BAD BANK).")
                # Mark bucket as frozen
                self.bucket_stats[bucket_id].state = PositionState.BUCKET_FROZEN # [VALKYRIE] Frozen
                
                # [BAD BANK] Register Toxic Asset
                try:
                    self.bad_bank.register_toxic_asset(bucket_id, positions)
                except Exception as e:
                    logger.error(f"[BAD BANK] Failed to register asset {bucket_id}: {e}")
                    
                return True
                
            return False

    async def execute_calculated_recovery(self, broker, bucket_id: str, market_data: Dict, shield=None) -> bool:
        """
        [GOD MODE] LIQUIDITY VACUUM RECOVERY (The Muscle Upgrade)
        Uses Institutional Market Structure to time recovery entries.
        Replaces blind hedging/grid with precision structure-based averaging.
        """
        with self._lock:
            if bucket_id not in self.bucket_stats:
                return False
                
            stats = self.bucket_stats[bucket_id]
            
            # 1. Cooldown & Status Checks
            # [FIX] Use specific recovery timestamp
            time_since_recovery = time.time() - getattr(stats, 'last_recovery_time', 0.0)
            if time_since_recovery < 300: # 5 mins cooldown
                # logger.debug(f"[GOD MODE] Cooldown active for {bucket_id}")
                return False

            positions = [self.active_positions[t] for t in stats.positions if t in self.active_positions]
            if not positions:
                return False
                
            # Calculate Net Deficit
            # [FIX] Use getattr for safety against stale objects
            net_pnl = sum(p.profit for p in positions) + sum(getattr(p, 'swap', 0.0) for p in positions) + sum(getattr(p, 'commission', 0.0) for p in positions)
            
            if net_pnl >= 0:
                return False # Not in loss
                
            deficit = abs(net_pnl)
            symbol = positions[0].symbol
            # Net Volume: Positive = Net Long, Negative = Net Short
            net_vol = sum(p.volume if p.type == 0 else -p.volume for p in positions)
            
            # 2. GOD MODE: Structure Analysis
            strict_entry = bool(market_data.get('strict_entry', False))
            candles = market_data.get('candles') or []
            atr = market_data.get('atr', None)
            current_price = market_data.get('current_price', None)

            # Freshness gate: block ANY NEW recovery order if feed is stale
            enable_freshness = str(os.getenv("AETHER_ENABLE_FRESHNESS_GATE", "1")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if enable_freshness:
                now = time.time()
                tick_ts = float(market_data.get('time', 0.0) or 0.0)

                # [TIMEZONE AUTO-CORRECTION] Prefer explicit override; else auto-detect once.
                env_offset = os.getenv("AETHER_TIME_OFFSET_SECS", None)
                if env_offset is not None:
                    try:
                        self._time_offset = float(env_offset)
                    except Exception:
                        self._time_offset = None

                if self._time_offset is None and tick_ts > 0:
                    raw_diff = now - tick_ts
                    if abs(raw_diff) > 600:
                        self._time_offset = raw_diff
                        logger.debug(f"[FRESHNESS] Detected Timezone Offset: {self._time_offset:.2f}s. Adjusting...")
                    else:
                        self._time_offset = 0.0

                offset = self._time_offset if self._time_offset is not None else 0.0

                try:
                    if tick_ts > 0:
                        adjusted_ts = tick_ts + offset
                        tick_age = abs(now - adjusted_ts)
                    else:
                        tick_age = float(market_data.get('tick_age_s', float('inf')))
                except Exception:
                    tick_age = float('inf')

                try:
                    candle_close_age = float(market_data.get('candle_close_age_s', float('inf')))
                except Exception:
                    candle_close_age = float('inf')

                # Strict check for recovery
                try:
                    max_tick_age = float(os.getenv("AETHER_FRESH_TICK_MAX_AGE_S", os.getenv("AETHER_STRICT_TICK_MAX_AGE_S", "5.0")))
                except Exception:
                    max_tick_age = 5.0
                try:
                    max_candle_age = float(os.getenv("AETHER_FRESH_CANDLE_CLOSE_MAX_AGE_S", "0"))
                except Exception:
                    max_candle_age = 0.0
                if not max_candle_age or max_candle_age <= 0:
                    try:
                        tf_s = float(market_data.get('timeframe_s', 60) or 60)
                    except Exception:
                        tf_s = 60.0
                    max_candle_age = max((2.0 * tf_s) + 10.0, 30.0)

                if max_tick_age > 0 and tick_age > max_tick_age:
                    logger.warning(f"[FRESHNESS] Calculated recovery blocked: stale tick age={tick_age:.2f}s max={max_tick_age:.2f}s (offset={offset:.2f}s)")
                    return False
                if max_candle_age > 0 and candle_close_age > max_candle_age:
                    logger.warning(f"[FRESHNESS] Calculated recovery blocked: stale candle close age={candle_close_age:.2f}s max={max_candle_age:.2f}s")
                    return False

            atr_ok = bool(market_data.get('atr_ok', True))
            trend_ok = bool(market_data.get('trend_ok', True))
            rsi_ok = bool(market_data.get('rsi_ok', True))

            # Default to "Wait"
            signal = False
            # If Net Long (>0), we BUY. If Net Short (<0), we SELL.
            action = "BUY" if net_vol > 0 else "SELL" 
            
            # [FIX] Trend Veto (IronShield) - Only if strict
            if strict_entry and shield:
                if not trend_ok:
                     return False
                trend_direction = float(market_data.get('trend_direction', 0.0) or 0.0)
                if action == "BUY" and trend_direction < -0.6: return False
                if action == "SELL" and trend_direction > 0.6: return False

            # STRUCTURE-BASED TRIGGER
            if candles and len(candles) >= 20:
                last_candle = candles[-1]
                candle_range = last_candle['high'] - last_candle['low']
                is_compressed = candle_range < (atr * 2.0) if atr else True
                
                highs = [c['high'] for c in candles[-20:]]
                lows = [c['low'] for c in candles[-20:]]
                lowest_low = min(lows)
                highest_high = max(highs)
                
                if action == "BUY": 
                    is_at_structure = current_price <= (lowest_low + (atr * 0.5 if atr else 0.0))
                    # Simplified reversal check: just green candle is enough in recovery mode
                    is_reversal = last_candle['close'] >= last_candle['open'] 
                    if is_at_structure and is_reversal and is_compressed:
                        signal = True
                        logger.info(f"[GOD MODE] BUY Signal: Liquidity Sweep @ {lowest_low:.2f}")
                else: 
                    is_at_structure = current_price >= (highest_high - (atr * 0.5 if atr else 0.0))
                    is_reversal = last_candle['close'] <= last_candle['open']
                    if is_at_structure and is_reversal and is_compressed:
                        signal = True
                        logger.info(f"[GOD MODE] SELL Signal: Liquidity Sweep @ {highest_high:.2f}")

            # EMERGENCY FALLBACK TRIGGER
            # If we are failing to trigger via structure but Deficit is huge (> 5% Equity), FORCE RECOVERY.
            # This ensures we don't bleed to death waiting for a perfect candle.
            acct_equity = market_data.get('equity', 1000.0) # Need equity passed in market_data ideally
            # Approximate equity check if not in market_data
            if not signal:
                 # Check if deficit is critical
                 if deficit > (acct_equity * 0.20):
                     # [OPTIMIZATION] Check Trade Authority Limits FIRST to avoid Veto Loop
                     # If we are already at max positions, attempting recovery is futile and causes log spam.
                     if self.trade_authority and hasattr(self.trade_authority, 'current_global_cap'):
                         current_count = self.get_total_positions()
                         max_cap = self.trade_authority.current_global_cap
                         
                         if current_count >= max_cap:
                             # Throttle "Capped" Log
                             if not hasattr(self, '_last_capped_log'): self._last_capped_log = 0
                             if time.time() - self._last_capped_log > 60:
                                 logger.warning(f"[GOD MODE] EMERGENCY CRITICAL: Deficit ${deficit:.2f} but Global Cap Reached ({current_count}/{max_cap}). Recovery Postponed.")
                                 self._last_capped_log = time.time()
                             return False

                     signal = True
                     # Throttle Log (prevent spam loop with TradeAuthority)
                     if not hasattr(self, '_last_emergency_log'): self._last_emergency_log = 0
                     if time.time() - self._last_emergency_log > 60:
                         logger.warning(f"[GOD MODE] EMERGENCY TRIGGER: Deficit ${deficit:.2f} > 20% Equity. Forcing Recovery.")
                         self._last_emergency_log = time.time()
                 
                 # Also fallback to time-based if > 30 mins
                 duration_mins = (time.time() - stats.open_time) / 60
                 if duration_mins > 30:
                     signal = True
                     logger.info(f"[GOD MODE] Time-based Trigger: >30mins in drawdown")

            if not signal:
                return False

        # 3. Calculate Recovery Volume (Outside Lock)
        # Aim to recover 50% of deficit with a 3.0 swing (Gold)
        target_recovery = deficit * 0.5
        swing_price = 3.00 
        contract_size = 100 
        
        recovery_volume = target_recovery / (swing_price * contract_size)
        recovery_volume = round(recovery_volume, 2)
        
        # Smart Sizing: Don't exceed 1.0x existing volume (Martingale Limit)
        total_existing_vol = sum(p.volume for p in positions)
        if recovery_volume > total_existing_vol:
            recovery_volume = total_existing_vol
            
        # Hard Caps
        if recovery_volume > 0.5: recovery_volume = 0.5 # Hard cap for safety
        if recovery_volume < 0.01: recovery_volume = 0.01
        
        # 4. Execute Recovery Trade
        ask = market_data.get('ask')
        bid = market_data.get('bid')
        if strict_entry and (ask is None or bid is None):
            logger.warning("[GOD MODE] Strict: Recovery blocked (missing bid/ask)")
            return False

        price = ask if action == "BUY" else bid
        
        # [FIX] Price Sanity Check
        # Prevent "Suicide Recovery" at bad prices
        ask = market_data.get('ask')
        bid = market_data.get('bid')
        
        if action == "BUY":
            # Check 1: Spread (Proxy for bad liquidity/spikes)
            if (ask - bid) > 1.0: # > 100 points (Gold)
                logger.warning(f"🛑 [SAFETY] Recovery BUY blocked. Spread too high: {ask-bid:.2f}")
                return False

        # Check 2: Freshness gate (ANY new recovery order must use fresh tick + fresh candles)
        enable_freshness = str(os.getenv("AETHER_ENABLE_FRESHNESS_GATE", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if enable_freshness:
            # [TIMEZONE AUTO-CORRECTION]
            now = time.time()
            tick_ts = float(market_data.get('time', 0.0) or 0.0)
            
            if self._time_offset is None and tick_ts > 0:
                raw_diff = now - tick_ts
                if abs(raw_diff) > 600:
                    self._time_offset = raw_diff
                    logger.debug(f"[FRESHNESS] Detected Timezone Offset: {self._time_offset:.2f}s. Adjusting...")
                else:
                    self._time_offset = 0.0
            
            offset = self._time_offset if self._time_offset is not None else 0.0

            try:
                # If we have a raw timestamp, calculate age using offset
                if tick_ts > 0:
                    adjusted_ts = tick_ts + offset
                    tick_age = abs(now - adjusted_ts)
                else:
                    # Fallback to pre-calculated age (which might be wrong if it didn't use offset)
                    tick_age = float(market_data.get('tick_age_s', float('inf')))
            except Exception:
                tick_age = float('inf')

            try:
                candle_close_age = float(market_data.get('candle_close_age_s', float('inf')))
            except Exception:
                candle_close_age = float('inf')

            try:
                max_tick_age = float(os.getenv("AETHER_FRESH_TICK_MAX_AGE_S", os.getenv("AETHER_STRICT_TICK_MAX_AGE_S", "2.5")))
            except Exception:
                max_tick_age = 2.5

            try:
                max_candle_age = float(os.getenv("AETHER_FRESH_CANDLE_CLOSE_MAX_AGE_S", "0"))
            except Exception:
                max_candle_age = 0.0

            if not max_candle_age or max_candle_age <= 0:
                try:
                    tf_s = float(market_data.get('timeframe_s', 60) or 60)
                except Exception:
                    tf_s = 60.0
                max_candle_age = max((2.0 * tf_s) + 10.0, 30.0)

            if max_tick_age > 0 and tick_age > max_tick_age:
                logger.warning(f"🛑 [SAFETY] Recovery {action} blocked. Stale Tick (age={tick_age:.1f}s max={max_tick_age:.1f}s)")
                return False
            if max_candle_age > 0 and candle_close_age > max_candle_age:
                logger.warning(f"🛑 [SAFETY] Recovery {action} blocked. Stale Candle Close (age={candle_close_age:.1f}s max={max_candle_age:.1f}s)")
                return False
        
        logger.info(f"[GOD MODE] Executing Liquidity Recovery: {action} {recovery_volume} lots @ {price}")

        obi_applicable = bool(market_data.get('obi_applicable', False)) if isinstance(market_data, dict) else False
        if strict_entry:
            atr_ok = bool(market_data.get('atr_ok', False))
            rsi_ok = bool(market_data.get('rsi_ok', False))
        else:
            atr_ok = bool(market_data.get('atr_ok', True))
            rsi_ok = bool(market_data.get('rsi_ok', True))
        obi_ok = bool(market_data.get('obi_ok', False))
        strict_ok = (not bool(strict_entry)) or (atr_ok and rsi_ok and ((not obi_applicable) or obi_ok))
        
        strict_ok = (not bool(strict_entry)) or (atr_ok and rsi_ok and ((not obi_applicable) or obi_ok))
        
        # [SUPREME COURT] Constitution Check
        # Before we execute, we MUST ask the Trade Authority.
        approved, reason = self.trade_authority.check_constitution(broker, symbol, recovery_volume, "OPEN")
        if not approved:
            logger.warning(f"⚖️ [GOD MODE] BLOCKED by Supreme Court: {reason}")
            return False
        
        result = broker.execute_order(
            action="OPEN",
            symbol=symbol,
            order_type=action,
            price=price,
            volume=recovery_volume,
            sl=0.0,
            tp=0.0,
            strict_entry=bool(strict_entry),
            strict_ok=bool(strict_ok),
            atr_ok=bool(atr_ok),
            rsi_ok=bool(rsi_ok),
            obi_ok=bool(obi_ok) if obi_applicable else None,
            trace_reason="OPEN_LIQUIDITY_RECOVERY",
            comment="Liquidity Recovery"
        )
        
        if result:
            with self._lock:
                # Re-fetch stats to ensure we update the correct object
                if bucket_id in self.bucket_stats:
                    self.bucket_stats[bucket_id].last_update = time.time()
                    self.bucket_stats[bucket_id].last_recovery_time = time.time()
            
            # [FIX] Update TPs immediately for the new bucket composition
            # This ensures we don't leave stale (low) TPs on the broker
            logger.info(f"[GOD MODE] Updating Bucket TPs after recovery trade...")
            await self.update_bucket_tp(broker, symbol, bucket_id, price)

            return True
            
        return False

    def record_trade_metadata(self, ticket: int, metadata: Dict) -> None:
        """
        Thread-safe method to record trade metadata and persist state.
        CRITICAL for preserving the 'Virtual Target' plan across restarts.
        """
        with self._lock:
            self.active_learning_trades[ticket] = metadata
            self._save_state()
            logger.debug(f"Persisted metadata for ticket #{ticket}")

    def _get_position_lock(self, ticket: int) -> Lock:
        """
        Get or create a lock for a specific position.
        Ensures atomic operations on individual positions.
        """
        if ticket not in self._position_locks:
            self._position_locks[ticket] = Lock()
        return self._position_locks[ticket]

    def _get_position_state(self, position: Position) -> PositionState:
        """
        Determine current state of a position.
        
        Returns:
            PositionState enum value
        """
        # [FIX] Unified Bucket ID: One bucket per symbol (Buys + Sells combined)
        bucket_id = position.symbol
        stats = self.bucket_stats.get(bucket_id)
        
        if not stats:
            return PositionState.UNKNOWN
        
        return stats.state

    def _set_position_state(self, bucket_id: str, new_state: PositionState) -> None:
        """
        Atomically update position state with validation.
        
        Args:
            bucket_id: Bucket identifier
            new_state: Target state to transition to
        """
        with self._state_transition_lock:
            stats = self.bucket_stats.get(bucket_id)
            if not stats:
                logger.warning(f"Cannot set state for unknown bucket {bucket_id}")
                return
            
            old_state = stats.state
            
            # Validate state transition
            valid_transitions = {
                PositionState.OPENING: [PositionState.SINGLE_ACTIVE, PositionState.CLOSED],
                PositionState.SINGLE_ACTIVE: [PositionState.TRANSITIONING, PositionState.PENDING_CLOSE, PositionState.CLOSED],
                PositionState.TRANSITIONING: [PositionState.BUCKET_ACTIVE, PositionState.SINGLE_ACTIVE],  # Rollback on failure
                PositionState.BUCKET_ACTIVE: [PositionState.PENDING_CLOSE, PositionState.CLOSED],
                PositionState.PENDING_CLOSE: [PositionState.CLOSED, PositionState.BUCKET_ACTIVE, PositionState.SINGLE_ACTIVE],
            }
            
            allowed = valid_transitions.get(old_state, [])
            if new_state not in allowed and new_state != PositionState.UNKNOWN:
                logger.error(f"Invalid state transition: {old_state.name} -> {new_state.name} for {bucket_id}")
                return
            
            stats.state = new_state
            stats.last_state_check = time.time()
            logger.info(f"[STATE] {bucket_id}: {old_state.name} -> {new_state.name}")

    async def transition_to_bucket(
        self, 
        symbol: str, 
        position_type: str, 
        broker_adapter,
        hedge_ticket: int
    ) -> bool:
        """
        ATOMIC transition from SINGLE_ACTIVE to BUCKET_ACTIVE.
        
        This method performs the critical transition when adding a hedge:
        1. Mark state as TRANSITIONING
        2. Remove broker TP/SL (must happen BEFORE hedge)
        3. Execute hedge order
        4. Update state to BUCKET_ACTIVE
        5. Rollback on failure
        
        Args:
            symbol: Trading symbol
            position_type: BUY or SELL
            broker_adapter: Broker interface for TP/SL removal
            hedge_ticket: Ticket of original position to hedge
            
        Returns:
            bool: True if transition successful, False otherwise
        """
        # [FIX] Unified Bucket ID: One bucket per symbol
        bucket_id = symbol
        
        with self._state_transition_lock:
            stats = self.bucket_stats.get(bucket_id)
            if not stats:
                logger.error(f"Cannot transition unknown bucket {bucket_id}")
                return False
            
            if stats.state != PositionState.SINGLE_ACTIVE:
                logger.error(f"Cannot transition from {stats.state.name} to BUCKET_ACTIVE")
                return False
            
            # Step 1: Mark transitioning
            old_state = stats.state
            stats.state = PositionState.TRANSITIONING
            stats.mode = BucketMode.SINGLE  # Still single until hedge confirms
            logger.info(f"[TRANSITION] {bucket_id}: Starting SINGLE -> BUCKET transition")
        
        try:
            # Step 2: Remove broker TP/SL atomically
            logger.info(f"[TRANSITION] Removing broker TP/SL for ticket #{hedge_ticket}")
            remove_success = await broker_adapter.remove_tp_sl(hedge_ticket)
            
            if not remove_success:
                raise Exception(f"Failed to remove TP/SL for ticket #{hedge_ticket}")
            
            # Step 3: Hedge will be executed by calling code after this returns True
            # Step 4: Update to BUCKET_ACTIVE
            with self._state_transition_lock:
                stats.state = PositionState.BUCKET_ACTIVE
                stats.mode = BucketMode.BUCKET
                stats.last_state_check = time.time()
                logger.info(f"[TRANSITION] {bucket_id}: Successfully transitioned to BUCKET_ACTIVE")
            
            # [ADD THIS] --- VISUAL HEDGE ACTIVATION SUMMARY ---
            clean_msg = (
                f"\n--- HEDGE ACTIVATED ---\n"
                f"Parent:   #{hedge_ticket}\n"
                f"Type:     {position_type} (Survival Mode)\n"
                f"Goal:     Break-even recovery\n"
                f"----------------------------------------------------"
            )
            
            import sys
            if sys.platform == 'win32':
                ui_logger.info(clean_msg)
            else:
                try:
                    msg = (
                        f"\n[HEDGE ACTIVATED]\n"
                        f"Parent:   #{hedge_ticket}\n"
                        f"Type:     {position_type} (Survival Mode)\n"
                        f"Goal:     Break-even recovery\n"
                        f"----------------------------------------------------"
                    )
                    ui_logger.info(msg)
                except Exception:
                    ui_logger.info(clean_msg)

            return True
            
        except Exception as e:
            # Step 5: Rollback on failure
            logger.error(f"[TRANSITION] Failed to transition {bucket_id}: {e}")
            with self._state_transition_lock:
                stats.state = old_state  # Rollback to SINGLE_ACTIVE
                stats.mode = BucketMode.SINGLE
                logger.warning(f"[TRANSITION] Rolled back {bucket_id} to {old_state.name}")
            return False

    def cleanup_stale_positions(self, broker_positions: List) -> None:
        """
        Remove positions from tracking that no longer exist in the broker.
        CRITICAL for preventing "Position doesn't exist" errors during hedging.
        
        Args:
            broker_positions: List of position objects from broker
        """
        # SAFETY: If broker_positions is None (API error), DO NOT cleanup.
        # Treating None as empty list would wipe all positions from memory!
        if broker_positions is None:
            return

        # Get set of valid ticket numbers from broker
        broker_tickets = {p.ticket if hasattr(p, 'ticket') else p['ticket'] for p in broker_positions} if broker_positions else set()
        
        with self._lock:
            # Find ALL stale tickets across all tracking structures
            tracked_tickets = set(self.active_positions.keys())
            
            # Also find stale tickets in bucket_stats (restored from position_state.json)
            bucket_tickets = set()
            for stats in self.bucket_stats.values():
                bucket_tickets.update(stats.positions)
            
            # Combine all tracked tickets
            all_tracked = tracked_tickets | bucket_tickets
            stale_tickets = all_tracked - broker_tickets
            
            if stale_tickets:
                logger.warning(f"[CLEANUP] Found {len(stale_tickets)} stale positions not in broker: {list(stale_tickets)}")
                
                # Remove stale positions
                for ticket in stale_tickets:
                    if ticket in self.active_positions:
                        del self.active_positions[ticket]
                        logger.info(f"[CLEANUP] Removed stale position #{ticket} from active tracking")
                    
                    # Clean from learning trades
                    if ticket in self.active_learning_trades:
                        del self.active_learning_trades[ticket]
                        logger.info(f"[CLEANUP] Removed stale learning data for ticket #{ticket}")
                
                # Clean from bucket stats
                for bucket_id, stats in list(self.bucket_stats.items()):
                    original_count = len(stats.positions)
                    stats.positions = [t for t in stats.positions if t not in stale_tickets]
                    
                    if len(stats.positions) < original_count:
                        logger.info(f"[CLEANUP] Removed {original_count - len(stats.positions)} stale positions from bucket {bucket_id}")
                        
                        # Mark bucket as closed if empty
                        if len(stats.positions) == 0:
                            stats.closed = True
                            self.closed_buckets.add(bucket_id)
                            logger.info(f"[CLEANUP] Bucket {bucket_id} marked as closed (no remaining positions)")
                
                # Persist cleaned state
                self._save_state()
                logger.info(f"[CLEANUP] State cleaned and persisted")

    def update_positions(self, broker_positions: List[Dict]) -> None:
        """
        Update internal position tracking from broker data.
        
        IMPORTANT: MT5 in netting mode consolidates multiple trades for the same symbol
        into ONE position with averaged entry price. Bot tracks individual trades but
        MT5 shows consolidated volume and averaged price.

        Args:
            broker_positions: List of position dicts from broker
        """
        if broker_positions is None:
            return

        # Build new position dictionary atomically to avoid race conditions
        new_positions = {}
        for pos_data in broker_positions:
                try:
                    # Handle both dictionary and object (dataclass/namedtuple) input
                    if isinstance(pos_data, dict):
                        ticket = pos_data['ticket']
                    else:
                        ticket = pos_data.ticket
                    
                    # Skip known ghost tickets
                    if ticket in self._ghost_tickets:
                        continue

                    if isinstance(pos_data, dict):
                        # ticket already extracted
                        symbol = pos_data['symbol']
                        p_type = pos_data['type']
                        volume = pos_data['volume']
                        price_open = pos_data['price_open']
                        price_current = pos_data.get('price_current', pos_data['price_open'])
                        profit = pos_data['profit']
                        sl = pos_data['sl']
                        tp = pos_data['tp']
                        time_val = pos_data['time']
                        swap = pos_data.get('swap', 0.0)
                        commission = pos_data.get('commission', 0.0)
                        magic = pos_data.get('magic', 0)
                        comment = pos_data.get('comment', '')
                    else:
                        ticket = pos_data.ticket
                        symbol = pos_data.symbol
                        p_type = pos_data.type
                        volume = pos_data.volume
                        price_open = pos_data.price_open
                        # MT5Adapter Position might not have price_current
                        price_current = getattr(pos_data, 'price_current', pos_data.price_open)
                        profit = pos_data.profit
                        sl = pos_data.sl
                        tp = pos_data.tp
                        time_val = pos_data.time
                        swap = getattr(pos_data, 'swap', 0.0)
                        commission = getattr(pos_data, 'commission', 0.0)
                        magic = getattr(pos_data, 'magic', 0)
                        comment = getattr(pos_data, 'comment', '')

                    # Check if we have stored entry targets for this position
                    entry_tp_pips = 0.0
                    entry_sl_pips = 0.0
                    entry_atr = 0.0
                    if ticket in self.active_learning_trades:
                        metadata = self.active_learning_trades[ticket]
                        entry_tp_pips = metadata.get('entry_tp_pips', 0.0)
                        entry_sl_pips = metadata.get('entry_sl_pips', 0.0)
                        entry_atr = metadata.get('entry_atr', 0.0)
                    
                    # Determine position state - check if it's in a bucket
                    # [FIX] Unified Bucket ID: One bucket per symbol
                    bucket_id = symbol
                    if bucket_id in self.bucket_stats and not self.bucket_stats[bucket_id].closed:
                        # Part of existing bucket
                        pos_state = self.bucket_stats[bucket_id].state
                    else:
                        # New single position - assume broker TP/SL active
                        pos_state = PositionState.SINGLE_ACTIVE
                    
                    position = Position(
                        ticket=ticket,
                        symbol=symbol,
                        type=p_type,
                        volume=volume,
                        price_open=price_open,
                        price_current=price_current,
                        profit=profit,
                        sl=sl,
                        tp=tp,
                        time=time_val,
                        swap=swap,
                        commission=commission,
                        magic=magic,
                        comment=comment,
                        state=pos_state,
                        entry_tp_pips=entry_tp_pips,
                        entry_sl_pips=entry_sl_pips,
                        entry_atr=entry_atr
                    )
                    new_positions[position.ticket] = position

                except (KeyError, AttributeError) as e:
                    logger.warning(f"Invalid position data: {e}")
                    continue
        
        # Atomic replacement under lock to prevent race conditions
        with self._lock:
            self.active_positions = new_positions
            
            # [SELF-HEALING] Scan for orphans and adopt them into buckets
            # This ensures that if a manual trade or hedge appears, it is IMMEDIATELY managed.
            for ticket, pos in self.active_positions.items():
                bucket_id = pos.symbol
                if bucket_id in self.bucket_stats and not self.bucket_stats[bucket_id].closed:
                    stats = self.bucket_stats[bucket_id]
                    if ticket not in stats.positions:
                        stats.positions.append(ticket)
                        stats.last_update = time.time()
                        # Update the position state to match the bucket
                        pos.state = stats.state 
                        logger.info(f"[ADOPTION] Bucket {bucket_id} adopted orphan position #{ticket}")
                        self._save_state()

    def mark_position_as_ghost(self, ticket: int) -> None:
        """
        Mark a position as 'ghost' (broker reports it but it doesn't exist).
        This prevents the position from being re-added by update_positions.
        """
        with self._lock:
            self._ghost_tickets.add(ticket)
            if ticket in self.active_positions:
                del self.active_positions[ticket]

    def _get_bucket_stats(self, bucket_id: str) -> BucketStats:
        """
        Get bucket statistics, creating a new entry if it doesn't exist.

        Args:
            bucket_id: Bucket identifier

        Returns:
            BucketStats object
        """
        if bucket_id not in self.bucket_stats:
            self.bucket_stats[bucket_id] = BucketStats(
                bucket_id=bucket_id,
                positions=[],
                net_profit=0.0,
                open_time=time.time(),
                last_update=time.time(),
                closed=False,
                state=PositionState.UNKNOWN,
                mode=BucketMode.SINGLE,
                last_state_check=time.time(),
                exit_reason=""
            )
        return self.bucket_stats[bucket_id]

    def _update_bucket_stats(self) -> None:
        """Update statistics for all active buckets."""
        with self._lock:
            for bucket_id, stats in self.bucket_stats.items():
                if stats.closed:
                    continue

                # Calculate current net profit
                total_profit = 0.0
                active_positions = 0

                for ticket in stats.positions:
                    if ticket in self.active_positions:
                        total_profit += self.active_positions[ticket].profit
                        active_positions += 1

                stats.net_profit = total_profit
                stats.last_update = time.time()

                # Mark bucket as closed if no positions remain
                if active_positions == 0:
                    stats.closed = True
                    self.closed_buckets.add(bucket_id)

    def get_positions_for_symbol(self, symbol: str) -> List[Position]:
        """Get all active positions for a specific symbol."""
        with self._lock:
            return [pos for pos in self.active_positions.values() if pos.symbol == symbol]

    def find_bucket_by_tickets(self, tickets: List[int]) -> Optional[str]:
        """
        Find existing bucket containing any of the given tickets.
        
        Args:
            tickets: List of position ticket numbers
            
        Returns:
            Bucket ID if found, None otherwise
        """
        with self._lock:
            ticket_set = set(tickets)
            for bucket_id, stats in self.bucket_stats.items():
                if not stats.closed and ticket_set.intersection(set(stats.positions)):
                    return bucket_id
            return None

    def create_bucket(self, positions: List[Position]) -> str:
        """
        Create a new position bucket for coordinated management.

        Args:
            positions: List of positions to group

        Returns:
            Bucket ID string
        """
        if not positions:
            return ""

        # [FIX] Unified Bucket ID: One bucket per symbol
        bucket_id = positions[0].symbol

        with self._lock:
            # Determine initial state and mode based on position count
            initial_state = PositionState.SINGLE_ACTIVE if len(positions) == 1 else PositionState.BUCKET_ACTIVE
            initial_mode = BucketMode.SINGLE if len(positions) == 1 else BucketMode.BUCKET
            
            self.bucket_stats[bucket_id] = BucketStats(
                bucket_id=bucket_id,
                positions=[p.ticket for p in positions],
                net_profit=sum(p.profit for p in positions),
                open_time=time.time(),
                last_update=time.time(),
                closed=False,
                state=initial_state,
                mode=initial_mode,
                last_state_check=time.time()
            )

            self._save_state()

        logger.info(f"Created bucket {bucket_id} with {len(positions)} positions | State: {initial_state.name} | Mode: {initial_mode.name}")
        return bucket_id

    def calculate_breakeven(self, positions: List[Any], atr_value: Optional[float] = None) -> float:
        """
        Calculates the monetary target ($) to close the bucket.
        Dynamic: Scales with Volatility (ATR).
        Formula: Target = Volume * ContractSize * (ATR * 0.01)
        """
        if not positions:
            return 0.0
            
        total_lots = sum(pos.volume for pos in positions)
        symbol = positions[0].symbol
        
        target = 0.0
        
        # Try to calculate dynamic target based on ATR
        if atr_value and atr_value > 0:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                # 1% of Daily ATR as pure profit target
                # XAUUSD Example: ATR=20.0, Contract=100. Target = 1.0 * 100 * (20.0 * 0.01) = $20.00
                # EURUSD Example: ATR=0.0060, Contract=100k. Target = 1.0 * 100k * (0.00006) = $6.00
                target = total_lots * symbol_info.trade_contract_size * (atr_value * 0.01)
        
        # Fallback if ATR/Symbol info fails
        if target == 0.0:
            target = total_lots * 15.0  # Safe default ($15 per lot)

        # Absolute floor to cover basic slippage
        return max(target, 1.0)

    def _calibrated_profit_buffer(self, symbol: str, total_volume: float, base_buffer: float) -> float:
        """
        Calculate calibrated profit buffer based on volume and symbol volatility.
        """
        # XAUUSD/Gold typically has higher spread/slippage
        is_gold = "XAU" in symbol or "GOLD" in symbol
        
        # Estimated slippage cost per lot (conservative)
        slippage_per_lot = 15.0 if is_gold else 5.0
        
        # Calculate volume-based buffer component
        volume_component = total_volume * slippage_per_lot
        
        # The final buffer should be at least the base_buffer (which handles bucket state)
        # but also needs to cover the volume-based slippage risk.
        return max(base_buffer, volume_component)

    def calculate_net_pnl(self, positions: List[Any]) -> Tuple[float, float, float, float]:
        """
        Calculate True Net PnL (Profit + Swap + Commission).
        
        Args:
            positions: List of position objects
            
        Returns:
            Tuple[net_pnl, gross_pnl, swap, commission]
        """
        gross_pnl = sum(pos.profit for pos in positions)
        swap = sum(getattr(pos, 'swap', 0.0) for pos in positions)
        commission = sum(getattr(pos, 'commission', 0.0) for pos in positions)
        
        # Some brokers include comm/swap in profit, but MT5 standard is separate
        net_pnl = gross_pnl + swap + commission
        
        return net_pnl, gross_pnl, swap, commission

    def calculate_bucket_target_usd(self, current_spread_pips: float, total_volume: float, volatility_ratio: float = 1.0, symbol: str = "", pip_value: float = 10.0) -> float:
        """
        INTELLIGENCE UPGRADE: The 'Velvet Cushion' Protocol.
        Calculates a target that GUARANTEES profit after execution costs.
        
        Args:
            current_spread_pips: Spread in pips
            total_volume: Total lot size
            volatility_ratio: Market volatility ratio
            symbol: Trading symbol
            pip_value: Dollar value of 1 pip per 1 lot (e.g. $1 for Gold, $10 for Forex)
            
        Returns:
            Target profit in USD
        """
        # 1. Base Profit: The minimum net money we want to bank
        # INTELLIGENCE FIX: Scale with volume and pip value.
        # We aim for approx 3.0 pips of pure profit, with a $0.50 floor.
        base_profit = max(0.50, total_volume * pip_value * 3.0) 
        
        # 2. Spread Cost: The cost to cross the bid/ask spread immediately
        # We multiply by 1.5 to account for spread widening during the close
        spread_cost = (current_spread_pips * total_volume * pip_value) * 1.5
        
        # 3. Volatility Penalty: If market is violent, demand higher profit
        # because slippage will be worse.
        vol_penalty = 0.0
        # Use 2.0 as threshold for "High Volatility" based on ratio
        if volatility_ratio > 2.0: 
            vol_penalty = 5.00 # Add $5 extra buffer
            
        # 4. Execution Delay Buffer (The "Slippage Cushion")
        # We add $3.00 per lot to account for the 200ms-500ms closing delay
        # [INTELLIGENCE UPGRADE] Dynamic Slippage Prediction
        # If volatility is high, slippage is exponential, not linear.
        if volatility_ratio > 1.5:
             # High Volatility: $10.00 per lot buffer
             execution_buffer = total_volume * 10.0
        else:
             # Normal: $3.00 per lot buffer
             execution_buffer = total_volume * 3.0
        
        # Final Calculation
        total_target = base_profit + spread_cost + vol_penalty + execution_buffer
        
        return total_target

    def calculate_dynamic_target(self, base_target_pips: float, spread_pips: float, volatility_factor: float = 1.0) -> float:
        """
        Sniper Exit: Calculates dynamic target based on market conditions.
        Target = Base + (Spread * Vol * Safety)
        """
        safety_margin = 2.0 # Ensure we cover spread 2x
        dynamic_buffer = spread_pips * volatility_factor * safety_margin
        return base_target_pips + dynamic_buffer

    def should_close_bucket(self, bucket_id: str, ppo_guardian, market_data=None) -> Tuple[bool, float]:
        """
        Futuristic AI-driven bucket exit logic for high-intelligence scalping.

        Enhanced with:
        - Sniper Exit (Dynamic Spread Buffer)
        - Survival Protocol (Target Decay)
        - Time-based scalping exits
        - ATR-based profit/loss targets
        - Multi-agent AI consensus
        """
        with self._lock:
            if bucket_id not in self.bucket_stats:
                logger.info(f"[TP_CHECK_SKIP] Bucket {bucket_id} not in bucket_stats")
                return False, 0.0

            stats = self.bucket_stats[bucket_id]
            
            # State validation - skip if in transitioning state
            if stats.state == PositionState.TRANSITIONING:
                logger.debug(f"[TP_CHECK_SKIP] Bucket {bucket_id} in TRANSITIONING state")
                return False, 0.0
            
            if stats.state == PositionState.PENDING_CLOSE or stats.state == PositionState.CLOSED:
                logger.debug(f"[TP_CHECK_SKIP] Bucket {bucket_id} in {stats.state.name} state")
                return False, 0.0
            
            if stats.closed:
                logger.info(f"[TP_CHECK_SKIP] Bucket {bucket_id} already closed")
                return False, 0.0

            positions = [self.active_positions[t] for t in stats.positions if t in self.active_positions]
            
        if not positions:
            logger.info(f"[TP_CHECK_SKIP] Bucket {bucket_id} has no active active positions")
            return False, 0.0
        
        # Log bucket status for hedged positions
        if len(positions) > 1:
            total_profit, _, _, _ = self.calculate_net_pnl(positions)
            logger.debug(f"[BUCKET] {bucket_id}: {len(positions)} positions, Net P&L: ${total_profit:.2f}")

        try:
            first_pos = positions[0]
            current_time = time.time()
            # FIX: Use .time instead of .time_open (Position dataclass uses 'time')
            position_age_seconds = current_time - first_pos.time
            position_age_minutes = position_age_seconds / 60

            # Live Net PnL is always the most reliable real-time signal.
            net_pnl, gross_pnl, swap, comm = self.calculate_net_pnl(positions)

            # Get market data for intelligent analysis (avoid synthetic defaults where possible)
            strict_entry = bool(market_data.get('strict_entry', False)) if market_data else False
            atr_ok = bool(market_data.get('atr_ok', True)) if market_data else True
            rsi_ok = bool(market_data.get('rsi_ok', True)) if market_data else True

            atr_value = None
            if market_data and atr_ok:
                atr_value = market_data.get('atr', None)

            spread_points = None
            if market_data and ('bid' in market_data) and ('ask' in market_data):
                try:
                    spread_points = abs(float(market_data['ask']) - float(market_data['bid']))
                except Exception:
                    spread_points = None
            if spread_points is None and market_data:
                spread_points = market_data.get('spread', None)

            trend_strength = market_data.get('trend_strength', 0.0) if market_data else 0.0
            point = market_data.get('point', 0.00001) if market_data else 0.00001
            volatility_ratio = market_data.get('volatility_ratio', 1.0) if market_data else 1.0

            # Determine precision for logging
            precision = int(-math.log10(point)) if point > 0 else 2

            # Calculate pip multiplier: XAUUSD uses 100 (1 pip = 0.01), forex uses 10000 (1 pip = 0.0001)
            if "XAU" in first_pos.symbol or "GOLD" in first_pos.symbol:
                pip_multiplier = 100
            elif "JPY" in first_pos.symbol:
                pip_multiplier = 100
            else:
                pip_multiplier = 10000
            
            pip_size = 1.0 / pip_multiplier
            spread_pips = float(spread_points) * pip_multiplier if spread_points is not None else 0.0

            # Calculate precise metrics
            current_price = first_pos.price_current
            entry_price = first_pos.price_open
            
            # [PHASE 2] GHOST PROTOCOL: Predictive Exit
            # Logic: If Profit > 80% of Target AND Momentum Drops -> CLOSE
            # CRITICAL FIX: Use NET P&L of entire bucket, not just first position
            ghost_exit = False
            ghost_reason = ""
            
            # Calculate current profit in pips FOR THE ENTIRE BUCKET (not just first position)
            # For hedged buckets, we must use NET P&L, not individual position pips
            contract_size = 100 if "XAU" in first_pos.symbol or "GOLD" in first_pos.symbol else 100000
            total_volume = sum(pos.volume for pos in positions)
            
            # Convert NET P&L to pips equivalent
            if total_volume > 0:
                current_pips = (net_pnl / (total_volume * contract_size)) * pip_multiplier
            else:
                current_pips = 0.0
                
            # Get Target Pips (prefer stored entry TP; only fall back to ATR if ATR is valid)
            target_pips = 0.0
            if first_pos.entry_tp_pips > 0:
                target_pips = float(first_pos.entry_tp_pips)
            elif atr_value is not None and float(atr_value) > 0:
                target_pips = float(atr_value) * pip_multiplier * 0.3
            
            # CRITICAL: Only trigger Ghost Protocol if:
            # 1. We have a valid target
            # 2. Current pips are POSITIVE (in profit)
            # 3. Current pips are > 80% of target
            # 4. Momentum is fading
            if target_pips > 0 and current_pips > 0 and current_pips > (target_pips * 0.8):
                # Check Momentum (OBI or RSI)
                obi_ok = bool(market_data.get('obi_ok', False)) if market_data else False
                obi = float(market_data.get('obi', 0.0) or 0.0) if (market_data and obi_ok) else None
                rsi = None
                if market_data and rsi_ok:
                    rsi = market_data.get('rsi', None)
                obi_str = f"{obi:.2f}" if obi is not None else "NA"
                
                # If BUY and Momentum Fades
                if first_pos.is_buy:
                    if (obi is not None and obi < -0.2) or (rsi is not None and rsi > 70): # Sell Pressure or Overbought
                        ghost_exit = True
                        ghost_reason = f"Ghost Protocol (OBI={obi_str}, RSI={(float(rsi)):.1f}" if rsi is not None else f"Ghost Protocol (OBI={obi_str}, RSI=NA)"
                # If SELL and Momentum Fades
                elif first_pos.is_sell:
                    if (obi is not None and obi > 0.2) or (rsi is not None and rsi < 30): # Buy Pressure or Oversold
                        ghost_exit = True
                        ghost_reason = f"Ghost Protocol (OBI={obi_str}, RSI={(float(rsi)):.1f}" if rsi is not None else f"Ghost Protocol (OBI={obi_str}, RSI=NA)"
            
            if ghost_exit:
                logger.info(f"[GHOST PROTOCOL] Triggered! {ghost_reason} | Net Profit: {current_pips:.1f}/{target_pips:.1f} pips | Bucket P&L: ${net_pnl:.2f}")
                stats.exit_reason = "GHOST_PROTOCOL"
                return True, 1.0

            # Use STORED TP/SL from entry time (fixed targets, not recalculated)
            # If not set, fall back to ATR only if ATR is valid.
            atr_pips = float(atr_value) * pip_multiplier if (atr_value is not None and float(atr_value) > 0) else 0.0
            
            scalping_tp_pips = 0.0
            scalping_sl_pips = 0.0
            using_stored_tp = False

            if first_pos.entry_tp_pips > 0:
                scalping_tp_pips = first_pos.entry_tp_pips  # Fixed TP target from entry
                scalping_sl_pips = first_pos.entry_sl_pips  # Fixed SL from entry (usually 0)
                using_stored_tp = True
            else:
                # Fallback: Recalculate (for backward compatibility with old positions)
                if atr_pips > 0:
                    scalping_tp_pips = atr_pips * 0.3  # 0.3 ATR target for scalping
                    scalping_sl_pips = atr_pips * 1.0  # 1.0 ATR stop for scalping
                else:
                    # If ATR is unavailable, do not invent a tiny target.
                    # We'll rely on Net PnL + calibrated buffer instead.
                    scalping_tp_pips = 0.0
                    scalping_sl_pips = 0.0

            # --- LAYER 2: SNIPER EXIT (Dynamic Target) ---
            # Adjust TP based on current spread and volatility to prevent negative closing
            # If spread is wide, we need more profit to cover it.
            final_tp_pips = self.calculate_dynamic_target(scalping_tp_pips, spread_pips, volatility_ratio)
            
            # --- LAYER 3: SURVIVAL PROTOCOL (Target Decay) ---
            # If we are deep in hedges (Hedge 3+), drop target to break-even ($0.50)
            # Initial (1) + H1 (2) + H2 (3) + H3 (4)
            is_survival_mode = len(positions) >= 10  # Increased from 4 to 10
            
            # Position P&L calculation
            volume = first_pos.volume
            pip_value = 0.0001  # For XAUUSD (adjust for other pairs)
            
            # [FIX] Define price_movement before usage
            if first_pos.is_buy:
                price_movement = current_price - entry_price
            else:
                price_movement = entry_price - current_price
                
            current_pnl = price_movement * volume * 10000 * pip_value

            # === FUTURISTIC AI EXIT CRITERIA ===

            # 1. TIME-BASED SCALPING EXIT (REMOVED)
            # User requested to remove the 3-minute hard limit to allow trades to run.
            # The "Sniper Exit" (Dynamic Target) and "Chameleon" (Trend Detection) now handle exits.
            time_exit = False 

            # 2. PROFIT TARGET ACHIEVED
            profit_exit = False
            
            # Check if this is a BUCKET (multiple positions) or SINGLE position
            if len(positions) == 1:
                # SINGLE POSITION: Use stored TP target from entry (broker TP/SL active)
                
                exit_price = current_price
                if market_data:
                    # If we have live market data, use explicit Bid/Ask for exit check
                    exit_price = market_data.get('current_price', current_price) # Fallback
                    if 'bid' in market_data and 'ask' in market_data:
                        exit_price = market_data['bid'] if first_pos.is_buy else market_data['ask']

                if first_pos.is_buy:
                    profit_pips = (exit_price - entry_price) * pip_multiplier
                    profit_exit = profit_pips >= final_tp_pips # Use DYNAMIC TARGET
                    
                    # DEBUG LOGGING FOR TP DIAGNOSIS
                    if profit_pips > (final_tp_pips * 0.8):
                        logger.debug(f"[TP CHECK] {first_pos.symbol} BUY | Price: {exit_price:.2f} | Entry: {entry_price:.2f} | Profit: {profit_pips:.1f} pips | Target: {final_tp_pips:.1f} pips (Base: {scalping_tp_pips:.1f}) | Exit: {profit_exit}")
                        
                else:  # SELL
                    profit_pips = (entry_price - exit_price) * pip_multiplier
                    profit_exit = profit_pips >= final_tp_pips # Use DYNAMIC TARGET
                    
                    # DEBUG LOGGING FOR TP DIAGNOSIS
                    if profit_pips > (final_tp_pips * 0.8):
                        logger.debug(f"[TP CHECK] {first_pos.symbol} SELL | Price: {exit_price:.2f} | Entry: {entry_price:.2f} | Profit: {profit_pips:.1f} pips | Target: {final_tp_pips:.1f} pips (Base: {scalping_tp_pips:.1f}) | Exit: {profit_exit}")

                # === USD-BASED FALLBACK CHECK ===
                # If pip-based check fails (due to bad tick data), check actual broker profit
                if not profit_exit and final_tp_pips > 0:
                    try:
                        # Calculate Target USD
                        # XAUUSD: 1 pip (0.01) on 1 lot = $1.00
                        # Forex: 1 pip (0.0001) on 1 lot = $10.00 (approx)
                        usd_per_pip_per_lot = 1.0 if "XAU" in first_pos.symbol or "GOLD" in first_pos.symbol else 10.0
                        target_profit_usd = final_tp_pips * usd_per_pip_per_lot * first_pos.volume
                        
                        # Check if actual profit exceeds target
                        # [FIX] Use Net PnL (Profit + Swap + Comm) to avoid closing in loss
                        net_pnl, _, _, _ = self.calculate_net_pnl(positions)
                        
                        if net_pnl >= target_profit_usd:
                            logger.info(f"[TP FALLBACK] BROKER PROFIT TARGET REACHED: Net ${net_pnl:.2f} >= ${target_profit_usd:.2f} (Target: {final_tp_pips:.1f} pips)")
                            profit_exit = True
                    except Exception as e:
                        logger.error(f"Error in USD fallback check: {e}")
                # =====================================

                if profit_exit:
                    # [CRITICAL FIX] Double Check Net PnL before closing
                    if net_pnl < 0:
                        logger.warning(f"[TP ABORT] Profit Exit triggered but Net PnL is negative (${net_pnl:.2f}). Holding.")
                        profit_exit = False
                    else:
                        tp_price = entry_price + (final_tp_pips / pip_multiplier) if first_pos.is_buy else entry_price - (final_tp_pips / pip_multiplier)
                        logger.info(f"[TP HIT] PROFIT TARGET REACHED: {profit_pips:.1f} pips >= {final_tp_pips:.1f} pips target | TP Price: {tp_price:.{precision}f}")
            else:
                # BUCKET: Use DYNAMIC BREAK-EVEN LOGIC
                total_profit_usd, _, _, _ = self.calculate_net_pnl(positions)
                total_volume = sum(pos.volume for pos in positions)
                num_trades = len(positions)
                
                # INTELLIGENCE UPGRADE: Use Velvet Cushion Protocol
                # Calculate dynamic target based on Spread, Volume, and Volatility
                
                # Determine Pip Value ($ per pip per lot)
                # XAUUSD: 1 pip (0.01) = $1.00
                # Forex: 1 pip (0.0001) = $10.00
                pip_value = 1.0 if "XAU" in first_pos.symbol or "GOLD" in first_pos.symbol else 10.0
                
                # Pass raw spread_pips (e.g. 20 for Gold) and correct pip_value
                base_target_usd = self.calculate_bucket_target_usd(
                    spread_pips, 
                    total_volume, 
                    volatility_ratio, 
                    first_pos.symbol,
                    pip_value=pip_value
                )
                
                # --- SURVIVAL MODE LOGIC (User Requested) ---
                # 1 Trade: Normal Target (Handled in 'if' block above)
                # 2 Trades: 50% Target
                # 3+ Trades: Survival Mode (Just cover costs + $1)
                
                if num_trades == 2:
                    target_profit_usd = base_target_usd * 0.5
                    logger.info(f"[SURVIVAL] 2 Trades Active. Target reduced to 50%: ${target_profit_usd:.2f}")
                elif num_trades >= 3:
                    # Survival Mode: Just cover costs + small profit ($5.00)
                    target_profit_usd = 5.0
                    logger.debug(f"[SURVIVAL] 3+ Trades Active! Target SLASHED to $5.00 (Survival Mode)")
                else:
                    target_profit_usd = base_target_usd

                # For hedged positions, we want AT LEAST break-even + dynamic profit
                # net_pnl already computed above
                
                # [DEBUG] Detailed Exit Calculation Log
                if len(positions) >= 2:
                    logger.debug(f"[EXIT CALC] {bucket_id} | Net PnL: ${net_pnl:.2f} | Target: ${target_profit_usd:.2f} | Trades: {num_trades} | VolRatio: {volatility_ratio:.2f}")

                # [ENHANCEMENT] Stalemate Breaker Logic
                # If stuck for too long, reduce target
                current_time = int(time.time())
                oldest_time = min(p.time for p in positions)
                duration_minutes = (current_time - oldest_time) / 60
                
                effective_target = target_profit_usd
                status_msg = ""
                
                if len(positions) >= 10:  # Increased from 4 to 10
                    if duration_minutes > 45:
                        effective_target = 0.50 # Escape
                        status_msg = " [ESCAPE: Deep Hedge > 45m]"
                    elif duration_minutes > 20:
                        effective_target = target_profit_usd * 0.25 # Decay
                        status_msg = " [DECAY: Deep Hedge > 20m]"
                elif len(positions) >= 3:
                    if duration_minutes > 90:
                        effective_target = 0.50
                        status_msg = " [ESCAPE: Med Hedge > 90m]"
                
                # --- [INTELLIGENCE] PROFIT RATCHET LOGIC ---
                # 1. Update High Water Mark (Highest Profit Seen)
                current_high = self.high_water_marks.get(bucket_id, -99999.0)
                if net_pnl > current_high:
                    self.high_water_marks[bucket_id] = net_pnl
                    current_high = net_pnl
                
                # 2. Check Ratchet Conditions
                ratchet_exit = False
                ratchet_reason = ""
                
                # Rule A: The "Don't Be Greedy" Lock (80% -> 50%)
                # If we hit 80% of target but dropped back to 50%, CLOSE.
                if current_high >= (effective_target * 0.8) and net_pnl <= (effective_target * 0.5):
                    if net_pnl > 1.0: # Ensure we are still profitable
                        ratchet_exit = True
                        ratchet_reason = f"[RATCHET PROTECT] Peak ${current_high:.2f} -> Now ${net_pnl:.2f}"
                        status_msg += " [RATCHET]"

                # Rule B: The "Breakeven Assist" (50% -> 10%)
                # If we hit 50% of target but dropped to near zero, CLOSE at small profit.
                elif current_high >= (effective_target * 0.5) and net_pnl <= (effective_target * 0.1):
                    # [INTELLIGENCE UPGRADE] Slippage Buffer for Ratchet
                    # Don't close at $0.50 if volatility is high, slippage will kill it.
                    min_profit_buffer = 2.0 if volatility_ratio > 1.5 else 0.50
                    
                    if net_pnl > min_profit_buffer: # Ensure positive with buffer
                        ratchet_exit = True
                        ratchet_reason = f"[BE DEFENSE] Peak ${current_high:.2f} -> Now ${net_pnl:.2f}"
                        status_msg += " [DEFENSE]"

                profit_exit = (net_pnl >= effective_target) or ratchet_exit
                
                # Log status if positive but waiting
                if net_pnl > 0 and not profit_exit:
                    if net_pnl > (effective_target * 0.5):
                         logger.info(f"[{first_pos.symbol}] PnL: Net ${net_pnl:.2f} / Target ${effective_target:.2f}{status_msg} (Gross ${gross_pnl:.2f} - Costs ${abs(comm)+abs(swap):.2f})")
                
                if profit_exit:
                    if ratchet_exit:
                        logger.info(f"[RATCHET EXIT] {ratchet_reason}")
                    
                    # [OPTIMIZATION] Removed blocking log here. 
                    # The close trigger will happen immediately in the caller or close_bucket_positions.
                    # logger.info(f"[BUCKET TP HIT] VELVET CUSHION TARGET REACHED: Net ${net_pnl:.2f} >= ${effective_target:.2f}{status_msg} | {len(positions)} positions")
                    # Intentionally no action here; caller triggers the close.

            # 3. STOP LOSS DISABLED - Hedging strategy manages risk through zone recovery
            # No hard stop losses - hedges are placed when price moves against position
            stop_loss_exit = False

            # 4. AI CONFIDENCE EXIT (PPO Guardian with bucket-specific logic)
            ai_exit = False
            
            # [WICK INTELLIGENCE EXIT] Check if we should exit at wick extreme
            # OPPOSITE LOGIC FROM ENTRIES: Wicks are GOOD exit zones!
            wick_exit = False
            wick_exit_reason = ""
            
            try:
                from src.ai_core.wick_intelligence import get_wick_intelligence
                
                # Get recent candles for wick analysis
                recent_candles = []
                if hasattr(self.mt5_adapter, 'get_candles'):
                    try:
                        recent_candles = self.mt5_adapter.get_candles(first_pos.symbol, timeframe='M1', count=10)
                    except (AttributeError, ValueError, KeyError, TypeError) as e:
                        logger.debug(f"[WICK EXIT] Could not fetch candles: {e}")
                        pass
                
                if recent_candles and len(positions) > 0:
                    wick_intel = get_wick_intelligence()
                    
                    # Determine position type
                    position_type = "BUY" if first_pos.is_buy else "SELL"
                    
                    # Calculate current profit in pips
                    if first_pos.is_buy:
                        profit_pips = (current_price - entry_price) * pip_multiplier
                    else:
                        profit_pips = (entry_price - current_price) * pip_multiplier
                    
                    # Check if we should exit at wick
                    should_exit, wick_reason = wick_intel.should_exit_at_wick(
                        position_type=position_type,
                        current_price=current_price,
                        recent_candles=recent_candles,
                        profit_pips=profit_pips
                    )
                    
                    if should_exit:
                        wick_exit = True
                        wick_exit_reason = wick_reason
                        logger.info(f"[WICK EXIT] {wick_reason}")
                        
                        # If in profit, this is a strong exit signal
                        if profit_pips > 5:
                            logger.info(f"[WICK EXIT] ✅ Taking profit at wick rejection zone (+{profit_pips:.1f} pips)")
                    else:
                        # Log if we're monitoring a potential wick exit
                        if "HOLD" in wick_reason:
                            logger.debug(f"[WICK MONITOR] {wick_reason}")
            
            except Exception as wick_e:
                logger.debug(f"[WICK EXIT] Check failed: {wick_e}")
                # Continue without wick exit if check fails

            ai_confidence = 0.0
            ppo_reason = ""
            
            try:
                if ppo_guardian and len(positions) > 1:
                    # BUCKET MODE: Use sophisticated should_exit_bucket
                    total_pnl, _, _, _ = self.calculate_net_pnl(positions)
                    total_volume = sum(pos.volume for pos in positions)
                    
                    # If ATR is unavailable, do not run ATR-dependent AI exit logic.
                    if atr_value is None or float(atr_value) <= 0:
                        ppo_exit, ppo_conf, ppo_reason = False, 0.0, "ATR unavailable"
                    else:
                        ppo_exit, ppo_conf, ppo_reason = ppo_guardian.should_exit_bucket(
                        net_pnl_usd=total_pnl,
                        position_age_seconds=position_age_minutes * 60,
                        atr=atr_value,
                        num_positions=len(positions),
                        total_volume=total_volume,
                        account_equity=0.0  # Not used in current logic
                        )
                    ai_exit = ppo_exit
                    ai_confidence = ppo_conf
                    if ai_exit:
                        logger.debug(f"[PPO_EXIT] BUCKET EXIT: {ppo_reason} | Confidence: {ppo_conf:.2f}")
                elif ppo_guardian:
                    # SINGLE POSITION: NO AI EXIT - Uses broker TP/SL for instant execution
                    # AI exits only apply to buckets (multiple positions) for intelligent management
                    ai_confidence = 0.0  # Disable AI exit for single positions
                    ai_exit = False
                    logger.debug(f"[PPO_EXIT] Single position - using broker TP/SL, no AI exit")
            except Exception as e:
                logger.warning(f"[PPO] Error getting exit decision: {e}")
                ai_confidence = 0.0

            # 5. EMERGENCY EXIT (Increased to allow full zone recovery)
            # Was 4, increased to 10 to allow max_layers (5) + buffer
            emergency_exit = len(positions) >= 10

            # 6. SPREAD COST EXIT (If spread > ATR, avoid holding) - ONLY FOR SINGLE POSITIONS
            spread_cost_exit = False
            if len(positions) == 1:
                if spread_points is not None and atr_pips > 0:
                    spread_cost_pips = float(spread_points) / pip_size
                    spread_cost_exit = spread_cost_pips > (atr_pips * 0.3) and position_age_minutes > 2.0
                    if spread_cost_exit:
                        logger.info(f"[SCALP] SPREAD EXIT: Spread {spread_cost_pips:.1f}pips > ATR*0.3 and age >2min")

            # === INTELLIGENT CONSENSUS DECISION ===
            should_close = (
                time_exit or
                profit_exit or
                stop_loss_exit or
                ai_exit or
                wick_exit or  # [WICK INTELLIGENCE] Exit at rejection zones
                emergency_exit or
                spread_cost_exit
            )

            # [INTELLIGENT FIX] NO-LOSS + SLIPPAGE BUFFER PROTECTION
            # Unless it's Emergency, require a POSITIVE buffer to absorb spread/slippage.
            # This hardens exits against "flip-to-loss on close" issues.
            allow_emergency_close_loss = str(os.getenv("AETHER_ALLOW_EMERGENCY_CLOSE_LOSS", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

            if should_close and not (stop_loss_exit or (emergency_exit and allow_emergency_close_loss)):
                # Dynamic buffer: higher in volatile markets
                base_buffer = 2.0 if volatility_ratio > 1.5 else 0.50
                # Heavier buffer for deeper buckets (execution risk grows with size)
                if len(positions) >= 10:  # Increased from 4 to 10
                    base_buffer *= 2.0  # e.g., $1.0 -> $2.0, $2.0 -> $4.0

                # [CALIBRATION] Replace static buffer with rolling P95-based buffer when available
                total_volume_for_buffer = sum(pos.volume for pos in positions)
                min_profit_buffer = self._calibrated_profit_buffer(first_pos.symbol, total_volume_for_buffer, base_buffer)

                if net_pnl < min_profit_buffer:
                    if str(os.getenv("AETHER_SLIPPAGE_CALIBRATION_LOG", "1")).strip().lower() in ("1", "true", "yes", "on"):
                        sc = self._get_slippage_sample_count(first_pos.symbol)
                        p95pl = self._get_slippage_p95_per_lot_usd(first_pos.symbol)
                        logger.debug(
                            f"[CALIBRATED BUFFER] {bucket_id}: base=${base_buffer:.2f} calibrated=${min_profit_buffer:.2f} vol={total_volume_for_buffer:.2f} p95_per_lot=${p95pl:.4f} mult={self._slippage_p95_multiplier:.2f} samples={sc}"
                        )
                    logger.debug(
                        f"[VETO] Exit blocked for {bucket_id}: Net ${net_pnl:.2f} < Buffer ${min_profit_buffer:.2f}."
                    )
                    should_close = False
                    final_confidence = 0.0

            # If we lack ATR + stored TP (legacy positions), still allow a safe net-PnL-based close.
            # This prevents "skip forever" while avoiding tiny synthetic targets.
            if len(positions) == 1 and not should_close:
                if first_pos.entry_tp_pips <= 0 and (atr_value is None or float(atr_value) <= 0):
                    total_volume_for_buffer = sum(pos.volume for pos in positions)
                    min_profit_buffer = self._calibrated_profit_buffer(first_pos.symbol, total_volume_for_buffer, 0.50)
                    if net_pnl >= min_profit_buffer:
                        stats.exit_reason = "NET_PNL_BUFFER"
                        return True, 0.8

            # Calculate final confidence score (weighted average)
            exit_factors = [
                (time_exit, 0.8),      # Time most important for scalping
                (profit_exit, 0.9),    # Profit target very important
                (stop_loss_exit, 1.0), # Stop loss critical
                (ai_exit, ai_confidence), # AI confidence as-is
                (emergency_exit, 1.0), # Emergency always 1.0
                (spread_cost_exit, 0.6) # Spread less critical
            ]

            active_factors = [weight for condition, weight in exit_factors if condition]
            final_confidence = sum(active_factors) / len(active_factors) if active_factors else ai_confidence

            if should_close:
                exit_reasons = []
                if time_exit: exit_reasons.append("TIME")
                if profit_exit: exit_reasons.append("PROFIT")
                if stop_loss_exit: exit_reasons.append("STOP_LOSS")
                if ai_exit: exit_reasons.append("AI")
                if emergency_exit: exit_reasons.append("EMERGENCY")
                if spread_cost_exit: exit_reasons.append("SPREAD")

                # Store exit reason in bucket stats for TradingLogger summary
                with self._lock:
                    stats.exit_reason = ', '.join(exit_reasons)
                    stats.exit_confidence = final_confidence
                
                # Simple exit signal log - detailed summary will come when positions close
                logger.info(f"[EXIT SIGNAL] {bucket_id} | Reasons: {stats.exit_reason} | Confidence: {final_confidence:.3f}")

            return should_close, final_confidence

        except Exception as e:
            logger.error(f"Error evaluating bucket {bucket_id} exit: {e}")
            return False, 0.0

    async def update_bucket_tp(self, broker, symbol: str, bucket_id: str, current_price: float = None):
        """
        Updates the Take Profit for all positions in a bucket.
        FORCE VIRTUAL EXIT ONLY: Clears any broker-side TP to 0.0.
        """
        positions = self.get_positions_in_bucket(bucket_id)
        if not positions:
            return

        # Apply to all trades in the bucket
        for pos in positions:
            # Only update if TP is not already 0.0
            current_tp = getattr(pos, 'tp', 0.0)
            if current_tp != 0.0:
                try:
                    # Use broker adapter to modify position
                    order_type_str = "BUY" if pos.type == 0 else "SELL"
                    broker.execute_order(
                        symbol=pos.symbol, 
                        action="MODIFY", 
                        volume=pos.volume, 
                        order_type=order_type_str, 
                        sl=pos.sl, 
                        tp=0.0, # FORCE 0.0
                        ticket=pos.ticket
                    )
                    logger.debug(f"[SURVIVAL] Cleared TP for Ticket {pos.ticket} (Virtual Exit Only)")
                        
                except Exception as e:
                    logger.error(f"Failed to clear TP for {pos.ticket}: {e}")



    async def execute_ai_sniper_logic(self, bucket_id: str, oracle, broker, candles: List[Dict], symbol: str) -> bool:
        """
        Executes the 'AI Sniper' Proportional Unwind logic (v5.6.0).
        Instead of 'Bank & Kill' (which strips the hedge), we use 'Proportional Unwind'.
        We close a PARTIAL amount of the Winner to fund a PARTIAL close of the Loser.
        This maintains the hedge protection while reducing total exposure.
        """
        with self._lock:
            if bucket_id not in self.bucket_stats:
                return False
            
        positions = self.get_positions_in_bucket(bucket_id)
        if len(positions) < 2:
            return False 
            
        # 1. Oracle Check (Trend Reversal)
        oracle_result = await oracle.get_sniper_signal_v2(symbol, candles)
        signal_val = oracle_result.get('signal', 0)
        confidence = oracle_result.get('confidence', 0.0)
        reason = oracle_result.get('reason', '')
        
        if signal_val == 0:
            return False

        # 2. Identify Winners and Losers
        # Signal 1 (UP) -> Expect UP -> Sells are Winners (if we are at bottom? No.)
        # Wait, if Signal is UP (1), we expect price to rise.
        # If price rises: Buys gain, Sells lose.
        # So if we are at the BOTTOM (Signal UP), our Sells should be in profit (Winners) and Buys in loss.
        # We want to close the Sells (Winners) to buy back lower, and use profit to reduce Buys (Losers).
        
        sniper_mode = "UNKNOWN"
        winners = []
        losers = []

        if signal_val == 1: # Expect UP (Bottom)
            sniper_mode = "BUY_SNIPER"
            # At bottom, Sells should be in profit (Winners), Buys in loss (Losers)
            winners = [p for p in positions if p.type == 1] 
            losers = [p for p in positions if p.type == 0]
        elif signal_val == -1: # Expect DOWN (Top)
            sniper_mode = "SELL_SNIPER"
            # At top, Buys should be in profit (Winners), Sells in loss (Losers)
            winners = [p for p in positions if p.type == 0]
            losers = [p for p in positions if p.type == 1]
            
        if not winners or not losers:
            return False

        logger.info(f"[SNIPER] Signal Detected: {sniper_mode} ({confidence:.2f}) | {reason} for Bucket {bucket_id}")

        # 3. Select Best Winner and Worst Loser
        # Sort winners by highest profit (Net PnL)
        winners_with_pnl = []
        for p in winners:
            net, _, _, _ = self.calculate_net_pnl([p])
            winners_with_pnl.append((p, net))
        
        winners_with_pnl.sort(key=lambda x: x[1], reverse=True)
        best_winner, winner_net_pnl = winners_with_pnl[0]
        
        if winner_net_pnl <= 0:
            return False # Best winner is not winning

        # Sort losers by highest loss (most negative Net PnL)
        losers_with_pnl = []
        for p in losers:
            net, _, _, _ = self.calculate_net_pnl([p])
            losers_with_pnl.append((p, net))
            
        losers_with_pnl.sort(key=lambda x: x[1]) # Ascending (most negative first)
        worst_loser, loser_net_pnl = losers_with_pnl[0]
        
        # 4. Calculate Proportional Unwind
        # We want to close a chunk of the winner, say 20% of its volume, or 0.01 lots minimum.
        
        # Determine Unwind Volume for Winner (10% to 20%)
        unwind_vol_winner = round(best_winner.volume * 0.20, 2)
        if unwind_vol_winner < 0.01: unwind_vol_winner = 0.01
        
        if unwind_vol_winner > best_winner.volume:
            unwind_vol_winner = best_winner.volume

        # Calculate Banked Profit from this chunk
        # Profit = (TotalProfit / Volume) * ChunkVolume
        banked_profit = (winner_net_pnl / best_winner.volume) * unwind_vol_winner
        
        if banked_profit <= 0.50: # Minimum bank to make it worth it
            return False

        # Calculate Budget for Loser
        # [CRITICAL FIX] Use safer buffer: 10% profit + $0.50 fixed slippage pad
        budget = (banked_profit * 0.90) - 0.50
        
        if budget <= 0:
            return False
        
        # Calculate Volume of Loser to close
        # LossPerLot = abs(TotalLoss) / Volume
        loss_per_lot = abs(loser_net_pnl) / worst_loser.volume
        
        if loss_per_lot <= 0: loss_per_lot = 0.0001 # Avoid div/0
        
        unwind_vol_loser = budget / loss_per_lot
        
        # [CRITICAL FIX] Always round DOWN to avoid exceeding budget
        # round() can round up (e.g. 0.016 -> 0.02), which would cost more than budget
        unwind_vol_loser = int(unwind_vol_loser * 100) / 100.0
        
        # Sanity Checks
        if unwind_vol_loser < 0.01:
            # Can't close any loser with this small winner amount.
            return False
            
        if unwind_vol_loser > worst_loser.volume:
            unwind_vol_loser = worst_loser.volume
            
        # --- AI PLAN LOG ---
        plan_msg = (
            f"\n>>> [AI SNIPER PLAN] PROPORTIONAL UNWIND <<<\n"
            f"Signal:       {sniper_mode} (Trend Exhaustion)\n"
            f"Action:       Partial Winner -> Partial Loser\n"
            f"Winner:       #{best_winner.ticket} (Close {unwind_vol_winner} lots, Bank ${banked_profit:.2f})\n"
            f"Loser:        #{worst_loser.ticket} (Close {unwind_vol_loser} lots)\n"
            f"Objective:    Reduce Exposure while Maintaining Hedge\n"
            f"----------------------------------------------------"
        )
        import sys
        if sys.platform == 'win32':
            ui_logger.info(plan_msg)
        else:
            try:
                ui_logger.info(plan_msg)
            except Exception:
                ui_logger.info(plan_msg)
        # -------------------

        # Execute (SAFETY: never close loser unless winner banked successfully)
        # Use close_positions with known position data to avoid re-fetch latency and "ticket not found".
        winner_close = {
            'ticket': best_winner.ticket,
            'symbol': best_winner.symbol,
            'volume': unwind_vol_winner,
            'type': best_winner.type,
            'magic': getattr(best_winner, 'magic', 0),
        }

        winner_res = await broker.close_positions([winner_close])
        winner_ok = (
            isinstance(winner_res, dict)
            and best_winner.ticket in winner_res
            and (winner_res[best_winner.ticket] or {}).get('retcode') == mt5.TRADE_RETCODE_DONE
        )
        if not winner_ok:
            logger.error(
                f"[SNIPER] ABORT: Failed to close WINNER #{best_winner.ticket} (vol={unwind_vol_winner}). "
                f"Skipping loser close to avoid realizing red."
            )
            return False

        loser_close = {
            'ticket': worst_loser.ticket,
            'symbol': worst_loser.symbol,
            'volume': unwind_vol_loser,
            'type': worst_loser.type,
            'magic': getattr(worst_loser, 'magic', 0),
        }

        loser_res = await broker.close_positions([loser_close])
        loser_ok = (
            isinstance(loser_res, dict)
            and worst_loser.ticket in loser_res
            and (loser_res[worst_loser.ticket] or {}).get('retcode') == mt5.TRADE_RETCODE_DONE
        )
        if not loser_ok:
            logger.warning(
                f"[SNIPER] PARTIAL: Winner banked, but failed to close LOSER #{worst_loser.ticket} (vol={unwind_vol_loser})."
            )

        return winner_ok or loser_ok

    async def execute_eraser_logic(self, bucket_id: str, broker, market_data: Dict = None, ai_context: Dict = None) -> bool:
        """
        v5.4.0: THE NEXUS HARVESTER (God Tier)
        Combines Partial Scalping with AI Trajectory Prediction.
        Only harvests the winner when the AI says the move is 'exhausted'.
        """
        with self._lock:
            if bucket_id not in self.bucket_stats:
                return False
            
        positions = self.get_positions_in_bucket(bucket_id)
        if len(positions) < 2:
            return False 

        # 1. Identify the "God Mode" Trade (Best Winner)
        # [FIX] Use Net PnL (Profit + Swap + Commission) for accurate sorting
        positions_with_pnl = []
        for p in positions:
            net, _, _, _ = self.calculate_net_pnl([p])
            positions_with_pnl.append((p, net))
            
        sorted_by_profit = sorted(positions_with_pnl, key=lambda x: x[1], reverse=True)
        best_winner, winner_net_pnl = sorted_by_profit[0]
        
        if winner_net_pnl <= 10.0:  # Minimum $10 profit to activate
            return False

        # 2. Identify the "Target" (Worst Loser)
        sorted_by_loss = sorted(positions_with_pnl, key=lambda x: x[1])
        worst_loser, loser_net_pnl = sorted_by_loss[0]

        if loser_net_pnl >= 0:
            return False # No losers to erase

        winner_profit = winner_net_pnl
        loser_loss = abs(loser_net_pnl)
        
        # Define trade type for AI checks
        trade_type = best_winner.type
        
        # 3. AI Context Check (The Nexus Harvester Upgrade)
        # We look at the 'pressure' and 'rsi' from the latest market data
        if market_data and ai_context:
            current_rsi = market_data.get('rsi', 50.0)
            # pressure_dominance might be in ai_context or we derive it
            pressure_dominance = ai_context.get('pressure_dominance', 0.0) 
            # v5.5.0: ARCHITECT TRIGGER (The Wall Check)
            hit_wall = False
            if self.architect:
                structure = self.architect.get_market_structure(best_winner.symbol)
                if structure:
                    if trade_type == 0 and structure['status'] == 'BLOCKED_UP': # Buy hitting Resistance
                        hit_wall = True
                        logger.info(f"[ARCHITECT] TRIGGER: Winner hitting Resistance at {structure['resistance']:.2f}. HARVEST NOW.")
                    elif trade_type == 1 and structure['status'] == 'BLOCKED_DOWN': # Sell hitting Support
                        hit_wall = True
                        logger.info(f"[ARCHITECT] TRIGGER: Winner hitting Support at {structure['support']:.2f}. HARVEST NOW.")

            # DECISION MATRIX: Should we hold for more profit?
            should_hold = False
            
            # Only hold if we haven't hit a wall AND momentum is strong
            if not hit_wall:
                if trade_type == 0: # BUY Trade
                    # Hold if RSI is not yet overbought AND Buying Pressure is strong
                    if current_rsi < 70 and pressure_dominance > 0.2:
                        should_hold = True
                        logger.info(f"[HARVESTER] Holding Winner (+{winner_profit:.2f}) - Momentum is Strong (RSI {current_rsi:.1f}, Press {pressure_dominance:.2f})")
                
                elif trade_type == 1: # SELL Trade
                    # Hold if RSI is not yet oversold AND Selling Pressure is strong
                    if current_rsi > 30 and pressure_dominance < -0.2:
                        should_hold = True
                        logger.info(f"[HARVESTER] Holding Winner (+{winner_profit:.2f}) - Momentum is Strong (RSI {current_rsi:.1f}, Press {pressure_dominance:.2f})")

            if should_hold:
                return False # EXIT. We let the profit grow.

        # 4. Execution (The Scalpel)
        
        # BUFFER: We want to keep $5.00 net profit after the operation
        buffer = 5.0 
        
        # SCENARIO A: FULL ERASE (Winner covers entire Loser)
        if winner_profit > (loser_loss + buffer):
            logger.info(f"[ERASER] FULL ERASE: Winner (+{winner_profit:.2f}) covers Loser (-{loser_loss:.2f})")
            winner_close = {
                'ticket': best_winner.ticket,
                'symbol': best_winner.symbol,
                'volume': best_winner.volume,
                'type': best_winner.type,
                'magic': getattr(best_winner, 'magic', 0),
            }

            winner_res = await broker.close_positions([winner_close])
            winner_ok = (
                isinstance(winner_res, dict)
                and best_winner.ticket in winner_res
                and (winner_res[best_winner.ticket] or {}).get('retcode') == mt5.TRADE_RETCODE_DONE
            )
            if not winner_ok:
                logger.error(
                    f"[ERASER] ABORT: Failed to close WINNER #{best_winner.ticket}. "
                    f"Skipping loser close to avoid realizing red."
                )
                return False

            loser_close = {
                'ticket': worst_loser.ticket,
                'symbol': worst_loser.symbol,
                'volume': worst_loser.volume,
                'type': worst_loser.type,
                'magic': getattr(worst_loser, 'magic', 0),
            }

            loser_res = await broker.close_positions([loser_close])
            loser_ok = (
                isinstance(loser_res, dict)
                and worst_loser.ticket in loser_res
                and (loser_res[worst_loser.ticket] or {}).get('retcode') == mt5.TRADE_RETCODE_DONE
            )
            if not loser_ok:
                logger.warning(
                    f"[ERASER] PARTIAL: Winner closed, but failed to close LOSER #{worst_loser.ticket}."
                )

            return winner_ok or loser_ok

        # SCENARIO B: PROPORTIONAL HARVEST (Was: Partial Scalpel)
        # Instead of closing the full winner (which kills the hedge), we close a portion.
        
        # We want to use the winner to reduce the loser, but NOT at the cost of exposure.
        # Let's close 25% of the winner.
        
        unwind_vol_winner = round(best_winner.volume * 0.25, 2)
        if unwind_vol_winner < 0.01: unwind_vol_winner = 0.01
        if unwind_vol_winner > best_winner.volume: unwind_vol_winner = best_winner.volume
        
        # Calculate Banked Profit
        banked_profit = (winner_profit / best_winner.volume) * unwind_vol_winner
        
        if banked_profit < 2.0: # Minimum $2 to bother
            return False
            
        # Calculate Budget
        # [CRITICAL FIX] Use safer buffer: 10% profit + $0.50 fixed slippage pad
        budget = (banked_profit * 0.90) - 0.50
        
        if budget <= 0:
            return False
        
        # Calculate Loser Volume
        # Use the accurate loss_per_lot calculation if possible, or simple division
        current_price = market_data.get('ask') if worst_loser.type == 1 else market_data.get('bid')
        if not current_price:
             loss_per_lot = loser_loss / worst_loser.volume
        else:
             contract_size = 100.0 if "XAU" in worst_loser.symbol or "GOLD" in worst_loser.symbol else 100000.0
             price_diff = abs(worst_loser.price_open - current_price)
             loss_per_lot = price_diff * contract_size
        
        if loss_per_lot <= 0: loss_per_lot = 0.0001
        
        unwind_vol_loser = budget / loss_per_lot
        
        # [CRITICAL FIX] Always round DOWN to avoid exceeding budget
        unwind_vol_loser = int(unwind_vol_loser * 100) / 100.0
        
        if unwind_vol_loser < 0.01:
            return False
            
        if unwind_vol_loser > worst_loser.volume:
            unwind_vol_loser = worst_loser.volume
            
        logger.info(f"[HARVESTER] Proportional Harvest: Closing {unwind_vol_winner} Winner to kill {unwind_vol_loser} Loser. Bank: ${banked_profit:.2f}")

        winner_close = {
            'ticket': best_winner.ticket,
            'symbol': best_winner.symbol,
            'volume': unwind_vol_winner,
            'type': best_winner.type,
            'magic': getattr(best_winner, 'magic', 0),
        }

        winner_res = await broker.close_positions([winner_close])
        winner_ok = (
            isinstance(winner_res, dict)
            and best_winner.ticket in winner_res
            and (winner_res[best_winner.ticket] or {}).get('retcode') == mt5.TRADE_RETCODE_DONE
        )
        if not winner_ok:
            logger.error(
                f"[HARVESTER] ABORT: Failed to close WINNER #{best_winner.ticket} (vol={unwind_vol_winner}). "
                f"Skipping loser close to avoid realizing red."
            )
            return False

        loser_close = {
            'ticket': worst_loser.ticket,
            'symbol': worst_loser.symbol,
            'volume': unwind_vol_loser,
            'type': worst_loser.type,
            'magic': getattr(worst_loser, 'magic', 0),
        }

        loser_res = await broker.close_positions([loser_close])
        loser_ok = (
            isinstance(loser_res, dict)
            and worst_loser.ticket in loser_res
            and (loser_res[worst_loser.ticket] or {}).get('retcode') == mt5.TRADE_RETCODE_DONE
        )
        if not loser_ok:
            logger.warning(
                f"[HARVESTER] PARTIAL: Winner banked, but failed to close LOSER #{worst_loser.ticket} (vol={unwind_vol_loser})."
            )

        return winner_ok or loser_ok
            
        return False

    async def close_bucket_positions(self, broker, bucket_id: str, symbol: str, trace: dict = None, ppo_guardian=None) -> bool:
        """
        Close all positions in a bucket.

        Args:
            broker: Broker adapter instance
            bucket_id: Bucket to close
            symbol: Trading symbol

        Returns:
            True if all positions closed successfully
        """
        with self._lock:
            if bucket_id not in self.bucket_stats:
                logger.debug(f"Bucket {bucket_id} not in bucket_stats")
                return False

            stats = self.bucket_stats[bucket_id]
            
            # Set state to PENDING_CLOSE to prevent concurrent operations
            self._set_position_state(bucket_id, PositionState.PENDING_CLOSE)
            
            positions = [self.active_positions[t] for t in stats.positions if t in self.active_positions]

        if not positions:
            logger.debug(f"Bucket {bucket_id} has no positions")
            logger.warning(f"Bucket {bucket_id} has no active positions to close")
            # Mark as fully closed
            with self._lock:
                self._set_position_state(bucket_id, PositionState.CLOSED)
            return True

        # [SAFETY] PRE-CLOSE REVALIDATION (Live check just before firing orders)
        # Abort if profit no longer clears execution buffer.
        live_net_pnl, _, _, _ = self.calculate_net_pnl(positions)

        # [FIX] GHOST_PROTOCOL Safety Abort (v5.5.7)
        # If we are trying to clean up a zombie bucket (GHOST_PROTOCOL) but it is currently losing money,
        # we must NOT force close it. We should abort and let the Recovery logic handle the drawdown.
        if stats.exit_reason == 'GHOST_PROTOCOL' and live_net_pnl < 0:
            logger.warning(f"[GHOST ABORT] Bucket {bucket_id} is negative (${live_net_pnl:.2f}). Aborting cleanup to prevent realized loss.")
            # Roll state back to BUCKET_ACTIVE and exit early
            with self._lock:
                self._set_position_state(bucket_id, PositionState.BUCKET_ACTIVE)
            return False

        # Estimate dynamic buffer similar to should_close_bucket
        # Without market_data here, approximate volatility via bucket size
        approx_vol_ratio = 2.0 if len(positions) >= 10 else 1.0  # Increased from 4 to 10
        base_buffer = 2.0 if approx_vol_ratio > 1.5 else 0.50
        
        # [FIX] Relax buffer for GHOST_PROTOCOL to avoid loops on small profits
        if stats.exit_reason == "GHOST_PROTOCOL":
            base_buffer = 0.10

        if len(positions) >= 10:  # Increased from 4 to 10
            base_buffer *= 2.0

        # [CALIBRATION] Apply rolling P95-based buffer when available
        total_volume_for_buffer = sum(pos.volume for pos in positions)
        min_profit_buffer = self._calibrated_profit_buffer(symbol, total_volume_for_buffer, base_buffer)

        allow_emergency_close_loss = str(os.getenv("AETHER_ALLOW_EMERGENCY_CLOSE_LOSS", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        is_emergency_close = (stats.exit_reason or "").find("EMERGENCY") >= 0

        if live_net_pnl < min_profit_buffer and not (allow_emergency_close_loss and is_emergency_close):
            logger.debug(f"Abort Close. Live={live_net_pnl} Buffer={min_profit_buffer} Reason={stats.exit_reason}")
            if str(os.getenv("AETHER_SLIPPAGE_CALIBRATION_LOG", "1")).strip().lower() in ("1", "true", "yes", "on"):
                sc = self._get_slippage_sample_count(symbol)
                p95pl = self._get_slippage_p95_per_lot_usd(symbol)
                logger.info(
                    f"[CALIBRATED BUFFER] {bucket_id}: base=${base_buffer:.2f} calibrated=${min_profit_buffer:.2f} vol={total_volume_for_buffer:.2f} p95_per_lot=${p95pl:.4f} mult={self._slippage_p95_multiplier:.2f} samples={sc}"
                )
            logger.info(
                f"[ABORT CLOSE] {bucket_id}: Live Net ${live_net_pnl:.2f} < Buffer ${min_profit_buffer:.2f}. Holding."
            )
            # Roll state back to BUCKET_ACTIVE and exit early
            self._set_position_state(bucket_id, PositionState.BUCKET_ACTIVE)
            return False

        # [OPTIMIZATION] FIRE FIRST, LOG LATER
        # Initiate the close immediately using the full position objects
        # This is the "Zero-Latency" trigger
        close_batch_id = f"{bucket_id}:{int(time.time())}"
        logger.info(
            f"[BUCKET CLOSE] batch={close_batch_id} symbol={symbol} tickets={[p.ticket for p in positions]}"
        )

        if str(os.getenv("AETHER_SLIPPAGE_CALIBRATION_LOG", "1")).strip().lower() in ("1", "true", "yes", "on"):
            sc = self._get_slippage_sample_count(symbol)
            p95pl = self._get_slippage_p95_per_lot_usd(symbol)
            logger.info(
                f"[CALIBRATED BUFFER] {bucket_id}: base=${base_buffer:.2f} calibrated=${min_profit_buffer:.2f} vol={total_volume_for_buffer:.2f} p95_per_lot=${p95pl:.4f} mult={self._slippage_p95_multiplier:.2f} samples={sc}"
            )

        try:
            close_task = asyncio.create_task(broker.close_positions(positions, trace=trace))
        except TypeError:
            close_task = asyncio.create_task(broker.close_positions(positions))

        close_failures = 0
        close_attempts = len(positions)

        # Pre-execution Analysis (silent calculations for summary)
        total_volume = sum(pos.volume for pos in positions)
        avg_entry = sum(pos.price_open * pos.volume for pos in positions) / total_volume if total_volume > 0 else 0
        total_pnl, _, _, _ = self.calculate_net_pnl(positions)
        bucket_start_time = stats.open_time
        bucket_duration = time.time() - bucket_start_time

        # Generate Detailed Exit Report
        duration_min = bucket_duration / 60.0
        exit_reason_str = stats.exit_reason if stats.exit_reason else "Manual/Unknown"
        
        # Pips calculation (approximate)
        total_pips = 0.0
        if "XAU" in symbol:
            pip_val = 0.01
        elif "JPY" in symbol:
            pip_val = 0.01
        else:
            pip_val = 0.0001
            
        # Calculate raw pips gained
        for pos in positions:
            diff = (pos.price_current - pos.price_open) if pos.type == 0 else (pos.price_open - pos.price_current)
            total_pips += diff / pip_val

        # Formatting duration
        if duration_min < 60:
            dur_str = f"{duration_min:.1f}m"
        else:
            dur_str = f"{duration_min/60.0:.1f}h"

        exit_report = (
            f"\n>>> [EXIT REPORT] 🏁 <<<\n"
            f"Symbol:        {symbol} (Bucket: {bucket_id})\n"
            f"Net PnL:       ${total_pnl:+.2f} ({total_pips:+.1f} pips)\n"
            f"Volume:        {total_volume:.2f} lots ({len(positions)} pos)\n"
            f"----------------------------------------------------\n"
            f"Duration:      {dur_str}\n"
            f"Reason:        {exit_reason_str}\n"
            f"Confidence:    {stats.exit_confidence:.0%}\n"
            f"===================================================="
        )
        logger.info(exit_report)
        
        # Wait for the close to complete
        close_results = await close_task

        # [CALIBRATION] Ingest slippage samples from successful closes
        try:
            for ticket, result in (close_results or {}).items():
                if not isinstance(result, dict):
                    continue
                if result.get('retcode') != mt5.TRADE_RETCODE_DONE:
                    continue
                self._record_close_slippage_sample(
                    symbol=result.get('symbol') or symbol,
                    request_price=result.get('request_price'),
                    fill_price=result.get('price'),
                )
        except Exception:
            pass
        
        # [CRITICAL FIX] Verify Close Results & Retry Failed
        successful_closes = 0
        failed_tickets = []
        
        for ticket, result in close_results.items():
            if result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                successful_closes += 1
            else:
                failed_tickets.append(ticket)
                logger.warning(f"[CLOSE RETRY] Ticket {ticket} failed: {result.get('comment')} ({result.get('retcode')})")

        # Always emit a compact per-ticket result line for auditability
        try:
            for ticket in close_results.keys():
                res = close_results.get(ticket, {}) or {}
                logger.info(
                    f"[BUCKET CLOSE RESULT] batch={close_batch_id} ticket={ticket} retcode={res.get('retcode')} comment={res.get('comment', '')}"
                )
        except Exception as e:
            logger.exception(f"[BUCKET CLOSE RESULT] Failed to emit per-ticket results: {e}")
        
        # RETRY LOOP for failed tickets
        if failed_tickets:
            logger.info(f"[CLOSE RETRY] Attempting to close {len(failed_tickets)} failed positions...")
            # Get position objects for failed tickets
            retry_positions = [self.active_positions[t] for t in failed_tickets if t in self.active_positions]
            
            if retry_positions:
                # Retry once with high priority
                try:
                    retry_results = await broker.close_positions(retry_positions, trace=trace)
                except TypeError:
                    retry_results = await broker.close_positions(retry_positions)

                # [CALIBRATION] Ingest retry slippage samples
                try:
                    for ticket, result in (retry_results or {}).items():
                        if not isinstance(result, dict):
                            continue
                        if result.get('retcode') != mt5.TRADE_RETCODE_DONE:
                            continue
                        self._record_close_slippage_sample(
                            symbol=result.get('symbol') or symbol,
                            request_price=result.get('request_price'),
                            fill_price=result.get('price'),
                        )
                except Exception:
                    pass
                
                for ticket, result in retry_results.items():
                    if result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                        successful_closes += 1
                        # Remove from failed list
                        if ticket in failed_tickets:
                            failed_tickets.remove(ticket)
                        logger.info(f"[CLOSE RETRY] Success: Closed {ticket}")
                    else:
                        logger.error(f"[CLOSE FINAL FAIL] Ticket {ticket}: {result.get('comment')}")
                        
                # [LAST RESORT] Individual Close Loop
                # If batch close failed twice, try one-by-one (slower but safer)
                if failed_tickets:
                    logger.warning(f"[CLOSE LAST RESORT] Switching to individual close for {len(failed_tickets)} tickets")
                    still_failed = []
                    for ticket in failed_tickets:
                        if ticket in self.active_positions:
                            try:
                                res = await broker.close_position(ticket, trace=trace)
                            except TypeError:
                                res = await broker.close_position(ticket)
                            if res:
                                successful_closes += 1
                                logger.info(f"[CLOSE INDIVIDUAL] Success: Closed {ticket}")
                            else:
                                still_failed.append(ticket)
                                logger.error(f"[CLOSE DEAD] Failed to close {ticket} individually")
                    failed_tickets = still_failed

        if failed_tickets:
            logger.error(f"[CLOSE FAILURE] Bucket {bucket_id}: {len(failed_tickets)} positions remained open.")
            
            # PARTIAL CLOSE HANDLING
            # 1. Remove successful tickets from the bucket
            # 2. Keep bucket ACTIVE so it manages the remaining ones
            
            # Identify successful tickets
            all_tickets = [p.ticket for p in positions]
            closed_tickets = [t for t in all_tickets if t not in failed_tickets]
            
            with self._lock:
                # Immediately remove successfully closed tickets from tracking to prevent
                # confusing "stale positions" warnings on the next broker sync.
                for t in closed_tickets:
                    self.active_positions.pop(t, None)
                    self.active_learning_trades.pop(t, None)

                # Update bucket positions list
                stats.positions = [t for t in stats.positions if t in failed_tickets]
                # Reset state to ACTIVE so we keep managing the leftovers
                self._set_position_state(bucket_id, PositionState.BUCKET_ACTIVE)
                
            logger.warning(f"[PARTIAL CLOSE] Bucket {bucket_id} kept ACTIVE with {len(stats.positions)} positions.")

            # Persist cleaned state (remaining open tickets only)
            self._save_state()
            return False # Signal that we are NOT fully done

        if successful_closes == 0:
            logger.error(f"[CLOSE FAILURE] FAILED TO CLOSE ANY POSITIONS in Bucket {bucket_id}. Aborting summary.")
            # Reset state to ACTIVE so we retry later
            self._set_position_state(bucket_id, PositionState.BUCKET_ACTIVE)
            return False

        # [ADD THIS] --- VISUAL BUCKET CLOSE SUMMARY ---
        # Calculate duration string
        minutes = int(bucket_duration // 60)
        seconds = int(bucket_duration % 60)
        duration_str = f"{minutes}m {seconds}s"
        
        # Calculate pips (approximate)
        pip_multiplier = 100 if "XAU" in symbol or "GOLD" in symbol else 10000
        contract_size = 100 if "XAU" in symbol or "GOLD" in symbol else 100000
        try:
            total_pips = (total_pnl / (total_volume * contract_size)) * pip_multiplier if total_volume > 0 else 0.0
        except Exception:
            total_pips = 0.0

        exit_reason = stats.exit_reason if stats.exit_reason else "PROFIT"
        
        # Generate detailed trader-friendly explanation
        if EXPLAINER_AVAILABLE:
            try:
                explainer = TradeExplainer()
                explanation = explainer.explain_bucket_close(
                    symbol=symbol,
                    positions=positions,
                    total_pnl=total_pnl,
                    total_volume=total_volume,
                    bucket_duration=bucket_duration,
                    exit_reason=exit_reason,
                    ai_metrics=None
                )
                print(explanation, flush=True)
            except Exception as e:
                logger.warning(f"[EXPLAINER] Failed to generate detailed explanation: {e}")
                # Fallback to simple summary
                clean_msg = (
                    f"\n>>> [AI EXIT PLAN] CLOSING BUCKET <<<\n"
                    f"Symbol:          {symbol}\n"
                    f"Reason:          {exit_reason}\n"
                    f"Plan:            Close {len(positions)} positions to secure ${total_pnl:.2f}\n"
                    f"Duration:        {duration_str}\n"
                    f"Total PnL:       ${total_pnl:.2f} ({total_pips:+.1f} pips)\n"
                    f"Volume:          {total_volume:.2f} lots\n"
                    f"Status:          EXECUTING NOW...\n"
                    f"----------------------------------------------------"
                )
                ui_logger.info(clean_msg)
        else:
            # Fallback if explainer not available
            clean_msg = (
                f"\n>>> [AI EXIT PLAN] CLOSING BUCKET <<<\n"
                f"Symbol:          {symbol}\n"
                f"Reason:          {exit_reason}\n"
                f"Plan:            Close {len(positions)} positions to secure ${total_pnl:.2f}\n"
                f"Duration:        {duration_str}\n"
                f"Total PnL:       ${total_pnl:.2f} ({total_pips:+.1f} pips)\n"
                f"Volume:          {total_volume:.2f} lots\n"
                f"Status:          EXECUTING NOW...\n"
                f"----------------------------------------------------"
            )
            ui_logger.info(clean_msg)

        # [INTELLIGENCE] Clear High Water Mark Memory
        self.high_water_marks.pop(bucket_id, None)

        # === PPO MEMORY FEEDBACK (Optional) ===
        # Feed realized outcomes back into PPO as experience so session-end evolve() has real data.
        enable_ppo_memory = str(os.getenv("AETHER_ENABLE_PPO_MEMORY", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if enable_ppo_memory and ppo_guardian is not None and hasattr(ppo_guardian, "remember"):
            try:
                with self._lock:
                    tickets_for_learning = list(stats.positions)
                    learning_samples = [
                        (t, (self.active_learning_trades.get(t) or {}))
                        for t in tickets_for_learning
                    ]

                reward = float(total_pnl)
                fed = 0
                for t, meta in learning_samples:
                    obs = meta.get("obs")
                    action = meta.get("action")
                    if not isinstance(obs, (list, tuple)) or len(obs) != 4:
                        continue
                    if not isinstance(action, (list, tuple)) or len(action) != 2:
                        continue
                    try:
                        ppo_guardian.remember(list(obs), list(action), reward)
                        fed += 1
                    except Exception:
                        continue

                if fed > 0:
                    logger.info(f"[PPO_MEMORY] Fed {fed} experience(s) reward={reward:.2f} bucket={bucket_id}")
            except Exception as e:
                logger.debug(f"[PPO_MEMORY] Failed feeding experiences: {e}")
        
        # Fully closed: remove all tickets from tracking immediately
        with self._lock:
            for t in list(stats.positions):
                self.active_positions.pop(t, None)
                self.active_learning_trades.pop(t, None)

            stats.positions = []
            stats.closed = True
            stats.last_update = time.time()
            self.closed_buckets.add(bucket_id)
            self._set_position_state(bucket_id, PositionState.CLOSED)

        self.clear_pending_close(symbol)

        # Persist the cleaned state (no remaining open positions for this bucket)
        self._save_state()

        # === STRUCTURED EXIT SUMMARY ===
        # Generate AI explanation based on actual outcomes
        ai_explanation = (
            f"Closed {len(positions)} position(s) after {bucket_duration/60:.1f} minutes. "
            f"{'Profit target achieved' if total_pnl > 0 else 'Stop loss triggered' if total_pnl < 0 else 'Break-even exit'}. "
            f"Market conditions: {'Favorable' if total_pnl > 0 else 'Adverse'}."
        )
        
        # Construct AI analysis object
        ai_analysis = {
            'decision': "CLOSE",
            'confidence': stats.exit_confidence if hasattr(stats, 'exit_confidence') else 0.0,
            'explanation': ai_explanation
        }

        # Use the centralized TradingLogger
        if TradingLogger:
            try:
                TradingLogger.log_trade_close(
                    symbol=symbol,
                    exit_reason=exit_reason,
                    duration=bucket_duration,
                    pnl_usd=total_pnl,
                    pnl_pips=total_pips,
                    positions_closed=len(positions),
                    ai_analysis=ai_analysis
                )
            except Exception as e:
                logger.error(f"Failed to log trade close: {e}")
        
        # INTEGRATION FIX: Trigger performance metrics update (Enhancement #7)
        if 'on_trade_close' in self.callbacks:
            try:
                self.callbacks['on_trade_close'](
                    profit=total_pnl,
                    duration_seconds=bucket_duration
                )
            except Exception as e:
                logger.warning(f"Failed to update performance metrics: {e}")
        
        # INTEGRATION FIX: Record outcome for model monitor (Enhancement #8)
        if 'on_prediction_outcome' in self.callbacks:
            try:
                actual_direction = 'UP' if total_pnl > 0 else 'DOWN' if total_pnl < 0 else 'NEUTRAL'
                self.callbacks['on_prediction_outcome'](
                    timestamp=bucket_start_time,
                    actual_direction=actual_direction,
                    profit=total_pnl
                )
            except Exception as e:
                logger.warning(f"Failed to record prediction outcome: {e}")
        
        # INTEGRATION FIX: Update strategist win rate (Enhancement #4)
        if 'on_strategist_update' in self.callbacks:
            try:
                self.callbacks['on_strategist_update'](
                    profit=total_pnl,
                    win=(total_pnl > 0)
                )
            except Exception as e:
                logger.warning(f"Failed to update strategist stats: {e}")

        # [PHASE 3] BAD BANK TITHE SYSTEM (Debt Repayment)
        # If we made a profit, we tax it (10%) to pay off toxic debt.
        if total_pnl > 0 and hasattr(self, 'bad_bank'):
             try:
                 tithe = total_pnl * 0.10 # 10% Tax
                 self.bad_bank.deposit_tithe(tithe, symbol)
             except Exception as e:
                 logger.error(f"[BAD BANK] Failed to collect tithe: {e}")

        return True



    def is_bucket_closed_recently(self, bucket_id: str, cooldown_seconds: float = 15.0) -> bool:
        """
        Check if a bucket was closed recently (within cooldown period).

        Args:
            bucket_id: Bucket identifier
            cooldown_seconds: Cooldown period in seconds

        Returns:
            True if bucket was closed recently
        """
        with self._lock:
            if bucket_id not in self.closed_buckets:
                return False

            # Find the close time
            for stats in self.bucket_stats.values():
                if stats.bucket_id == bucket_id and stats.closed:
                    time_since_close = time.time() - stats.last_update
                    return time_since_close < cooldown_seconds

        return False

    def record_learning_trade(self, ticket: int, symbol: str, trade_data: Dict) -> None:
        """
        Record a trade for AI learning purposes.

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            trade_data: Learning data dictionary
        """
        with self._lock:
            self.active_learning_trades[ticket] = trade_data

    def get_learning_data(self, ticket: int) -> Optional[Dict]:
        """Get learning data for a specific ticket."""
        with self._lock:
            return self.active_learning_trades.get(ticket)

    def clear_learning_data(self, ticket: int) -> None:
        """Remove learning data for a closed trade."""
        with self._lock:
            self.active_learning_trades.pop(ticket, None)

    def get_pending_close_symbols(self) -> List[str]:
        """Get symbols that have pending close operations."""
        current_time = time.time()
        with self._lock:
            return [symbol for symbol, timestamp in self.pending_closes.items()
                    if current_time - timestamp < 5.0]  # 5 second timeout

    def set_pending_close(self, symbol: str) -> None:
        """Mark a symbol as having pending close operations."""
        with self._lock:
            self.pending_closes[symbol] = time.time()

    def clear_pending_close(self, symbol: str) -> None:
        """Clear pending close flag for a symbol."""
        with self._lock:
            self.pending_closes.pop(symbol, None)

    def get_positions_in_bucket(self, bucket_id: str) -> List[Position]:
        """
        Get all Position objects belonging to a specific bucket.
        """
        with self._lock:
            if bucket_id not in self.bucket_stats:
                return []
            
            ticket_ids = self.bucket_stats[bucket_id].positions
            positions = []
            for ticket in ticket_ids:
                if ticket in self.active_positions:
                    positions.append(self.active_positions[ticket])
            return positions

    def add_position_to_bucket(self, bucket_id: str, ticket: int) -> bool:
        """
        Add a new position ticket to an existing bucket.
        Thread-safe update of bucket stats.

        Args:
            bucket_id: Bucket identifier
            ticket: Position ticket to add

        Returns:
            bool: True if position added successfully, False otherwise
        """
        with self._lock:
            if bucket_id not in self.bucket_stats:
                logger.error(f"Cannot add ticket #{ticket} to unknown bucket {bucket_id}")
                return False
            
            stats = self.bucket_stats[bucket_id]
            if ticket not in stats.positions:
                stats.positions.append(ticket)
                stats.last_update = time.time()
                # Ensure mode is BUCKET
                stats.mode = BucketMode.BUCKET
                stats.state = PositionState.BUCKET_ACTIVE
                self._save_state()
                logger.info(f"[BUCKET] Added ticket #{ticket} to bucket {bucket_id}. Total positions: {len(stats.positions)}")
                return True
            else:
                logger.warning(f"[BUCKET] Ticket #{ticket} already in bucket {bucket_id}")
                return True