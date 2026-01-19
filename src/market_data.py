"""
Market Data Manager - Handles candle data, tick processing, and market state tracking.

This module provides a clean interface for market data operations, including:
- Candle data fetching and caching
- Tick data processing
- Market regime detection
- Adaptive sleep timing based on market conditions

Author: AETHER Development Team
License: MIT
Version: 1.0.0
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from threading import Lock
import MetaTrader5 as mt5

logger = logging.getLogger("MarketDataManager")


class CorrelationMonitor:
    """
    Phase 1 Upgrade: Monitors correlated assets (USDJPY, US500) 
    to provide macro context to the AI.
    """
    def __init__(self, usd_symbol="USDJPY", risk_symbol="US500"):
        self.usd_symbol = usd_symbol
        self.risk_symbol = risk_symbol
        self.initialized = False
        self._initialize_symbols()

    def _initialize_symbols(self):
        """Ensure correlation symbols are available in MT5."""
        for sym in [self.usd_symbol, self.risk_symbol]:
            if not mt5.symbol_select(sym, True):
                logger.warning(f"[WARN] Could not select correlation symbol: {sym}. Macro vision will be limited.")
            else:
                logger.info(f"[MACRO] Macro Eye active: Watching {sym}")
        self.initialized = True

    def get_macro_state(self) -> List[float]:
        """
        Returns a vector representing the macro environment.
        [USD_Velocity, Risk_Velocity]
        Velocity > 0 means Bullish, < 0 means Bearish.
        """
        if not self.initialized:
            return [0.0, 0.0]

        data = {}
        for sym in [self.usd_symbol, self.risk_symbol]:
            # Get last 2 ticks to calculate immediate velocity
            try:
                ticks = mt5.copy_ticks_from(sym, datetime.now(), 10, mt5.COPY_TICKS_ALL)
                if ticks is None or len(ticks) < 2:
                    data[sym] = 0.0
                    continue
                
                # Calculate simple velocity (Price Change)
                # Normalized by price to get percentage
                current = ticks[-1][1] # bid
                prev = ticks[0][1]
                if prev == 0:
                    data[sym] = 0.0
                else:
                    velocity = ((current - prev) / prev) * 10000 # Scaled up for AI
                    data[sym] = velocity
            except Exception as e:
                logger.debug(f"Failed to fetch macro data for {sym}: {e}")
                data[sym] = 0.0

        # Return [USD_Strength, Risk_Sentiment]
        # Note: If USDJPY rises, Dollar is Strong.
        return [data.get(self.usd_symbol, 0.0), data.get(self.risk_symbol, 0.0)]

    def get_macro_state_checked(self) -> Tuple[Optional[List[float]], bool, str]:
        """Return macro vector only when underlying ticks are available.

        This is intended for *strict entry gating*: if correlations are enabled but
        proxy symbols have no ticks, we return ok=False instead of defaulting to 0.0.
        """
        if not self.initialized:
            return None, False, "Macro eye not initialized"

        data = {}
        missing = []
        for sym in [self.usd_symbol, self.risk_symbol]:
            try:
                # [FIX] Use time-based window (60s) instead of raw ticks to filter HFT noise
                # Fetch last 2 M1 candles
                candles = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 0, 2)
                
                if candles is None or len(candles) < 2:
                     missing.append(sym)
                     continue

                current_close = float(candles[-1]['close'])
                prev_close = float(candles[-2]['close']) # Close of previous minute
                
                if prev_close == 0:
                    missing.append(sym)
                    continue

                # Calculate velocity based on 1-minute change
                velocity = ((current_close - prev_close) / prev_close) * 10000
                data[sym] = float(velocity)
            except Exception:
                missing.append(sym)

        if missing:
            return None, False, f"Macro proxies missing ticks: {','.join(missing)}"

        return [data.get(self.usd_symbol, 0.0), data.get(self.risk_symbol, 0.0)], True, "OK"


@dataclass
class MarketRegime:
    """Represents current market conditions and volatility state."""
    regime: str  # "NORMAL", "HIGH_VOL", "PANIC", "LOW_VOL"
    spread_multiplier: float
    sleep_base: float
    should_skip_trading: bool


class MarketStateManager:
    """
    Tracks market volatility and determines trading conditions.
    Provides adaptive behavior based on current market regime.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.spread_history: List[float] = []
        self.price_history: List[float] = []
        self._lock = Lock()

    def update(self, price: float, spread_points: float) -> None:
        """Update market state with new price and spread data."""
        with self._lock:
            self.price_history.append(price)
            self.spread_history.append(spread_points)

            # Keep only recent history
            if len(self.price_history) > self.window_size:
                self.price_history = self.price_history[-self.window_size:]
                self.spread_history = self.spread_history[-self.window_size:]

    def get_regime(self) -> str:
        """Determine current market regime based on volatility."""
        if len(self.spread_history) < 10:
            return "NORMAL"

        with self._lock:
            avg_spread = sum(self.spread_history[-10:]) / 10

            if avg_spread > 50:
                return "PANIC"
            elif avg_spread > 20:
                return "HIGH_VOL"
            elif avg_spread < 5:
                return "LOW_VOL"
            else:
                return "NORMAL"

    def should_skip_trading(self) -> bool:
        """Check if trading should be paused due to extreme volatility."""
        return self.get_regime() == "PANIC"

    def get_adaptive_sleep(self) -> float:
        """Get adaptive sleep time based on market conditions."""
        regime = self.get_regime()

        base_sleep = {
            "PANIC": 5.0,      # Slow down during panic
            "HIGH_VOL": 2.0,   # Moderate slowdown
            "NORMAL": 1.0,     # Normal operation
            "LOW_VOL": 0.5     # Speed up in calm markets
        }.get(regime, 1.0)

        return base_sleep

    def get_volatility_ratio(self) -> float:
        """
        Calculate the ratio of recent volatility to average volatility.
        Returns:
            Ratio > 1.0 means higher than average volatility.
        """
        with self._lock:
            if len(self.price_history) < 20:
                return 1.0
            
            # Calculate returns (absolute price changes)
            prices = self.price_history
            moves = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            
            if not moves:
                return 1.0
                
            # Average move of the last 20 ticks
            avg_move = sum(moves[-20:]) / len(moves[-20:])
            
            if avg_move == 0:
                return 1.0
                
            # Last move
            last_move = moves[-1]
            
            return last_move / avg_move


class CandleManager:
    """
    Manages candle data fetching, caching, and processing.
    Provides different candle formats for various AI components.
    """

    def __init__(self, broker_adapter, timeframe: str = "M1"):
        self.broker = broker_adapter
        self.timeframe = timeframe
        self._cache: Dict[str, Dict] = {}
        self._cache_lock = Lock()
        try:
            self._cache_timeout = float(os.getenv("AETHER_CANDLE_CACHE_TIMEOUT_S", "5"))
        except Exception:
            self._cache_timeout = 5.0
        if self._cache_timeout < 0:
            self._cache_timeout = 0.0

    def _timeframe_seconds(self) -> int:
        tf = (self.timeframe or "M1").upper()
        return {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
        }.get(tf, 60)

    def _normalize_candles(self, candles: List[Dict], *, drop_incomplete: bool = True) -> List[Dict]:
        """Return candles sorted ascending by time; optionally drop the still-forming bar."""
        if not candles:
            return []
        if not isinstance(candles[0], dict) or 'time' not in candles[0]:
            return candles

        # Normalize epoch units (some brokers return ms timestamps)
        for c in candles:
            try:
                ts = int(c.get('time', 0) or 0)
                if ts > 10_000_000_000:
                    c['time'] = int(ts / 1000)
            except Exception:
                pass

        candles_sorted = sorted(candles, key=lambda c: int(c.get('time', 0)))

        if not drop_incomplete or not candles_sorted:
            return candles_sorted

        tf_sec = self._timeframe_seconds()
        now = time.time()
        last_time = int(candles_sorted[-1].get('time', 0))
        # MT5 bars are timestamped at the bar OPEN; if now is before bar close, it's still forming.
        if last_time > 0 and now < (last_time + tf_sec):
            return candles_sorted[:-1]

        return candles_sorted

    def get_history(self, symbol: str, force_refresh: bool = False) -> List[Dict]:
        """
        Get recent price history for trading decisions.

        Args:
            symbol: Trading symbol
            force_refresh: Force fresh data from broker

        Returns:
            List of candle dictionaries with OHLC data
        """
        cache_key = f"{symbol}_history"

        # Check cache first
        if not force_refresh:
            with self._cache_lock:
                if cache_key in self._cache:
                    cached_data, timestamp = self._cache[cache_key]
                    if time.time() - timestamp < self._cache_timeout:
                        return cached_data

        try:
            # Fetch from broker
            candles = self.broker.get_market_data(symbol, self.timeframe, 100)
            if not candles:
                logger.warning(f"No candle data received for {symbol}")
                return []

            # Normalize ordering and remove incomplete candle to prevent lookahead bias
            candles = self._normalize_candles(candles, drop_incomplete=True)

            # Cache the result
            with self._cache_lock:
                self._cache[cache_key] = (candles, time.time())

            return candles

        except Exception as e:
            logger.error(f"Failed to fetch candle history for {symbol}: {e}")
            return []

    def get_full_candles(self, symbol: str) -> List[Dict]:
        """Get full candle data for technical analysis."""
        return self.get_history(symbol)

    def get_nexus_candles(self, symbol: str) -> List[List]:
        """
        Get candle data formatted for NexusBrain predictions.
        Format: [open, high, low, close, volatility]
        CRITICAL: Must match training data features (volatility = high - low)
        """
        candles = self.get_history(symbol)
        # Calculate volatility (High - Low) to match training data
        return [[c['open'], c['high'], c['low'], c['close'], c['high'] - c['low']] for c in candles]

    def get_latest_candle(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent completed candle.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary containing the latest candle data or None if unavailable
        """
        candles = self.get_history(symbol)
        if candles:
            return candles[-1]
        return None

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._cache_lock:
            self._cache.clear()


class MarketDataManager:
    """
    Main interface for market data operations.
    Coordinates between different data sources and provides unified access.
    """

    def __init__(self, broker_adapter, timeframe: str = "M1", config: Dict = None):
        self.broker = broker_adapter
        self.candles = CandleManager(broker_adapter, timeframe)
        self.market_state = MarketStateManager()
        
        # HFT: Order Book Imbalance Cache
        self.last_obi = 0.0
        self.last_obi_time = 0.0
        self.last_obi_applicable = bool(hasattr(broker_adapter, 'get_order_book'))
        self.last_obi_ok = False

        # [PHASE 1] Initialize Correlation Monitor
        self.macro_eye = None
        if config and config.get('trading', {}).get('correlations', {}).get('enable_correlation', False):
            corr_config = config['trading']['correlations']
            self.macro_eye = CorrelationMonitor(
                usd_symbol=corr_config.get('usd_proxy', 'USDJPY'),
                risk_symbol=corr_config.get('risk_proxy', 'US500')
            )

    def get_macro_context(self) -> List[float]:
        """
        Get current macro environment state.
        Returns: [USD_Velocity, Risk_Velocity]
        """
        if self.macro_eye:
            return self.macro_eye.get_macro_state()
        return [0.0, 0.0]

    def get_macro_context_checked(self) -> Tuple[Optional[List[float]], bool, str]:
        """Get macro context with an explicit availability signal.

        Returns:
            (vector, ok, reason)

        Notes:
        - If correlations are disabled (no macro_eye), ok=True and vector=None.
        - If correlations are enabled but proxy ticks are missing, ok=False.
        """
        if not self.macro_eye:
            return None, True, "Correlations disabled"

        try:
            vec, ok, reason = self.macro_eye.get_macro_state_checked()
            return vec, ok, reason
        except Exception as e:
            return None, False, f"Macro context error: {e}"

    def calculate_atr_checked(self, symbol: str, period: int = 14) -> Tuple[Optional[float], bool, str]:
        """ATR with explicit availability signal.

        Intended for strict entry gating to avoid using synthetic fallbacks.
        """
        try:
            candles = self.candles.get_history(symbol)
            if not candles or len(candles) < period + 1:
                return None, False, f"Insufficient candles for ATR: have={len(candles) if candles else 0} need={period + 1}"

            true_ranges = []
            for i in range(1, len(candles)):
                current = candles[i]
                previous = candles[i - 1]

                tr1 = current['high'] - current['low']
                tr2 = abs(current['high'] - previous['close'])
                tr3 = abs(current['low'] - previous['close'])
                true_ranges.append(max(tr1, tr2, tr3))

            if len(true_ranges) < period:
                return None, False, f"Insufficient TR series for ATR: have={len(true_ranges)} need={period}"

            atr = float(sum(true_ranges[-period:]) / period)
            if atr <= 0.0:
                return None, False, "ATR non-positive"
            return atr, True, "OK"
        except Exception as e:
            return None, False, f"ATR error: {e}"

    def calculate_rsi_checked(self, symbol: str, period: int = 14) -> Tuple[Optional[float], bool, str]:
        """RSI with explicit availability signal (no neutral fallback)."""
        try:
            candles = self.candles.get_history(symbol)
            if not candles or len(candles) < period + 1:
                return None, False, f"Insufficient candles for RSI: have={len(candles) if candles else 0} need={period + 1}"

            gains = []
            losses = []
            for i in range(1, len(candles)):
                change = candles[i]['close'] - candles[i - 1]['close']
                if change > 0:
                    gains.append(change)
                    losses.append(0.0)
                else:
                    gains.append(0.0)
                    losses.append(abs(change))

            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            if avg_loss == 0:
                return 100.0, True, "OK"

            rs = avg_gain / avg_loss
            rsi = float(100 - (100 / (1 + rs)))
            return rsi, True, "OK"
        except Exception as e:
            return None, False, f"RSI error: {e}"

    def calculate_trend_strength_checked(self, symbol: str, period: int = 20) -> Tuple[Optional[float], bool, str]:
        """Trend strength with explicit availability signal (no synthetic fallback)."""
        try:
            strength = self.calculate_trend_strength(symbol, period)
            # Valid strength is 0..1 in current implementation.
            if strength is None:
                return None, False, "Trend strength missing"
            strength_f = float(strength)
            if not (0.0 <= strength_f <= 1.0):
                return None, False, f"Trend strength out of range: {strength_f}"
            return strength_f, True, "OK"
        except Exception as e:
            return None, False, f"Trend strength error: {e}"

    def calculate_trend_direction_checked(self, symbol: str, period: int = 20) -> Tuple[Optional[float], bool, str]:
        """Signed trend direction with explicit availability signal.

        Returns:
            (value, ok, reason) where value is in [-1.0, +1.0]

        Notes:
        - This is direction-aware (negative = down, positive = up).
        - Intended for veto logic like "don't buy into a crash".
        """
        try:
            candles = self.candles.get_history(symbol)
            if not candles or len(candles) < max(10, period):
                return None, False, f"Insufficient candles for trend direction: have={len(candles) if candles else 0} need={max(10, period)}"
            v = float(self.calculate_trend_direction(symbol))
            if not (-1.0 <= v <= 1.0):
                return None, False, f"Trend direction out of range: {v}"
            return v, True, "OK"
        except Exception as e:
            return None, False, f"Trend direction error: {e}"

    def get_volume_z_score_checked(self, symbol: str) -> Tuple[Optional[float], bool, str]:
        """Volume Z-score with explicit availability signal (no silent 0.0 fallback)."""
        try:
            candles = self.candles.get_history(symbol)
            if not candles or len(candles) < 51:
                return None, False, f"Insufficient candles for vol_z: have={len(candles) if candles else 0} need=51"
            z = float(self.get_volume_z_score(symbol))
            return z, True, "OK"
        except Exception as e:
            return None, False, f"vol_z error: {e}"

    def calculate_trend_direction(self, symbol: str) -> float:
        """
        Calculates a signed direction-aware trend score.
        Returns:
            float: -1.0 (Strong Downtrend) to +1.0 (Strong Uptrend)
        """
        candles = self.candles.get_history(symbol)
        if not candles or len(candles) < 20:
            return 0.0

        # Simple Linear Regression Slope on last 10 closes
        closes = [c['close'] for c in candles[-10:]]
        n = len(closes)
        
        # X axis is just 0, 1, 2...
        sum_x = sum(range(n))
        sum_y = sum(closes)
        sum_xy = sum(i * closes[i] for i in range(n))
        sum_xx = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
        
        # Normalize slope by price (percentage change per bar)
        current_price = closes[-1]
        if current_price == 0: return 0.0
        
        norm_slope = (slope / current_price) * 10000 # Basis points per bar
        
        # Clamp to -1.0 to 1.0 (Assuming > 5 bps per bar is strong)
        strength = max(-1.0, min(1.0, norm_slope / 5.0))
        
        return strength

    def get_volume_z_score(self, symbol: str) -> float:
        """
        Calculates the Z-Score of the current volume relative to recent history.
        High Z-Score (> 2.0) indicates anomalous activity (breakout/climax).
        """
        candles = self.candles.get_history(symbol)
        if not candles or len(candles) < 50:
            return 0.0
            
        # Use last 50 candles for baseline, excluding the current forming candle
        history_vols = [c['tick_volume'] for c in candles[-51:-1]]
        if not history_vols:
            return 0.0
            
        current_vol = candles[-1]['tick_volume']
        
        avg_vol = sum(history_vols) / len(history_vols)
        
        # Calculate StdDev
        variance = sum((v - avg_vol) ** 2 for v in history_vols) / len(history_vols)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
            
        z_score = (current_vol - avg_vol) / std_dev
        return z_score

    def get_order_book_imbalance(self, symbol: str) -> float:
        """
        HFT Layer 7: Calculates Order Book Imbalance (OBI).
        Returns:
            -1.0 (Strong Sell Pressure) to +1.0 (Strong Buy Pressure)
        """
        try:
            # Fetch Depth of Market (Level 2)
            if hasattr(self.broker, 'get_order_book'):
                book = self.broker.get_order_book(symbol)
                if not book:
                    return 0.0
                
                # Calculate total volume on Bid and Ask sides
                # Assuming book returns {'bids': [{'price': 1.1, 'volume': 10}, ...], 'asks': ...}
                total_bid_vol = sum(item.get('volume', 0) for item in book.get('bids', []))
                total_ask_vol = sum(item.get('volume', 0) for item in book.get('asks', []))
                
                if total_bid_vol + total_ask_vol == 0:
                    return 0.0
                    
                # OBI Formula: (BidVol - AskVol) / (BidVol + AskVol)
                obi = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
                return obi
            
            return 0.0
        except Exception as e:
            # logger.error(f"OBI Calculation failed: {e}") # Suppress spam
            return 0.0

    def get_order_book_imbalance_checked(self, symbol: str) -> Tuple[Optional[float], bool, str]:
        """Order book imbalance with explicit availability signal.

        Returns:
            (obi, ok, reason)

        Notes:
        - ok=True only when an order-book-derived value is available.
        - If the broker does not support order books, ok=False and obi=None.
        - If the broker supports order books but no book is returned, ok=False.
        """
        try:
            if not hasattr(self.broker, 'get_order_book'):
                return None, False, "OBI not supported"

            book = self.broker.get_order_book(symbol)
            if not book:
                return None, False, "Order book unavailable"

            bids = book.get('bids', []) or []
            asks = book.get('asks', []) or []
            total_bid_vol = float(sum(item.get('volume', 0) for item in bids))
            total_ask_vol = float(sum(item.get('volume', 0) for item in asks))
            if total_bid_vol + total_ask_vol <= 0:
                return None, False, "Order book empty"

            obi = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
            return float(obi), True, "OK"
        except Exception as e:
            return None, False, f"OBI error: {e}"

    def get_symbol_properties(self, symbol: str) -> Dict[str, Any]:
        """
        Get standardized properties for a symbol.
        
        Returns:
            Dict with keys: 'pip_multiplier', 'point_value', 'pip_divisor'
        """
        if "XAU" in symbol or "GOLD" in symbol:
            return {
                'pip_multiplier': 100,
                'point_value': 0.01,
                'pip_divisor': 100
            }
        elif "JPY" in symbol:
            return {
                'pip_multiplier': 100,
                'point_value': 0.01,
                'pip_divisor': 100
            }
        else:
            return {
                'pip_multiplier': 10000,
                'point_value': 0.0001,
                'pip_divisor': 10000
            }

    def get_tick_data(self, symbol: str) -> Optional[Dict]:
        """
        Get current tick data with validation.
        Includes HFT OBI calculation.

        Args:
            symbol: Trading symbol

        Returns:
            Tick data dict or None if failed
        """
        try:
            tick = self.broker.get_tick(symbol)
            if not tick or 'bid' not in tick or 'ask' not in tick:
                logger.warning(f"Invalid tick data received for {symbol}")
                return None

            # Normalize tick timestamp (seconds vs milliseconds)
            try:
                raw_ts = tick.get('time', 0) or 0
                ts = float(raw_ts)
                if ts > 10_000_000_000:
                    ts = ts / 1000.0
                tick['time'] = float(ts) if ts > 0 else 0.0
            except Exception:
                tick['time'] = 0.0

            current_time = time.time()

            # Freshness metrics (for gating NEW orders)
            try:
                tick_ts = float(tick.get('time', 0.0) or 0.0)
                tick_age_s = abs(current_time - tick_ts) if tick_ts > 0 else float('inf')
            except Exception:
                tick_age_s = float('inf')

            tf_s = 60
            try:
                tf_s = int(self.candles._timeframe_seconds())
            except Exception:
                tf_s = 60

            candle_close_age_s = float('inf')
            candle_close_ts = 0.0
            try:
                latest = self.candles.get_latest_candle(symbol)
                if latest and isinstance(latest, dict):
                    c_ts = float(latest.get('time', 0) or 0)
                    if c_ts > 10_000_000_000:
                        c_ts = c_ts / 1000.0
                    if c_ts > 0:
                        candle_close_ts = float(c_ts) + float(tf_s)
                        candle_close_age_s = current_time - candle_close_ts
                        if candle_close_age_s < 0:
                            candle_close_age_s = 0.0
            except Exception:
                candle_close_age_s = float('inf')
                candle_close_ts = 0.0

            tick['tick_age_s'] = float(tick_age_s)
            tick['candle_close_age_s'] = float(candle_close_age_s)
            tick['candle_close_ts'] = float(candle_close_ts)
            tick['timeframe_s'] = int(tf_s)

            # Update market state
            if 'bid' in tick and 'ask' in tick:
                mid_price = (tick['bid'] + tick['ask']) / 2
                spread_points = abs(tick['ask'] - tick['bid'])
                self.market_state.update(mid_price, spread_points)

            # HFT: Update Order Book Imbalance (every 1s to avoid API spam)
            if current_time - self.last_obi_time > 1.0:
                obi_applicable = hasattr(self.broker, 'get_order_book')
                self.last_obi_applicable = bool(obi_applicable)
                obi_val, obi_ok, _ = self.get_order_book_imbalance_checked(symbol)

                # Keep numeric cache for downstream consumers, but expose ok/applicable flags
                # so entry logic can avoid treating missing OBI as a real 0.0 signal.
                self.last_obi_ok = bool(obi_ok)
                self.last_obi = float(obi_val) if (obi_val is not None and obi_ok) else 0.0
                self.last_obi_time = current_time
            
            # Inject OBI into tick data for downstream logic
            tick['obi'] = self.last_obi
            tick['obi_ok'] = bool(getattr(self, 'last_obi_ok', False))
            tick['obi_applicable'] = bool(getattr(self, 'last_obi_applicable', False))

            return tick

        except Exception as e:
            logger.error(f"Failed to get tick data for {symbol}: {e}")
            return None

    def calculate_multi_timeframe_trends(self, symbol: str) -> Dict[str, str]:
        """
        Calculate trend direction for M1, M5, and M15 timeframes.
        Returns: Dict with 'm1_trend', 'm5_trend', 'm15_trend' (UP/DOWN/NEUTRAL).
        """
        trends = {}
        timeframes = {
            'm1': mt5.TIMEFRAME_M1,
            'm5': mt5.TIMEFRAME_M5,
            'm15': mt5.TIMEFRAME_M15
        }
        
        for name, tf in timeframes.items():
            try:
                # Fetch last 20 candles for this timeframe
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, 20)
                if rates is None or len(rates) < 20:
                    trends[f'{name}_trend'] = "NEUTRAL"
                    continue
                
                # Identify columns directly from numpy record array
                closes = rates['close']
                
                # Calculate slope (simple linreg)
                n = len(closes)
                sum_x = sum(range(n))
                sum_y = sum(closes)
                sum_xy = sum(i * closes[i] for i in range(n))
                sum_xx = sum(i * i for i in range(n))
                
                denom = (n * sum_xx - sum_x * sum_x)
                slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0
                
                # Normalize slope to basis points
                current_price = closes[-1]
                if current_price == 0:
                    trends[f'{name}_trend'] = "NEUTRAL"
                    continue
                    
                norm_slope = (slope / current_price) * 10000
                
                # Determine trend with threshold
                # > 0.5 bps per bar is a decent trend
                if norm_slope > 0.5:
                    trends[f'{name}_trend'] = "UP"
                elif norm_slope < -0.5:
                    trends[f'{name}_trend'] = "DOWN"
                else:
                    trends[f'{name}_trend'] = "NEUTRAL"
                    
            except Exception as e:
                # Silently fail to neutral on error to avoid spam
                trends[f'{name}_trend'] = "NEUTRAL"
                
        return trends

    def get_volatility_ratio(self) -> float:
        """Get current volatility ratio from market state."""
        return self.market_state.get_volatility_ratio()

    def validate_market_conditions(self, symbol: str, tick: Dict) -> Tuple[bool, str]:
        """
        Validate if market conditions are suitable for trading.

        Args:
            symbol: Trading symbol
            tick: Current tick data

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check market regime
        if self.market_state.should_skip_trading():
            return False, "Market in panic mode - trading paused"

        # Check spread
        spread_points = abs(tick['ask'] - tick['bid'])
        regime = self.market_state.get_regime()

        max_spread = {
            "PANIC": 100,
            "HIGH_VOL": 50,
            "NORMAL": 30,
            "LOW_VOL": 20
        }.get(regime, 30)

        if spread_points > max_spread:
            return False, f"Spread too wide: {spread_points:.1f} > {max_spread}"

        return True, "Market conditions OK"

    def get_adaptive_sleep_time(self) -> float:
        """Get recommended sleep time based on market conditions."""
        return self.market_state.get_adaptive_sleep()

    def calculate_bollinger_bands_checked(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> Tuple[Optional[Dict[str, float]], bool, str]:
        """Bollinger Bands with explicit availability signal (no synthetic zero bands)."""
        try:
            candles = self.candles.get_history(symbol)
            if not candles or len(candles) < period:
                return None, False, f"Insufficient candles for BB: have={len(candles) if candles else 0} need={period}"

            closes = [float(c['close']) for c in candles[-period:]]
            sma = sum(closes) / period
            variance = sum(((x - sma) ** 2) for x in closes) / period
            std = variance ** 0.5
            bands = {
                'upper': sma + (std * std_dev),
                'middle': sma,
                'lower': sma - (std * std_dev)
            }
            return bands, True, "OK"
        except Exception as e:
            return None, False, f"BB error: {e}"

    def calculate_macd(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """
        Calculate MACD metrics (pure python implementation).
        Returns dict with keys: 'value', 'signal', 'histogram'.
        """
        try:
            candles = self.candles.get_history(symbol)
            if not candles or len(candles) < slow + signal + 10:
                return {'value': 0.0, 'signal': 0.0, 'histogram': 0.0}
                
            closes = [float(c['close']) for c in candles]
            
            def ema(data, period):
                if not data: return []
                alpha = 2 / (period + 1)
                ema_values = [data[0]]  # Seed with first SMA (or just first price approx)
                for price in data[1:]:
                    ema_values.append((price * alpha) + (ema_values[-1] * (1 - alpha)))
                return ema_values
                
            ema_fast = ema(closes, fast)
            ema_slow = ema(closes, slow)
            
            # Ensure lengths match (they should)
            min_len = min(len(ema_fast), len(ema_slow))
            ema_fast = ema_fast[-min_len:]
            ema_slow = ema_slow[-min_len:]
            
            macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
            
            if len(macd_line) < signal:
                 return {'value': 0.0, 'signal': 0.0, 'histogram': 0.0}
                 
            signal_line = ema(macd_line, signal)
            
            last_macd = macd_line[-1]
            last_signal = signal_line[-1]
            hist = last_macd - last_signal
            
            return {'value': last_macd, 'signal': last_signal, 'histogram': hist}
            
        except Exception as e:
            logger.error(f"MACD calc failed: {e}")
            return {'value': 0.0, 'signal': 0.0, 'histogram': 0.0}

    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility measurement.

        Args:
            symbol: Trading symbol
            period: Period for ATR calculation (default 14)

        Returns:
            ATR value or 0.0010 if calculation fails
        """
        try:
            candles = self.candles.get_history(symbol)
            if len(candles) < period + 1:
                logger.warning(f"Insufficient candle data for ATR calculation: {len(candles)} < {period + 1}")
                return 0.0010

            # Calculate True Range for each candle
            true_ranges = []
            for i in range(1, len(candles)):
                current = candles[i]
                previous = candles[i-1]

                # True Range = max(high - low, |high - prev_close|, |low - prev_close|)
                tr1 = current['high'] - current['low']
                tr2 = abs(current['high'] - previous['close'])
                tr3 = abs(current['low'] - previous['close'])

                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)

            # Calculate ATR as simple moving average of True Ranges
            if len(true_ranges) >= period:
                atr = sum(true_ranges[-period:]) / period
                logger.debug(f"[ATR] Calculated ATR for {symbol}: {atr:.6f}")
                return atr
            else:
                logger.warning(f"Insufficient true ranges for ATR: {len(true_ranges)} < {period}")
                return 0.0010

        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return 0.0010

    def calculate_trend_strength(self, symbol: str, period: int = 20) -> float:
        """
        Calculate trend strength based on price direction consistency.

        Args:
            symbol: Trading symbol
            period: Period for trend calculation (default 20)

        Returns:
            Trend strength between 0.0 (no trend) and 1.0 (strong trend)
        """
        try:
            candles = self.candles.get_history(symbol)
            if len(candles) < period:
                logger.warning(f"Insufficient candle data for trend calculation: {len(candles)} < {period}")
                return 0.0

            # Calculate price changes
            price_changes = []
            for i in range(1, min(period + 1, len(candles))):
                change = candles[-i]['close'] - candles[-i-1]['close']
                direction = 1 if change > 0 else -1 if change < 0 else 0
                price_changes.append(direction)

            if not price_changes:
                return 0.0

            # Calculate trend consistency (how many consecutive moves in same direction)
            consistency = 0
            current_streak = 0
            prev_direction = 0

            for direction in price_changes:
                if direction == prev_direction and direction != 0:
                    current_streak += 1
                elif direction != 0:
                    current_streak = 1
                    prev_direction = direction
                else:
                    current_streak = 0
                    prev_direction = 0

                consistency = max(consistency, current_streak)

            # Normalize to 0-1 scale
            trend_strength = min(consistency / period, 1.0)
            logger.debug(f"[TREND] Calculated trend strength for {symbol}: {trend_strength:.3f}")
            return trend_strength

        except Exception as e:
            logger.error(f"Error calculating trend strength for {symbol}: {e}")
            return 0.0

    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI).
        Used for momentum filtering in Elastic Defense Protocol.
        """
        try:
            candles = self.candles.get_history(symbol)
            if len(candles) < period + 1:
                return 50.0 # Neutral fallback

            gains = []
            losses = []
            
            for i in range(1, len(candles)):
                change = candles[i]['close'] - candles[i-1]['close']
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            # Simple RSI calculation (not smoothed for speed, but sufficient)
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            return 50.0

    def calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """
        Calculate Bollinger Bands (Upper, Middle, Lower).
        Used for Momentum Exhaustion logic.
        """
        try:
            candles = self.candles.get_history(symbol)
            if len(candles) < period:
                return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}

            closes = [c['close'] for c in candles[-period:]]
            sma = sum(closes) / period
            
            variance = sum([((x - sma) ** 2) for x in closes]) / period
            std = variance ** 0.5
            
            return {
                'upper': sma + (std * std_dev),
                'middle': sma,
                'lower': sma - (std * std_dev)
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands for {symbol}: {e}")
            return {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}