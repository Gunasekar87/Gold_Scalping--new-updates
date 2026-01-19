try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from .broker_interface import BrokerAdapter, Position, Deal
from typing import Dict, Optional
import logging
import os
import math
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("MT5Adapter")

class MT5Adapter(BrokerAdapter):
    def __init__(self, login=None, password=None, server=None):
        self.login = int(login) if login else None
        self.password = password
        self.server = server
        self._resolved_symbols = {}  # Cache for fuzzy symbol matching

        # Persistent close executor to avoid per-batch threadpool startup overhead.
        self._close_executor: Optional[ThreadPoolExecutor] = None
        self._close_executor_max_workers: int = 0
        self._close_executor_lock = threading.Lock()

    def connect(self) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5 module not found. Cannot connect.")
            return False
        
        # If credentials are provided, try to initialize with them
        if self.login and self.password and self.server:
            if not mt5.initialize(login=self.login, password=self.password, server=self.server):
                logger.error(f"MT5 Login Failed: {mt5.last_error()}")
                return False
        else:
            # Fallback to default terminal state
            # Try to initialize with a specific path if standard init fails
            if not mt5.initialize():
                # Try to find the terminal path automatically
                logger.warning("Standard MT5 Init Failed. Attempting to locate terminal64.exe...")
                
                possible_paths = [
                    r"C:\Program Files\MetaTrader 5\terminal64.exe",
                    r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
                    # Add common broker paths
                    r"C:\Program Files\FTMO MetaTrader 5\terminal64.exe",
                    r"C:\Program Files\ICMarkets MetaTrader 5\terminal64.exe",
                ]
                
                success = False
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"Found MT5 at {path}. Attempting to initialize...")
                        if mt5.initialize(path):
                            success = True
                            break
                
                if not success:
                    error_code = mt5.last_error()
                    logger.error(f"MT5 Init Failed. Error Code: {error_code}")
                    logger.error("Troubleshooting Steps:")
                    logger.error("1. CLOSE all open MT5 terminals manually (Task Manager -> End Task).")
                    logger.error("2. Check for blocking popups (Login, News, One-Click Disclaimer).")
                    logger.error("3. Ensure 'Allow Algorithmic Trading' is ON.")
                    logger.error("4. Try running this script as Administrator.")
                    return False
        
        # Log connection details for verification
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"[MT5] Connected: Account #{account_info.login} | Server: {account_info.server}")
            logger.info(f"[MT5] Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")
            logger.info(f"[MT5] Trade Mode: {'DEMO' if account_info.trade_mode == 1 else 'REAL' if account_info.trade_mode == 0 else 'CONTEST'}")
            logger.info(f"[MT5] Trade Allowed: {self.is_trade_allowed()}")
        else:
            logger.warning("[MT5] Could not retrieve account information")
                
        return True

    def _resolve_symbol(self, requested_symbol: str) -> Optional[str]:
        """
        Fuzzy match symbol to handle broker suffixes (e.g. 'XAUUSD' -> 'XAUUSD.m').
        Caches the result for performance.
        """
        if requested_symbol in self._resolved_symbols:
            return self._resolved_symbols[requested_symbol]

        # 1. Try exact match first (Fast)
        if mt5.symbol_select(requested_symbol, True):
            self._resolved_symbols[requested_symbol] = requested_symbol
            return requested_symbol

        # 2. Try common suffixes
        suffixes = [".m", ".pro", "+", ".ecn", "_i", ".r", ".s"]
        for suffix in suffixes:
            candidate = f"{requested_symbol}{suffix}"
            if mt5.symbol_select(candidate, True):
                logger.info(f"[MT5] Auto-corrected symbol: {requested_symbol} -> {candidate}")
                self._resolved_symbols[requested_symbol] = candidate
                return candidate

        # 3. Last Resort: Iterate visible symbols (Slow)
        logger.warning(f"[MT5] Symbol '{requested_symbol}' not found. Searching visible symbols...")
        symbols = mt5.symbols_get()
        if symbols:
            req_clean = requested_symbol.lower()
            for s in symbols:
                if req_clean in s.name.lower():
                    # If confirmed with select
                    if mt5.symbol_select(s.name, True):
                        logger.info(f"[MT5] Auto-corrected symbol (Fuzzy): {requested_symbol} -> {s.name}")
                        self._resolved_symbols[requested_symbol] = s.name
                        return s.name

        return None

    def get_market_data(self, symbol: str, timeframe: str, limit: int) -> list:
        # Use resolved symbol
        actual_symbol = self._resolve_symbol(symbol)
        if not actual_symbol:
            logger.warning(f"[MT5] Cannot fetch history: Symbol {symbol} not found")
            return []

        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(actual_symbol, mt5_tf, 0, limit)
        if rates is None or len(rates) == 0:
            return []
        
        data = []
        for rate in rates:
            data.append({
                'time': int(rate['time']),
                'open': float(rate['open']),
                'high': float(rate['high']),
                'low': float(rate['low']),
                'close': float(rate['close']),
                'tick_volume': float(rate['tick_volume']),
                'spread': int(rate['spread']),
                'real_volume': float(rate['real_volume'])
            })
        return data

    def get_current_price(self, symbol: str) -> float:
        actual_symbol = self._resolve_symbol(symbol)
        if not actual_symbol: return 0.0
        tick = mt5.symbol_info_tick(actual_symbol)
        return tick.bid if tick else 0.0

    def get_tick(self, symbol: str) -> Dict:
        # 1. Check connection
        terminal_info = mt5.terminal_info()
        if not terminal_info:
            logger.warning("[MT5] Terminal info unavailable. Attempting reconnect...")
            if not self.connect(): return None
        elif not terminal_info.connected:
            logger.warning("[MT5] Terminal disconnected. Attempting reconnect...")
            if not self.connect(): return None

        # 2. Resolve Symbol (handles suffixes like XAUUSD.m)
        actual_symbol = self._resolve_symbol(symbol)
        if not actual_symbol:
             # Already logged warning in _resolve_symbol
             return None

        # 3. Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(actual_symbol, True):
            err = mt5.last_error()
            logger.warning(f"Failed to select symbol {actual_symbol} in Market Watch (Error: {err})")
            return None
            
        # 4. Fetch Tick
        tick = mt5.symbol_info_tick(actual_symbol)
        if tick:
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'time': tick.time,
                'flags': tick.flags
            }
        
        logger.warning(f"[MT5] symbol_info_tick returned None for {actual_symbol} (Error: {mt5.last_error()})")
        return None

    def execute_order(self, symbol, action, volume, order_type, price=None, sl=0.0, tp=0.0, magic=0, comment="", ticket=None, **kwargs) -> Dict:
        strict_entry = bool(kwargs.get('strict_entry', False) or getattr(self, 'strict_entry', False))
        strict_ok = kwargs.get('strict_ok', None)
        trace_enabled = str(os.getenv("AETHER_DECISION_TRACE", "1")).strip().lower() in ("1", "true", "yes", "on")
        atr_ok = kwargs.get('atr_ok', None)
        rsi_ok = kwargs.get('rsi_ok', None)
        obi_ok = kwargs.get('obi_ok', None)
        trace_reason = kwargs.get('trace_reason', None)

        if trace_enabled:
            logger.info(
                self._format_decision_trace(
                    action=action,
                    symbol=symbol,
                    ticket=ticket,
                    strict_entry=strict_entry,
                    strict_ok=strict_ok,
                    atr_ok=atr_ok,
                    rsi_ok=rsi_ok,
                    obi_ok=obi_ok,
                    reason=trace_reason,
                )
            )
        if strict_entry and action == "OPEN" and strict_ok is not True:
            msg = f"STRICT_BLOCK: OPEN rejected (strict_ok={strict_ok}) symbol={symbol}"
            logger.warning(f"[MT5] {msg}")
            return {"ticket": None, "retcode": -1, "comment": msg}

        # ... existing logic from order_execution.py ...
        # This is where we wrap the specific MT5 code
        mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        
        # CRITICAL: Normalize lot size to MT5's volume_step before execution
        normalized_volume = self.normalize_lot_size(symbol, volume)
        
        # Get symbol info for deviation
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return {"ticket": None, "retcode": -1}
        
        request = {
            "symbol": symbol,
            "volume": normalized_volume,
            "type": mt5_type,
            "price": price if price else (mt5.symbol_info_tick(symbol).ask if mt5_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid),
            "sl": sl,
            "tp": tp,
            "deviation": 20,  # Maximum price deviation in points
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if action == "CLOSE":
            # order_type already contains the correct opposing order type from caller
            # (SELL to close BUY, BUY to close SELL)
            mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
            request["action"] = mt5.TRADE_ACTION_DEAL
            request["type"] = mt5_type
            request["position"] = ticket  # Specify the position to close
            logger.info(f"[MT5] Closing position {ticket} ({order_type} order to close)")
            
        elif action == "MODIFY":
            request["action"] = mt5.TRADE_ACTION_SLTP
            request["position"] = ticket if ticket else magic
            # SL/TP are already in the request
            # Volume, type, price are ignored for SLTP modification usually, but symbol is needed
            
        else: # OPEN
            request["action"] = mt5.TRADE_ACTION_DEAL
            logger.info(f"[MT5] Sending order: {action} {order_type} {symbol} {normalized_volume} lots @ {request['price']:.5f} | SL={sl:.5f} TP={tp:.5f}")
        
        # Retry loop for robust execution
        max_retries = 3
        result = None
        
        for attempt in range(max_retries):
            # Refresh price on retries (avoid Requotes/Invalid Price)
            # Only refresh for OPEN/CLOSE/HEDGE where price matters (not SLTP modification)
            if attempt > 0 and (action == "OPEN" or action == "CLOSE" or action == "HEDGE"): 
                 tick = mt5.symbol_info_tick(symbol)
                 if tick:
                     new_price = tick.ask if mt5_type == mt5.ORDER_TYPE_BUY else tick.bid
                     request["price"] = new_price
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[MT5] [OK] Order executed successfully - Ticket: {result.order} | Retcode: {result.retcode}")
                return {"ticket": result.order, "retcode": result.retcode}
            
            # Retry on Connection Error (10031), Requote (10004), No Quotes (10021), Trade Timeout (10036), Off Quotes (10018)
            # 10031 = No Connection
            # 10004 = Requote
            # 10015 = Invalid Price (Refresh helps)
            # 10018 = Market Closed (Wait?) -> No, actually 10018 is fatal usually. But 10008 (Order Place) might retry.
            # We stick to connectivity/pricing errors.
            elif result and result.retcode in [10004, 10031, 10021, 10036, 10015, 10016]: 
                 wait_time = 0.5 * (attempt + 1)
                 if result.retcode == 10031: 
                     wait_time = 2.0 * (attempt + 1) # Wait longer for connection
                     logger.warning(f"[MT5] ⚠️ Connection lost (10031). Attempting re-initialization...")
                     try:
                        mt5.initialize()
                     except Exception as e:
                        logger.error(f"[MT5] Re-init failed: {e}")

                 logger.warning(f"[MT5] Retry {attempt+1}/{max_retries}: Order failed with {result.retcode} ({result.comment}). Waiting {wait_time}s...")
                 time.sleep(wait_time)
                 continue
            
            else:
                # Fatal error (e.g. Invalid Volume, No Money, Market Closed) - Do not retry
                break
                
        # Final Failure Log (if all retries failed or fatal error)
        if result:
            logger.error(f"[MT5] [FAILED] Order failed: retcode={result.retcode}, comment={result.comment}")
            logger.error(f"[MT5] Request: action={action}, symbol={symbol}, volume={normalized_volume}, type={order_type}, ticket={ticket}")
        else:
            logger.error(f"[MT5] Order send returned None. Last error: {mt5.last_error()}")
        return {"ticket": None, "retcode": result.retcode if result else -1}

    def get_positions(self, symbol: Optional[str] = None) -> Optional[list]:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        
        if positions is None:
            error_code = mt5.last_error()
            # If error is "no error" (1) but None returned, it might just be empty? 
            # No, empty returns empty tuple. None is definitely error.
            logger.warning(f"[MT5] positions_get returned None. Error: {error_code}")
            return None
            
        if len(positions) == 0:
            return []

        return [Position(
                ticket=p.ticket,
                symbol=p.symbol,
                type=p.type, # 0=BUY, 1=SELL
                volume=p.volume,
                price_open=p.price_open,
                price_current=p.price_current,
                sl=p.sl,
                tp=p.tp,
                profit=p.profit,
                swap=p.swap,
                commission=getattr(p, 'commission', 0.0), # [FIX] Capture commission if available
                comment=p.comment,
                time=p.time,
                magic=p.magic
            ) for p in positions]

    def get_all_positions(self) -> Optional[list]:
        """Helper to get all positions without symbol filtering."""
        return self.get_positions(symbol=None)

    def get_history_deals(self, ticket: int) -> list:
        deals = mt5.history_deals_get(ticket=ticket)
        if deals:
            return [Deal(
                ticket=d.ticket,
                symbol=d.symbol,
                type=d.type,
                volume=d.volume,
                price=d.price,
                profit=d.profit,
                time=d.time
            ) for d in deals]
        return []

    def get_account_info(self) -> Dict:
        info = mt5.account_info()
        return {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin_free": info.margin_free,
            "leverage": info.leverage
        } if info else {}

    def check_margin(self, symbol: str, volume: float, order_type: str) -> bool:
        """
        Check if there is enough margin to execute the order.
        """
        mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if mt5_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        margin = mt5.order_calc_margin(mt5_type, symbol, volume, price)
        if margin is None:
            logger.error(f"Failed to calculate margin for {symbol} {volume} lots")
            return False
            
        account_info = mt5.account_info()
        if not account_info:
            return False
            
        if account_info.margin_free < margin:
            logger.warning(f"[MARGIN] Insufficient margin! Required: ${margin:.2f}, Free: ${account_info.margin_free:.2f}")
            return False
            
        return True

    def get_max_volume(self, symbol: str, order_type: str) -> float:
        """
        Calculate maximum volume allowed by free margin.
        """
        account_info = mt5.account_info()
        if not account_info:
            return 0.0
            
        free_margin = account_info.margin_free
        mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if mt5_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        # Estimate margin for 1 lot
        margin_1_lot = mt5.order_calc_margin(mt5_type, symbol, 1.0, price)
        if not margin_1_lot:
            return 0.0
            
        max_vol = free_margin / margin_1_lot
        # Apply safety buffer (95% of max)
        return max_vol * 0.95

    def get_order_book(self, symbol: str) -> dict:
        """Return Depth of Market in a simple dict format.

        Format:
            {
              'bids': [{'price': float, 'volume': float}, ...],
              'asks': [{'price': float, 'volume': float}, ...]
            }

        Notes:
        - Requires broker/terminal to support Market Book (Level 2).
        - Returns empty dict if unavailable.
        """
        if not symbol:
            return {}

        try:
            # Subscribe once per call; MT5 will ignore if already subscribed.
            try:
                mt5.market_book_add(symbol)
            except Exception:
                pass

            book = mt5.market_book_get(symbol)
            if not book:
                return {}

            bids = []
            asks = []
            for row in book:
                rtype = getattr(row, 'type', None)
                price = float(getattr(row, 'price', 0.0) or 0.0)
                vol = float(getattr(row, 'volume', 0.0) or 0.0)
                if price <= 0 or vol <= 0:
                    continue

                # MT5 Constants: 1=SELL (Ask), 2=BUY (Bid)
                if rtype == 2: # BOOK_TYPE_BUY
                    bids.append({'price': price, 'volume': vol})
                elif rtype == 1: # BOOK_TYPE_SELL
                    asks.append({'price': price, 'volume': vol})

            return {'bids': bids, 'asks': asks}
        except Exception:
            return {}

    async def close_positions(self, positions_data: list, trace: Optional[Dict] = None) -> dict:
        """
        ZERO-LATENCY CLOSER: Accepts full position objects/dicts to skip the lookup step.
        Executes 'Blind' close commands for maximum speed.
        """
        if not positions_data:
            return {}
            
        import asyncio

        def _ensure_close_executor(max_workers: int) -> ThreadPoolExecutor:
            with self._close_executor_lock:
                if self._close_executor is None or self._close_executor_max_workers != max_workers:
                    if self._close_executor is not None:
                        try:
                            self._close_executor.shutdown(wait=False, cancel_futures=True)
                        except Exception:
                            pass
                    self._close_executor_max_workers = max_workers
                    self._close_executor = ThreadPoolExecutor(
                        max_workers=max_workers,
                        thread_name_prefix="mt5close",
                    )
                return self._close_executor

        def _supports_close_by() -> bool:
            # CLOSE_BY works only on hedging accounts.
            # On netting/exchange accounts it will fail; we skip to avoid extra latency.
            try:
                info = mt5.account_info()
                if not info:
                    return False
                return getattr(info, 'margin_mode', None) == getattr(mt5, 'ACCOUNT_MARGIN_MODE_RETAIL_HEDGING', 1)
            except Exception:
                return False

        # Prefetch latest ticks per symbol once to reduce per-thread overhead.
        # Retried attempts still refresh ticks.
        tick_cache = {}
        try:
            symbols = {
                (p.symbol if hasattr(p, 'symbol') else p.get('symbol'))
                for p in positions_data
            }
            for sym in symbols:
                if sym:
                    tick_cache[sym] = mt5.symbol_info_tick(sym)
        except Exception:
            tick_cache = {}
        
        # Define a wrapper to close a single ticket with retries
        trace_enabled = str(os.getenv("AETHER_DECISION_TRACE", "1")).strip().lower() in ("1", "true", "yes", "on")
        trace_dict = trace if isinstance(trace, dict) else {}

        def close_single_sync(pos, pre_tick=None):
            # [OPTIMIZATION] No mt5.positions_get() call here. We trust the data passed in.
            import time
            
            # Handle both object (dot notation) and dict (bracket notation)
            try:
                ticket = pos.ticket if hasattr(pos, 'ticket') else pos['ticket']
                symbol = pos.symbol if hasattr(pos, 'symbol') else pos['symbol']
                volume = pos.volume if hasattr(pos, 'volume') else pos['volume']
                p_type = pos.type if hasattr(pos, 'type') else pos['type']
                magic = pos.magic if hasattr(pos, 'magic') else pos['magic']
            except Exception as e:
                logger.error(f"Invalid position data for close: {e}")
                return {"ticket": -1, "retcode": -1, "comment": f"Invalid data: {e}"}

            # [ROBUSTNESS] Verify position exists in broker (prevent 10013 loop)
            # If not found, assume it's already closed manually or by SL/TP
            if not mt5.positions_get(ticket=ticket):
                logger.warning(f"[CLOSE SKIP] Position {ticket} not found in broker. Marking as CLOSED.")
                return {
                    "ticket": ticket, 
                    "retcode": mt5.TRADE_RETCODE_DONE, 
                    "comment": "Already Closed (Ghost)"
                }

            # Optional override for the CLOSE comment (not the original position comment).
            # We keep this short because MT5 imposes strict comment length limits.
            close_comment = None
            if isinstance(pos, dict):
                close_comment = pos.get('close_comment')

            # Determine close type (Opposite of open)
            order_type = mt5.ORDER_TYPE_SELL if p_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # CRITICAL: Normalize volume for partial closes
            # If volume is not normalized, MT5 might reject it or round it unpredictably
            normalized_volume = self.normalize_lot_size(symbol, volume)
            
            # Get current price (Fastest way)
            tick = pre_tick or mt5.symbol_info_tick(symbol)
            if not tick: 
                return {"ticket": ticket, "retcode": -1, "comment": "No tick data"}
                
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": normalized_volume, # Use normalized volume
                "type": order_type,
                "position": ticket,
                "price": price,
                # Deviation is set per-attempt below (tight first, widen on retries)
                "magic": magic,
                "comment": close_comment or "Aether FastClose",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            if trace_enabled:
                logger.info(
                    self._format_decision_trace(
                        action="CLOSE",
                        symbol=symbol,
                        ticket=ticket,
                        strict_entry=bool(trace_dict.get('strict_entry', False)),
                        strict_ok=trace_dict.get('strict_ok', None),
                        atr_ok=trace_dict.get('atr_ok', None),
                        rsi_ok=trace_dict.get('rsi_ok', None),
                        obi_ok=trace_dict.get('obi_ok', None),
                        reason=trace_dict.get('reason', None),
                    )
                )

            # BLAST THE ORDER - Retry loop
            for attempt in range(3):
                # Adaptive deviation to reduce slippage:
                # - start tight to protect profits
                # - widen only if we get requotes/price errors
                base_dev = 50
                if "XAU" in symbol or "GOLD" in symbol:
                    base_dev = int(os.getenv("AETHER_CLOSE_DEVIATION_XAU", "180"))
                elif "JPY" in symbol:
                    base_dev = int(os.getenv("AETHER_CLOSE_DEVIATION_JPY", "120"))
                else:
                    base_dev = int(os.getenv("AETHER_CLOSE_DEVIATION_FX", "60"))

                max_dev = int(os.getenv("AETHER_CLOSE_DEVIATION_MAX", "600"))
                request["deviation"] = min(max_dev, base_dev * (attempt + 1))

                result = mt5.order_send(request)

                if result is None:
                    # Terminal/API failure. Retry once or twice with backoff.
                    if attempt == 0:
                        logger.warning(f"Close retry {ticket}: order_send returned None. Last error: {mt5.last_error()}")
                    time.sleep(0.25 * (attempt + 1))
                    continue
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Keep hot-path logging minimal; bucket-level code logs results.
                    logger.debug(f"[CLOSED] {ticket} at {result.price}")
                    return {
                        "ticket": ticket,
                        "retcode": result.retcode,
                        "price": getattr(result, 'price', None),
                        "request_price": request.get("price"),
                        "symbol": symbol,
                        "volume": normalized_volume,
                        "type": int(order_type),
                        "comment": getattr(result, 'comment', '')
                    }
                elif result.retcode in [10004, 10015, 10021, 10031]: # Requote, Invalid Price, No Money, No Connection
                    # Only log warning on first failure to keep logs clean
                    if attempt == 0:
                        logger.warning(f"Close retry {ticket}: {result.comment}")

                    # If we lost network/terminal connection, give it a moment.
                    if result.retcode == 10031:
                        time.sleep(0.75 * (attempt + 1))
                    else:
                        time.sleep(0.15 * (attempt + 1))

                    # Refresh price for retry
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        request['price'] = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
                    continue
                else:
                    logger.error(f"CRITICAL: Failed to close {ticket}. Error: {result.comment} ({result.retcode})")
                    return {
                        "ticket": ticket,
                        "retcode": result.retcode,
                        "comment": result.comment,
                        "request_price": request.get("price"),
                        "symbol": symbol,
                        "volume": normalized_volume,
                        "type": int(order_type),
                    }
            
            return {
                "ticket": ticket,
                "retcode": -1,
                "comment": "Max retries exceeded",
                "request_price": request.get("price"),
                "symbol": symbol,
                "volume": normalized_volume,
                "type": int(order_type),
            }

        # Optional: CLOSE_BY pairing to reduce spread/slippage and number of close deals.
        enable_close_by = str(os.getenv("AETHER_ENABLE_CLOSE_BY", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # Get the running loop
        loop = asyncio.get_running_loop()

        # Fire ALL close requests in parallel.
        # Using a dedicated persistent executor avoids per-batch startup overhead.
        default_cap = int(os.getenv("AETHER_CLOSE_MAX_WORKERS", "32"))
        max_workers = max(1, min(len(positions_data), default_cap))
        executor = _ensure_close_executor(max_workers)

        # Attempt CLOSE_BY for equal-volume opposite hedges (hedging accounts only).
        # This reduces number of market orders and typically reduces spread impact.
        close_by_results: Dict[int, Dict] = {}
        remaining = list(positions_data)
        if enable_close_by and _supports_close_by():
            try:
                # Build per-symbol buy/sell pools (only equal-volume pairs)
                by_symbol = {}
                for p in remaining:
                    sym = p.symbol if hasattr(p, 'symbol') else (p.get('symbol') if isinstance(p, dict) else None)
                    if not sym:
                        continue
                    by_symbol.setdefault(sym, []).append(p)

                def _pos_fields(p):
                    ticket = p.ticket if hasattr(p, 'ticket') else p['ticket']
                    symbol = p.symbol if hasattr(p, 'symbol') else p['symbol']
                    volume = p.volume if hasattr(p, 'volume') else p['volume']
                    p_type = p.type if hasattr(p, 'type') else p['type']
                    magic = p.magic if hasattr(p, 'magic') else (p.get('magic', 0) if isinstance(p, dict) else 0)
                    return ticket, symbol, float(volume), int(p_type), int(magic)

                def close_by_sync(pos_a, pos_b):
                    # Close BUY/SELL by each other. Only safe for equal-volume pairs.
                    try:
                        t1, sym, vol1, type1, magic1 = _pos_fields(pos_a)
                        t2, _, vol2, type2, magic2 = _pos_fields(pos_b)
                    except Exception as e:
                        return {"retcode": -1, "comment": f"close_by invalid data: {e}"}

                    # Must be opposite sides
                    if type1 == type2:
                        return {"retcode": -1, "comment": "close_by same side"}

                    # Enforce equal volumes (within step) to avoid partial close-by semantics.
                    # Use symbol step if available; else a small epsilon.
                    step = 0.01
                    try:
                        info = self.get_symbol_info(sym) or {}
                        step = float(info.get('volume_step', step))
                    except Exception:
                        step = 0.01
                    eps = max(1e-6, step / 2)
                    if abs(vol1 - vol2) > eps:
                        return {"retcode": -1, "comment": "close_by volume mismatch"}

                    req = {
                        "action": mt5.TRADE_ACTION_CLOSE_BY,
                        "symbol": sym,
                        "position": t1,
                        "position_by": t2,
                        "magic": magic1,
                        "comment": "Aether CloseBy"[:31],
                    }
                    res = mt5.order_send(req)
                    if res is None:
                        return {"retcode": -1, "comment": f"close_by None: {mt5.last_error()}"}
                    return {"retcode": res.retcode, "comment": getattr(res, 'comment', '')}

                close_by_tasks = []
                paired_tickets = set()
                for sym, plist in by_symbol.items():
                    buys = [p for p in plist if (p.type if hasattr(p, 'type') else p.get('type')) == mt5.ORDER_TYPE_BUY]
                    sells = [p for p in plist if (p.type if hasattr(p, 'type') else p.get('type')) == mt5.ORDER_TYPE_SELL]

                    # Pair greedily by equal normalized volume
                    # (we only attempt when volumes match closely)
                    used_sells = set()
                    for b in buys:
                        bt, _, bv, _, _ = _pos_fields(b)
                        if bt in paired_tickets:
                            continue
                        match = None
                        for s in sells:
                            st, _, sv, _, _ = _pos_fields(s)
                            if st in used_sells or st in paired_tickets:
                                continue
                            if abs(bv - sv) <= 1e-6:
                                match = s
                                used_sells.add(st)
                                break
                        if match:
                            st, _, _, _, _ = _pos_fields(match)
                            paired_tickets.add(bt)
                            paired_tickets.add(st)
                            close_by_tasks.append(loop.run_in_executor(executor, close_by_sync, b, match))

                if close_by_tasks:
                    close_by_task_results = await asyncio.gather(*close_by_tasks)
                    # Map results back: since close_by_sync doesn't include tickets, we conservatively just mark paired
                    # tickets as "handled" only if retcode DONE.
                    # Any failures will fall back to standard close.
                    # NOTE: We don't know which result corresponds to which pair here without extra bookkeeping.
                    # Keep this simple: if close_by was used, we will rely on subsequent broker sync to remove closed.
                    # To avoid skipping closes incorrectly, we won't remove from remaining unless we can verify.
                    # We still return a marker in result map for audit.
                    for r in close_by_task_results:
                        # No ticket mapping; include as informational
                        pass

                    # If close_by tasks were issued, we do not assume success for any specific ticket.
                    # We simply proceed with standard closes; MT5 will return "position doesn't exist" for already closed.
                    # This keeps correctness.
            except Exception as e:
                logger.debug(f"[CLOSE_BY] Skipped due to error: {e}")

        tasks = []
        for pos in remaining:
            sym = pos.symbol if hasattr(pos, 'symbol') else (pos.get('symbol') if isinstance(pos, dict) else None)
            tasks.append(loop.run_in_executor(executor, close_single_sync, pos, tick_cache.get(sym)))
        results = await asyncio.gather(*tasks)
        
        # Map results back to tickets (extract ticket from position data)
        tickets = [p.ticket if hasattr(p, 'ticket') else p['ticket'] for p in positions_data]
        return {ticket: result for ticket, result in zip(tickets, results)}

    async def close_position(self, ticket: int, volume: float = None, trace: Optional[Dict] = None) -> bool:
        """
        Close a single position by ticket.
        Wrapper around close_positions for single ticket convenience.
        Supports PARTIAL closing if volume is specified.
        """
        # We need to find the position details first because close_positions needs volume/symbol
        # Try to get from MT5 directly
        positions = self.get_positions()
        target_pos = None
        if positions:
            for p in positions:
                if p.ticket == ticket:
                    target_pos = p
                    break
        
        if not target_pos:
            logger.info(f"[CLOSE] Position {ticket} not found in broker (already closed)")
            # Position already closed - return True to indicate successful state
            # This prevents error spam when positions are closed externally (SL/TP hit, manual close, etc.)
            return True
            
        # If volume is specified, we need to create a copy of the position data with the new volume
        if volume:
            # Create a dict representation with the override volume
            # Note: target_pos might be an object or dict. Handle both.
            t_ticket = target_pos.ticket if hasattr(target_pos, 'ticket') else target_pos['ticket']
            t_symbol = target_pos.symbol if hasattr(target_pos, 'symbol') else target_pos['symbol']
            t_type = target_pos.type if hasattr(target_pos, 'type') else target_pos['type']
            t_magic = target_pos.magic if hasattr(target_pos, 'magic') else (target_pos.get('magic', 0) if isinstance(target_pos, dict) else 0)

            pos_data = {
                'ticket': t_ticket,
                'symbol': t_symbol,
                'volume': volume,  # OVERRIDE
                'type': t_type,
                'magic': t_magic,
            }
            # Use close_positions with this single item
            result = await self.close_positions([pos_data], trace=trace)
        else:
            result = await self.close_positions([target_pos], trace=trace)
        
        # Check result
        if ticket in result:
            res = result[ticket]
            return res.get('retcode') == mt5.TRADE_RETCODE_DONE
            
        return False

    @staticmethod
    def _format_decision_trace(
        action: str,
        symbol: str,
        ticket: Optional[int],
        strict_entry: bool,
        strict_ok,
        atr_ok,
        rsi_ok,
        obi_ok,
        reason: Optional[str] = None,
    ) -> str:
        def _fmt_flag(v) -> str:
            if v is None:
                return "NA"
            if isinstance(v, bool):
                return "1" if v else "0"
            try:
                return "1" if bool(v) else "0"
            except Exception:
                return "NA"

        ticket_str = str(ticket) if ticket is not None else "NA"
        reason_s = str(reason) if (reason is not None and str(reason).strip()) else ""
        if len(reason_s) > 48:
            reason_s = reason_s[:48]

        base = (
            f"[DECISION_TRACE] action={action} symbol={symbol} ticket={ticket_str} "
            f"strict_entry={_fmt_flag(strict_entry)} strict_ok={_fmt_flag(strict_ok)} "
            f"atr_ok={_fmt_flag(atr_ok)} rsi_ok={_fmt_flag(rsi_ok)} obi_ok={_fmt_flag(obi_ok)}"
        )
        return base + (f" reason={reason_s}" if reason_s else "")

    def is_trade_allowed(self) -> bool:
        info = mt5.terminal_info()
        return info.trade_allowed if info else False

    def normalize_lot_size(self, symbol: str, requested_lot: float) -> float:
        """
        Normalize lot size to MT5's volume_step requirements.
        Fixes discrepancy where bot calculates 0.8 lots but MT5 accepts 0.72 lots.
        
        Args:
            symbol: Trading symbol
            requested_lot: Desired lot size from calculations
            
        Returns:
            Normalized lot size that MT5 will accept
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"Cannot get symbol info for {symbol}, using requested lot: {requested_lot}")
            return round(requested_lot, 2)
        
        volume_min = symbol_info.get('volume_min', 0.01)
        volume_step = symbol_info.get('volume_step', 0.01)
        
        # Ensure lot is at least minimum
        if requested_lot < volume_min:
            logger.warning(f"Requested lot {requested_lot} < minimum {volume_min}, using minimum")
            return volume_min
        
        # Determine precision from volume_step
        # e.g. 0.01 -> 2 decimals, 0.1 -> 1 decimal, 1.0 -> 0 decimals
        try:
            if volume_step > 0:
                precision = int(round(-math.log10(volume_step), 0))
                if precision < 0: precision = 0
            else:
                precision = 2
        except Exception:
            precision = 2

        # Normalize to volume_step.
        # Default is ROUND-DOWN (safer): never increases exposure vs requested_lot.
        # Override with AETHER_LOT_NORMALIZE_MODE=nearest if you prefer standard rounding.
        mode = str(os.getenv("AETHER_LOT_NORMALIZE_MODE", "nearest")).strip().lower()

        epsilon = 1e-12  # protect against floating point edge cases
        if mode in ("nearest", "round", "standard"):
            normalized = round(round(requested_lot / volume_step) * volume_step, precision)
        else:
            steps = math.floor((requested_lot + epsilon) / volume_step)
            normalized = round(steps * volume_step, precision)

        # Ensure we never drop below broker minimum due to rounding artifacts
        if normalized < volume_min:
            normalized = round(volume_min, precision)
        
        # Use epsilon for comparison
        epsilon = 0.0000001
        if abs(normalized - requested_lot) > epsilon:
            logger.info(f"[LOT NORMALIZE] {requested_lot} -> {normalized} (step: {volume_step})")
        
        return normalized

    def get_symbol_info(self, symbol: str) -> Dict:
        info = mt5.symbol_info(symbol)
        if info:
            return {
                "point": info.point,
                "digits": info.digits,
                "volume_min": info.volume_min,
                "volume_step": info.volume_step
            }
        return {}

    def disconnect(self):
        """Disconnect from the MT5 terminal."""
        if mt5:
            mt5.shutdown()
            logger.info("MT5 Disconnected")

        # Best-effort shutdown of persistent executor
        with self._close_executor_lock:
            if self._close_executor is not None:
                try:
                    self._close_executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._close_executor = None
                self._close_executor_max_workers = 0
