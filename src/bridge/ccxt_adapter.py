import ccxt
import time
from typing import Dict, Optional
from .broker_interface import BrokerAdapter, Position, Deal
import logging

logger = logging.getLogger("CCXTAdapter")

class CCXTAdapter(BrokerAdapter):
    def __init__(self, exchange_id='binance', api_key=None, secret=None, sandbox=False):
        self.exchange_id = exchange_id
        self.exchange_class = getattr(ccxt, exchange_id)
        self.exchange = self.exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        if sandbox:
            self.exchange.set_sandbox_mode(True)
        self.positions_cache = {}

    def connect(self) -> bool:
        try:
            self.exchange.load_markets()
            logger.info(f"Connected to {self.exchange_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {e}")
            return False

    def get_market_data(self, symbol: str, timeframe: str, limit: int) -> list:
        # Map MT5 timeframe to CCXT
        tf_map = {'M1': '1m', 'M5': '5m', 'H1': '1h', 'D1': '1d'}
        ccxt_tf = tf_map.get(timeframe, '1m')
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, ccxt_tf, limit=limit)
            # Format: [timestamp, open, high, low, close, volume]
            data = []
            for x in ohlcv:
                ts = int(x[0])
                # CCXT timestamps are typically milliseconds since epoch
                if ts > 10_000_000_000:
                    ts = int(ts / 1000)
                data.append({'time': ts, 'open': x[1], 'high': x[2], 'low': x[3], 'close': x[4], 'volume': x[5]})
            return data
        except Exception as e:
            logger.error(f"Fetch OHLCV failed: {e}")
            return []

    def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception:
            return 0.0

    def get_tick(self, symbol: str) -> Dict:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'time': int(time.time()), # CCXT timestamps are ms, but we might want seconds or just current time
                'flags': 0
            }
        except Exception:
            return None

    def execute_order(self, symbol, action, volume, order_type, price=None, sl=0.0, tp=0.0, magic=0, comment="", **kwargs) -> Dict:
        strict_entry = bool(kwargs.get('strict_entry', False) or getattr(self, 'strict_entry', False))
        strict_ok = kwargs.get('strict_ok', None)
        if strict_entry and action == "OPEN" and strict_ok is not True:
            msg = f"STRICT_BLOCK: OPEN rejected (strict_ok={strict_ok}) symbol={symbol}"
            logger.warning(msg)
            return {"ticket": None, "retcode": -1, "comment": msg}
        try:
            side = 'buy' if order_type == 'BUY' else 'sell'
            type_ = 'limit' if price else 'market'
            
            params = {}
            if comment:
                params['clientId'] = comment # Some exchanges support this

            if action == "OPEN":
                order = self.exchange.create_order(symbol, type_, side, volume, price, params)
                return {"ticket": order['id'], "retcode": 0}
            elif action == "CLOSE":
                # CCXT doesn't have "close position", we just do opposite trade
                side = 'sell' if order_type == 'BUY' else 'buy' # Close BUY means SELL
                order = self.exchange.create_order(symbol, 'market', side, volume, params=params)
                return {"ticket": order['id'], "retcode": 0}
                
        except Exception as e:
            logger.error(f"Order Execution Failed: {e}")
            return {"ticket": None, "retcode": -1}

    def get_positions(self, symbol: Optional[str] = None) -> list:
        # CCXT fetch_positions is not supported by all exchanges, but fetch_balance is
        # For futures, fetch_positions works. For spot, we check balance.
        # This is a simplified version for Futures (e.g. Binance Futures)
        try:
            positions = self.exchange.fetch_positions([symbol] if symbol else None)
            # Map to a standard format similar to MT5 for the bot
            mapped = []
            for p in positions:
                if float(p['contracts']) > 0: # Only open positions
                    mapped.append(Position(
                        ticket=p['id'] or symbol, # Use symbol as ticket if ID missing
                        symbol=p['symbol'],
                        type=0 if p['side'] == 'long' else 1, # 0=BUY, 1=SELL (MT5 standard)
                        volume=float(p['contracts']),
                        price_open=float(p['entryPrice']),
                        sl=0.0, # CCXT doesn't always give SL/TP in position info easily
                        tp=0.0,
                        profit=float(p['unrealizedPnl']),
                        swap=0.0,
                        comment="",
                        time=int(p['timestamp'] / 1000) if p['timestamp'] else int(time.time())
                    ))
            return mapped
        except Exception as e:
            logger.error(f"Get Positions Failed: {e}")
            return []

    def get_history_deals(self, ticket: int) -> list:
        # Hard to map 1:1 with MT5 deals. 
        # We might fetch_my_trades and filter by order id (ticket)
        try:
            trades = self.exchange.fetch_my_trades(limit=10) # Simplified
            # Filter by ticket if possible
            return [Deal(
                ticket=t['id'],
                symbol=t['symbol'],
                type=0 if t['side'] == 'buy' else 1,
                volume=float(t['amount']),
                price=float(t['price']),
                profit=float(t['info'].get('realizedPnl', 0.0)), # Exchange specific
                time=int(t['timestamp']/1000)
            ) for t in trades if t['order'] == str(ticket) or t['id'] == str(ticket)]
        except Exception:
            return []

    def get_account_info(self) -> Dict:
        try:
            balance = self.exchange.fetch_balance()
            return {
                "balance": balance['total']['USDT'], # Assuming USDT base
                "equity": balance['total']['USDT'], # Approx for spot
                "profit": 0.0
            }
        except (KeyError, ValueError, TypeError, Exception) as e:
            logger.warning(f"Failed to get account info: {e}")
            return {}

    def is_trade_allowed(self) -> bool:
        # If we are connected, we assume trading is allowed
        return self.exchange.check_required_credentials()

    def get_symbol_info(self, symbol: str) -> Dict:
        try:
            market = self.exchange.market(symbol)
            return {
                "point": market['precision']['price'] if 'precision' in market else 0.00001, # Fallback
                "digits": 5, # Approximation
                "volume_min": market['limits']['amount']['min'] if 'limits' in market else 0.001,
                "volume_step": market['precision']['amount'] if 'precision' in market else 0.001
            }
        except (KeyError, ValueError, TypeError, Exception) as e:
            logger.warning(f"Failed to get symbol info for {symbol}: {e}")
            return {"point": 0.00001}
