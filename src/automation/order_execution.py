import MetaTrader5 as mt5
import logging
import yaml
import os

logger = logging.getLogger("OrderExecution")

# Load Config
try:
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
        trading_config = config.get("trading", {})
        DEFAULT_MAGIC = trading_config.get("magic_number", 888888)
        DEFAULT_SLIPPAGE = trading_config.get("max_slippage", 10)
        # Map string filling type to MT5 constant
        filling_mode = trading_config.get("filling_mode", "IOC")
        if filling_mode == "FOK":
            DEFAULT_FILLING = mt5.ORDER_FILLING_FOK
        elif filling_mode == "RETURN":
            DEFAULT_FILLING = mt5.ORDER_FILLING_RETURN
        else:
            DEFAULT_FILLING = mt5.ORDER_FILLING_IOC
except Exception as e:
    logger.warning(f"Failed to load settings.yaml: {e}. Using defaults.")
    DEFAULT_MAGIC = 888888
    DEFAULT_SLIPPAGE = 10
    DEFAULT_FILLING = mt5.ORDER_FILLING_IOC

class OrderExecution:
    @staticmethod
    def execute_order(action, symbol, order_type, price, vol, sl=0.0, tp=0.0, ticket=None, magic=None, comment="AETHER AI"):
        """
        Executes an order directly via MT5 Python API.
        Supports: OPEN, CLOSE, MODIFY
        """
        if magic is None:
            magic = DEFAULT_MAGIC
        
        # Map string types to MT5 constants
        mt5_type = mt5.ORDER_TYPE_BUY
        if order_type == "SELL": mt5_type = mt5.ORDER_TYPE_SELL
        elif order_type == "BUY_STOP": mt5_type = mt5.ORDER_TYPE_BUY_STOP
        elif order_type == "SELL_STOP": mt5_type = mt5.ORDER_TYPE_SELL_STOP
        
        # Determine MT5 Action
        mt5_action = mt5.TRADE_ACTION_DEAL
        if action == "PENDING" or "STOP" in order_type or "LIMIT" in order_type:
            mt5_action = mt5.TRADE_ACTION_PENDING
        elif action == "MODIFY":
            mt5_action = mt5.TRADE_ACTION_SLTP
            
        request = {
            "action": mt5_action,
            "symbol": symbol,
            "volume": float(vol),
            "type": mt5_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": DEFAULT_SLIPPAGE,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": DEFAULT_FILLING,
        }
        
        if ticket:
            request["position"] = ticket # For Closing
            request["sl"] = float(sl)    # For Modifying
            request["tp"] = float(tp)    # For Modifying
            
        result = mt5.order_send(request)
        
        if result is None:
             logger.error("Order Send failed, result is None")
             return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode == 10027:
                logger.error("ORDER BLOCKED: Enable 'Algo Trading' in MT5 Toolbar!")
            else:
                logger.error(f"Order Failed: {result.comment} ({result.retcode})")
            return None
        else:
            logger.info(f"Order Executed Successfully. Ticket: {result.order} | {action} {symbol} {vol} @ {price}")
            return result.order
