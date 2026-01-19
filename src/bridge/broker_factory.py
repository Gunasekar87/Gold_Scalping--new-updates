from .mt5_adapter import MT5Adapter
from .ccxt_adapter import CCXTAdapter
import os

class BrokerFactory:
    @staticmethod
    def get_broker(broker_type="MT5", credentials=None):
        if credentials is None:
            credentials = {}

        if broker_type == "MT5":
            return MT5Adapter(
                login=credentials.get('mt5_login'),
                password=credentials.get('mt5_password'),
                server=credentials.get('mt5_server')
            )
        elif broker_type == "BINANCE":
            return CCXTAdapter(
                exchange_id='binance',
                api_key=credentials.get('api_key') or os.getenv("BINANCE_API_KEY"),
                secret=credentials.get('secret_key') or os.getenv("BINANCE_SECRET_KEY")
            )
        elif broker_type == "BYBIT":
            return CCXTAdapter(
                exchange_id='bybit',
                api_key=credentials.get('api_key') or os.getenv("BYBIT_API_KEY"),
                secret=credentials.get('secret_key') or os.getenv("BYBIT_SECRET_KEY")
            )
        else:
            raise ValueError(f"Unknown Broker Type: {broker_type}")
