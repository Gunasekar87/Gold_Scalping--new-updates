from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Position:
    ticket: int
    symbol: str
    type: int # 0=BUY, 1=SELL
    volume: float
    price_open: float
    price_current: float # Added current price field
    sl: float
    tp: float
    profit: float
    swap: float
    commission: float = 0.0 # Added commission field
    comment: str = ""
    time: int = 0
    magic: int = 0 # Added magic number field

    def calculate_net_pnl(self) -> float:
        """Calculates Net PnL including Swap and Commission."""
        return self.profit + self.swap + self.commission

@dataclass
class Deal:
    ticket: int
    symbol: str
    type: int
    volume: float
    price: float
    profit: float
    time: int

class BrokerAdapter(ABC):
    """
    The Universal Interface that ALL brokers must obey.
    The AI only talks to this class.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker API."""
        pass

    @abstractmethod
    def get_market_data(self, symbol: str, timeframe: str, limit: int) -> list:
        """Get historical candles."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get the current live price (Bid/Ask)."""
        pass

    @abstractmethod
    def get_tick(self, symbol: str) -> Dict:
        """
        Get full tick data.
        Returns: {'bid': float, 'ask': float, 'time': int, 'flags': int}
        """
        pass

    @abstractmethod
    def execute_order(self, 
                      symbol: str, 
                      action: str, 
                      volume: float, 
                      order_type: str, 
                      price: Optional[float] = None, 
                      sl: Optional[float] = None, 
                      tp: Optional[float] = None,
                      magic: int = 0,
                      comment: str = "",
                      ticket: Optional[int] = None,
                      **kwargs) -> Dict:
        """
        Execute a trade.
        action: OPEN, CLOSE, MODIFY
        order_type: BUY, SELL, BUY_LIMIT, etc.
        """
        pass

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> Optional[list]:
        """Get currently open trades. Returns None on error."""
        pass

    @abstractmethod
    def get_history_deals(self, ticket: int) -> list:
        """Get deal history for a specific ticket."""
        pass

    @abstractmethod
    def get_account_info(self) -> Dict:
        """Get balance, equity, margin."""
        pass

    @abstractmethod
    def is_trade_allowed(self) -> bool:
        """Check if algo trading is allowed."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol properties like point, digits, etc."""
        pass
