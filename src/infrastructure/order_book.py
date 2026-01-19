import logging
import numpy as np
from collections import OrderedDict

logger = logging.getLogger("OrderBook")

class OrderBook:
    """
    The "Vision" of the AI.
    Maintains a real-time Level 2 Order Book (Depth of Market).
    
    Why this matters:
    - Price is just the surface. The Order Book shows the *pressure*.
    - Large bid walls support price. Large ask walls resist it.
    - "Spoofing" and "Layering" can be detected here.
    """
    def __init__(self, symbol, depth=10):
        self.symbol = symbol
        self.depth = depth
        # Price -> Volume
        self.bids = {} 
        self.asks = {}
        self.last_update_time = 0

    def update(self, side, price, volume):
        """
        Updates a price level. Volume 0 means remove the level.
        """
        if side == 'BID':
            if volume == 0:
                if price in self.bids:
                    del self.bids[price]
            else:
                self.bids[price] = volume
        elif side == 'ASK':
            if volume == 0:
                if price in self.asks:
                    del self.asks[price]
            else:
                self.asks[price] = volume

    def get_snapshot(self):
        """
        Returns a sorted snapshot of the top N levels.
        Used as input for the NexusTransformer.
        """
        # Sort Bids: High to Low
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:self.depth]
        # Sort Asks: Low to High
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:self.depth]
        
        return {
            "bids": sorted_bids,
            "asks": sorted_asks,
            "spread": sorted_asks[0][0] - sorted_bids[0][0] if sorted_bids and sorted_asks else 0,
            "imbalance": self.calculate_imbalance(sorted_bids, sorted_asks)
        }

    def calculate_imbalance(self, bids, asks):
        """
        Calculates Order Flow Imbalance.
        Positive = Bullish Pressure (More Bids).
        Negative = Bearish Pressure (More Asks).
        """
        total_bid_vol = sum(vol for price, vol in bids)
        total_ask_vol = sum(vol for price, vol in asks)
        
        if total_bid_vol + total_ask_vol == 0:
            return 0
            
        return (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

    def to_tensor(self):
        """
        Converts the book to a numpy array for the Neural Network.
        Shape: (2 * depth, 2) -> [Price, Volume] for top N bids and asks.
        """
        snapshot = self.get_snapshot()
        data = []
        
        # Normalize prices relative to mid-price to make it stationarity-invariant
        mid_price = (snapshot['bids'][0][0] + snapshot['asks'][0][0]) / 2 if snapshot['bids'] and snapshot['asks'] else 0
        
        if mid_price == 0:
            return np.zeros((self.depth * 2, 2))

        # Add Bids
        for i in range(self.depth):
            if i < len(snapshot['bids']):
                price, vol = snapshot['bids'][i]
                data.append([(price - mid_price) / mid_price, vol]) # Normalized Price, Raw Volume
            else:
                data.append([0, 0])
                
        # Add Asks
        for i in range(self.depth):
            if i < len(snapshot['asks']):
                price, vol = snapshot['asks'][i]
                data.append([(price - mid_price) / mid_price, vol])
            else:
                data.append([0, 0])
                
        return np.array(data, dtype=np.float32)
