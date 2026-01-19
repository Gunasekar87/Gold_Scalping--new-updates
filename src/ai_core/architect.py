import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger("Architect")

class Architect:
    """
    v5.5.0: THE ARCHITECT (Spatial Awareness)
    Maps the 'Walls' of the market (Support & Resistance) using Macro Timeframes (H1).
    Prevents the bot from buying into ceilings or selling into floors.
    """
    def __init__(self, mt5_adapter):
        self.mt5 = mt5_adapter

    def get_market_structure(self, symbol: str) -> Optional[Dict]:
        """
        Scans H1 structure to find the nearest Support and Resistance zones.
        """
        # 1. Get Macro View (H1 Candles - Last 5 days / 120 candles)
        # Use the adapter's get_market_data method which returns list of dicts
        candles = self.mt5.get_market_data(symbol, timeframe="H1", limit=120)
        
        if not candles:
            logger.warning(f"[ARCHITECT] Failed to fetch H1 data for {symbol}")
            return None

        df = pd.DataFrame(candles)
        if df.empty:
            return None
            
        current_price = df['close'].iloc[-1]

        # 2. Identify Swing Highs and Lows (Fractals)
        # A simple but robust way: Local Max/Min over 5-candle window
        df['min'] = df['low'].rolling(window=5, center=True).min()
        df['max'] = df['high'].rolling(window=5, center=True).max()

        # 3. Cluster Levels to find "Zones"
        # We look for candles where the low equals the local min (Support)
        supports = df[df['low'] == df['min']]['low'].values
        # We look for candles where the high equals the local max (Resistance)
        resistances = df[df['high'] == df['max']]['high'].values

        # 4. Find Nearest Walls
        nearest_support = -1.0
        nearest_resistance = 999999.0

        # Find closest support below price
        below_price = supports[supports < current_price]
        if len(below_price) > 0:
            # We take the highest of the supports below us (the floor we are standing on)
            nearest_support = np.max(below_price)

        # Find closest resistance above price
        above_price = resistances[resistances > current_price]
        if len(above_price) > 0:
            # We take the lowest of the resistances above us (the ceiling we are hitting)
            nearest_resistance = np.min(above_price)

        # 5. Calculate "Room to Move"
        dist_to_supp = current_price - nearest_support
        dist_to_res = nearest_resistance - current_price
        
        # 6. Determine Status
        # If we are within a threshold to a wall, we are "BLOCKED"
        # Gold (XAUUSD) typically needs ~$2.00 room. Forex pairs need ~10-20 pips.
        threshold = 2.0 if "XAU" in symbol or "GOLD" in symbol else 0.0020 # Approx 20 pips for forex

        status = 'CLEAR'
        if dist_to_res < threshold:
            status = 'BLOCKED_UP'
        elif dist_to_supp < threshold:
            status = 'BLOCKED_DOWN'

        return {
            'price': current_price,
            'support': nearest_support,
            'resistance': nearest_resistance,
            'room_up': dist_to_res,
            'room_down': dist_to_supp,
            'status': status
        }
