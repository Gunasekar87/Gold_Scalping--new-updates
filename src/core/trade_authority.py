"""
Trade Authority - The "Supreme Court" of the AETHER System.
Part of the AETHER Intelligence Stack (Layer 0: Constitution).

This module acts as the Central Gatekeeper. It enforces immutable laws (The Constitution)
that NO other module (including God Mode) can bypass.

Responsibility:
1. Prevent "Unlimited Trades" by enforcing Global & Symbol Caps.
2. Adapt limits based on Market Volatility (Adaptive Constitution).
3. Veto any trade that violates risk parameters.

Author: AETHER Development Team
License: MIT
Version: 1.0.0
"""

import logging
from typing import Dict, Tuple, Optional
from src.utils.data_normalization import normalize_positions

logger = logging.getLogger("TradeAuthority")

class TradeAuthority:
    """
    The Supreme Court.
    All trade requests must be approved here before execution.
    """
    
    def __init__(self, config=None):
        # --- THE CONSTITUTION (Immutable Laws) ---
        # --- THE CONSTITUTION (Immutable Laws) ---
        self.max_global_positions_base = 10 # [TREASURY] GLOBAL HARD CAP
        self.max_hedges_per_symbol = 10 # Default Base (User requested increase)
        
        # Current State
        self.current_global_cap = self.max_global_positions_base
        self.current_hedge_cap = self.max_hedges_per_symbol
        self.volatility_state = "NORMAL"
        
        logger.info("[SUPREME COURT] Trade Authority Initialized. The Law is active.")

    def update_constitution(self, atr_value: float, equity: float):
        """
        Adaptive Adjustment of Laws based on Market Conditions.
        Called periodically (e.g., every minute).
        
        Args:
            atr_value: Current market volatility.
            equity: Account equity.
        """
        # 1. Volatility Adjustment
        # If ATR is high (> 2.0 on Gold), we clamp down.
        # If ATR is low (< 1.0 on Gold), we allow more breathing room.
        
        # Note: ATR thresholds dependent on timeframe/asset. Assuming Gold M1 approx.
        # 1. Volatility Adjustment (Dynamic Layering)
        # If ATR is high (> 2.0 on Gold), we clamp down to 4 hedges.
        # If ATR is low (< 1.0 on Gold), we allow 6 hedges (Deep Grid).
        
        # Note: ATR thresholds dependent on timeframe/asset. Assuming Gold M1 approx.
        if atr_value > 2.0: 
            self.current_hedge_cap = 8  # High volatility: reduce from 10 to 8
            self.volatility_state = "HIGH_VOLATILITY (Defense Mode)"
            # Global cap remains 10 (or lower if needed)
        elif atr_value < 1.0:
            self.current_hedge_cap = 12  # Low volatility: allow expansion to 12
            self.volatility_state = "LOW_VOLATILITY (Expansion Mode)"
        else:
            self.current_hedge_cap = 10  # Normal: use base 10
            self.volatility_state = "NORMAL"
            
        # Global Cap is always HARD LIMIT 10 (as per user request)
        self.current_global_cap = self.max_global_positions_base
            
        # 2. Equity Adjustment (Scale up for large accounts, but cap at 15)
        # We REMOVED the "20" limit that caused the issue. Hard Cap is now 15.
        if equity > 50000:
            self.current_global_cap = min(self.current_global_cap + 2, 15)

    def check_constitution(self, broker, symbol: str, volume: float, action: str) -> Tuple[bool, str]:
        """
        The Judgment.
        Decides if a trade request is Constitutional.
        
        Args:
            broker: Broker instance to fetch current state.
            symbol: Symbol to trade.
            volume: Requested volume.
            action: "OPEN" or "CLOSE" (Closes are always allowed).
            
        Returns:
            (Approved: bool, Reason: str)
        """
        # AMENDMENT 0: Closing trades is always Constitutional
        if action == "CLOSE" or action == "CLOSE_ALL":
            return True, "Closing trades is always allowed"

        # Fetch Current State
        all_positions = broker.get_positions()
        if all_positions is None:
            return False, "VETOED: Cannot read current positions (System Blind)"
            
        # Optimize: Normalize once using utility
        all_positions = normalize_positions(all_positions)
        total_positions = len(all_positions)
        symbol_positions = len([p for p in all_positions if p.get('symbol') == symbol])
        
        # AMENDMENT 1: Global Cap (The "Unlimited Trade" Fix)
        if total_positions >= self.current_global_cap:
            msg = f"VETOED: Global Cap Reached ({total_positions}/{self.current_global_cap}). State: {self.volatility_state}"
            
            # Throttle Veto Log
            import time
            current_time = time.time()
            if not hasattr(self, '_last_veto_log'): self._last_veto_log = {}
            last_time = self._last_veto_log.get('global_cap', 0)
            
            if current_time - last_time > 60:
                logger.warning(f"[SUPREME COURT] {msg}")
                self._last_veto_log['global_cap'] = current_time
                
            return False, msg
            
        # AMENDMENT 2: Symbol Cap (The "Dynamic Layers")
        # We allow up to 'current_hedge_cap' (4 to 6).
        if symbol_positions >= self.current_hedge_cap:
            msg = f"VETOED: Symbol Cap Reached for {symbol} ({symbol_positions}/{self.current_hedge_cap}). State: {self.volatility_state}"

            # Throttle Veto Log
            import time
            current_time = time.time()
            if not hasattr(self, '_last_veto_log'): self._last_veto_log = {}
            last_time = self._last_veto_log.get(f'symbol_cap_{symbol}', 0)
            
            if current_time - last_time > 60:
                logger.warning(f"[SUPREME COURT] {msg}")
                self._last_veto_log[f'symbol_cap_{symbol}'] = current_time

            return False, msg
            
        # AMENDMENT 3: Volume/Exposure Check
        # Prevent absurdly large trades (e.g. fat finger or bug)
        # Max single trade size = 5.0 lots (Safety)
        if volume > 5.0:
             msg = f"VETOED: Excessive Volume ({volume} lots > 5.0 allowed)"
             logger.warning(f"[SUPREME COURT] {msg}")
             return False, msg

        return True, "APPROVED"
