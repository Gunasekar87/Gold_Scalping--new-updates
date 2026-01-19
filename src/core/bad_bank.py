import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger("BadBank")

class BadBank:
    """
    The Bad Bank (Asset Management Company).
    
    Purpose:
    1. Holds "Toxic Assets" (Frozen Buckets) that were locked by the Valkyrie Protocol.
    2. Collects "Tithes" (Taxes) from healthy, profitable trades.
    3. Uses Tithes to slowly pay off (close) the toxic debt, chunk by chunk.
    """
    
    def __init__(self):
        self.toxic_assets: Dict[str, Dict] = {} # bucket_id -> {positions, frozen_time, initial_debt}
        self.tithe_balance: float = 0.0
        self.total_tithe_collected: float = 0.0
        self.total_debt_repaid: float = 0.0
        
    def register_toxic_asset(self, bucket_id: str, positions: list):
        """
        Take ownership of a Frozen Bucket.
        """
        if bucket_id in self.toxic_assets:
            logger.warning(f"[BAD BANK] Asset {bucket_id} is already registered.")
            return

        # Calculate total debt (Net Negative PnL)
        # Note: We track volume and floating PnL
        total_volume = sum(p.volume for p in positions)
        
        self.toxic_assets[bucket_id] = {
            'bucket_id': bucket_id,
            'positions': positions, # Storing object references or dicts? Ideally snapshot.
            'frozen_time': time.time(),
            'initial_volume': total_volume,
            'status': 'FROZEN'
        }
        
        logger.critical(f"🏦 [BAD BANK] ACQUIRED TOXIC ASSET: {bucket_id} | Vol: {total_volume:.2f} lots")
        
    def deposit_tithe(self, amount: float, source_symbol: str) -> float:
        """
        Deposit a portion of profits into the Bad Bank.
        Returns the accepted amount.
        """
        if amount <= 0:
            return 0.0
            
        self.tithe_balance += amount
        self.total_tithe_collected += amount
        
        logger.info(f"💰 [BAD BANK] DEPOSIT RECEIVED: ${amount:.2f} from {source_symbol} | Balance: ${self.tithe_balance:.2f}")
        
        # Trigger an attempt to use this cash to reduce debt
        self.attempt_debt_reduction()
        
        return amount

    def attempt_debt_reduction(self):
        """
        Check if we have enough cash to close a micro-chunk of debt.
        
        Logic (Simplified Phase 3):
        1. Find the smallest position in toxic assets.
        2. Calculate cost to close 0.01 lots of it.
        3. If Balance > Cost:
           - Close 0.01 lots.
           - Deduct from Balance.
        """
        if not self.toxic_assets:
            return
            
        # Placeholder for complex partial closure logic.
        # Real partial closures require Broker interaction, which BadBank doesn't have direct access to yet.
        # For now, we just log that we are accumulating power.
        
        logger.info(f"🏦 [BAD BANK] Holding ${self.tithe_balance:.2f} for future debt repayment. Toxic Assets: {len(self.toxic_assets)}")

    def get_stats(self) -> Dict:
        return {
            'assets_count': len(self.toxic_assets),
            'tithe_pool': self.tithe_balance,
            'total_collected': self.total_tithe_collected,
            'total_debt_repaid': self.total_debt_repaid
        }

    def get_status(self) -> str:
        return (f"Bad Bank Status:\n"
                f"  Assets: {len(self.toxic_assets)}\n"
                f"  Cash Ratio: ${self.tithe_balance:.2f} available\n"
                f"  Lifetime Collection: ${self.total_tithe_collected:.2f}")
