"""
Hedge Coordinator - Prevents duplicate/conflicting hedge placements

This module provides centralized coordination for hedge decisions to prevent
the "hedges from all corners" problem where multiple systems trigger hedges
simultaneously on the same bucket.
"""

import threading
import time
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger("HedgeCoordinator")


class HedgeCoordinator:
    """
    Central coordinator to prevent duplicate hedge placements.
    
    Provides locking mechanism to ensure only ONE hedge decision
    per bucket at a time, preventing loops and conflicts.
    """
    
    def __init__(self):
        self.active_hedges: Dict[str, Dict] = {}  # bucket_id -> hedge_info
        self.lock = threading.Lock()
        self.min_hedge_interval = 30.0  # Minimum seconds between hedges on same bucket
        logger.info(f"HedgeCoordinator initialized (min interval: {self.min_hedge_interval}s)")
    
    def can_hedge_bucket(self, bucket_id: str, min_interval: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if a bucket can be hedged (not hedged recently).
        
        Args:
            bucket_id: Unique identifier for the position bucket
            min_interval: Optional override for minimum interval (seconds)
            
        Returns:
            Tuple of (can_hedge: bool, reason: str)
        """
        interval = min_interval if min_interval is not None else self.min_hedge_interval
        
        with self.lock:
            if bucket_id not in self.active_hedges:
                return True, "No recent hedge on this bucket"
            
            hedge_info = self.active_hedges[bucket_id]
            last_hedge_time = hedge_info['time']
            time_since = time.time() - last_hedge_time
            
            if time_since < interval:
                return False, f"Hedged {time_since:.1f}s ago (min {interval}s required)"
            
            return True, f"Last hedge {time_since:.1f}s ago (OK to hedge)"
    
    def record_hedge(self, bucket_id: str, action: str, lots: float, price: float):
        """
        Record that a hedge was just placed on a bucket.
        
        Args:
            bucket_id: Bucket identifier
            action: Hedge action ('BUY' or 'SELL')
            lots: Hedge lot size
            price: Hedge execution price
        """
        with self.lock:
            self.active_hedges[bucket_id] = {
                'time': time.time(),
                'action': action,
                'lots': lots,
                'price': price
            }
            logger.info(f"[COORDINATOR] Recorded hedge: {bucket_id} -> {action} {lots} @ {price}")
    
    def clear_bucket(self, bucket_id: str):
        """
        Clear hedge history for a bucket (when bucket is closed).
        
        Args:
            bucket_id: Bucket identifier to clear
        """
        with self.lock:
            if bucket_id in self.active_hedges:
                del self.active_hedges[bucket_id]
                logger.info(f"[COORDINATOR] Cleared bucket: {bucket_id}")
    
    def get_bucket_status(self, bucket_id: str) -> Optional[Dict]:
        """
        Get the current hedge status for a bucket.
        
        Args:
            bucket_id: Bucket identifier
            
        Returns:
            Hedge info dict or None if no recent hedge
        """
        with self.lock:
            return self.active_hedges.get(bucket_id)
    
    def acquire_hedge_lock(self, bucket_id: str, timeout: float = 5.0) -> bool:
        """
        Try to acquire exclusive hedge lock for a bucket.
        
        This is a more advanced locking mechanism for critical sections.
        
        Args:
            bucket_id: Bucket identifier
            timeout: Maximum time to wait for lock (seconds)
            
        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            can_hedge, reason = self.can_hedge_bucket(bucket_id)
            if can_hedge:
                # Mark as "in progress" to prevent others
                with self.lock:
                    self.active_hedges[bucket_id] = {
                        'time': time.time(),
                        'action': 'PENDING',
                        'lots': 0.0,
                        'price': 0.0
                    }
                return True
            
            time.sleep(0.1)  # Brief wait before retry
        
        return False
    
    def release_hedge_lock(self, bucket_id: str):
        """
        Release hedge lock if it was acquired but hedge wasn't executed.
        
        Args:
            bucket_id: Bucket identifier
        """
        with self.lock:
            if bucket_id in self.active_hedges:
                if self.active_hedges[bucket_id]['action'] == 'PENDING':
                    del self.active_hedges[bucket_id]
                    logger.info(f"[COORDINATOR] Released lock: {bucket_id}")


# Singleton instance
_coordinator_instance = None

def get_hedge_coordinator() -> HedgeCoordinator:
    """Get or create the singleton HedgeCoordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = HedgeCoordinator()
    return _coordinator_instance
