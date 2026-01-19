import os
import logging
import asyncio
from datetime import datetime
from supabase import create_client, Client

logger = logging.getLogger("SupabaseAdapter")

class SupabaseAdapter:
    """
    Handles real-time data pushing to Supabase.
    Designed to be non-blocking and fail-safe.
    """
    def __init__(self, url=None, key=None):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        self.client: Client = None
        self.enabled = False
        
        if self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                self.enabled = True
                logger.info("[OK] Supabase Adapter Initialized")
            except Exception as e:
                logger.error(f"[ERROR] Supabase Connection Failed: {e}")
        else:
            logger.warning("[WARN] Supabase Credentials Missing. Cloud Sync Disabled.")

    async def push_tick(self, tick_data):
        """
        Pushes a new tick to the 'market_ticks' table.
        tick_data: dict with symbol, bid, ask, time
        """
        if not self.enabled: return
        
        try:
            # Run in executor to avoid blocking the event loop
            await asyncio.to_thread(
                lambda: self.client.table("market_ticks").insert(tick_data).execute()
            )
        except Exception as e:
            logger.debug(f"Failed to push tick: {e}")

    async def push_candle(self, candle_data):
        """
        Pushes a completed candle to 'market_candles'.
        """
        if not self.enabled: return
        
        try:
            await asyncio.to_thread(
                lambda: self.client.table("market_candles").insert(candle_data).execute()
            )
        except Exception as e:
            logger.error(f"Failed to push candle: {e}")

    async def push_trade_event(self, trade_data):
        """
        Pushes trade events to 'trade_journal'.
        """
        if not self.enabled: return
        
        try:
            # Map Bot Data to Dashboard Schema
            payload = {
                "ticket": int(trade_data.get("ticket")),  # Supabase uses bigint for ticket
                "symbol": trade_data.get("symbol"),
                "type": trade_data.get("type"),
                "volume": trade_data.get("volume"),
                "open_price": trade_data.get("price"),
                "is_open": True,  # Boolean flag instead of status string
                "reason": trade_data.get("reason")
            }
            
            await asyncio.to_thread(
                lambda: self.client.table("trade_journal").insert(payload).execute()
            )
        except Exception as e:
            logger.error(f"Failed to push trade event: {e}")

    async def update_trade(self, ticket, close_data):
        """
        Updates a closed trade in 'trade_journal'.
        """
        if not self.enabled: return
        
        try:
            payload = {
                "is_open": False,  # Mark as closed
                "close_price": close_data.get("close_price"),
                "profit": close_data.get("profit"),
                "close_time": datetime.utcnow().isoformat()
            }
            
            await asyncio.to_thread(
                lambda: self.client.table("trade_journal").update(payload).eq("ticket", int(ticket)).execute()
            )
        except Exception as e:
            logger.error(f"Failed to update trade: {e}")

    async def push_log(self, log_entry):
        """
        Pushes system logs to 'system_logs'.
        """
        if not self.enabled: return
        
        try:
            await asyncio.to_thread(
                lambda: self.client.table("system_logs").insert(log_entry).execute()
            )
        except Exception as e:
            logger.debug(f"Failed to push log: {e}")

    async def update_council_state(self, state_data):
        """
        Updates the latest state of the AI Council in 'council_state'.
        This is an 'upsert' operation (single row for current state).
        """
        if not self.enabled: return
        
        try:
            # We use a fixed ID=1 for the singleton state
            state_data['id'] = 1
            state_data['updated_at'] = datetime.utcnow().isoformat()
            
            await asyncio.to_thread(
                lambda: self.client.table("council_state").upsert(state_data).execute()
            )
        except Exception as e:
            logger.error(f"Failed to update council state: {e}")

    async def push_system_status(self, status_data):
        """
        Pushes system status updates to 'system_status'.
        """
        if not self.enabled: return
        
        try:
            # We use a fixed ID=1 for the singleton state
            status_data['id'] = 1
            status_data['updated_at'] = datetime.utcnow().isoformat()
            
            await asyncio.to_thread(
                lambda: self.client.table("system_status").upsert(status_data).execute()
            )
        except Exception as e:
            # Suppress error for missing table to avoid log spam
            pass
