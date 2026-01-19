"""
Async Database Operations - High-performance async database layer.

This module provides async database operations for:
- SQLite (using aiosqlite)
- TimescaleDB/PostgreSQL (using asyncpg)
- Connection pooling and non-blocking I/O
- Automatic batching and background processing

Author: AETHER Development Team
License: MIT
Version: 1.0.0
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

logger = logging.getLogger("AsyncDatabase")


@dataclass
class TickData:
    """Represents tick data for async recording."""
    symbol: str
    bid: float
    ask: float
    timestamp: float
    flags: int = 0


@dataclass
class CandleData:
    """Represents candle data for async recording."""
    symbol: str
    timeframe: str
    open_price: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float


@dataclass
class TradeData:
    """Represents trade data for async recording."""
    ticket: int
    symbol: str
    trade_type: str
    volume: float
    open_price: float
    close_price: Optional[float]
    profit: Optional[float]
    open_time: float
    close_time: Optional[float]
    strategy_reason: str


class AsyncDatabaseManager(ABC):
    """Abstract base class for async database managers."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    async def initialize_schema(self) -> None:
        """Create necessary tables and schema."""
        pass

    @abstractmethod
    async def record_tick(self, tick: TickData) -> None:
        """Record a single tick asynchronously."""
        pass

    @abstractmethod
    async def record_ticks_batch(self, ticks: List[TickData]) -> None:
        """Record multiple ticks in batch."""
        pass

    @abstractmethod
    async def record_candle(self, candle: CandleData) -> None:
        """Record a single candle asynchronously."""
        pass

    @abstractmethod
    async def record_candles_batch(self, candles: List[CandleData]) -> None:
        """Record multiple candles in batch."""
        pass

    @abstractmethod
    async def record_trade(self, trade: TradeData) -> None:
        """Record a trade asynchronously."""
        pass

    @abstractmethod
    async def update_trade_close(self, ticket: int, close_price: float, profit: float) -> None:
        """Update trade with close information."""
        pass

    @abstractmethod
    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        pass


class AsyncSQLiteManager(AsyncDatabaseManager):
    """
    Async SQLite database manager using aiosqlite.

    Provides high-performance async operations for SQLite with:
    - Non-blocking I/O operations
    - WAL mode for concurrent reads/writes
    - Connection pooling via aiosqlite
    """

    def __init__(self, db_path: str = "data/market_memory.db"):
        if not AIOSQLITE_AVAILABLE:
            raise ImportError("aiosqlite not available. Install with: pip install aiosqlite")

        self.db_path = db_path
        self.connection: Optional[aiosqlite.Connection] = None
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    async def connect(self) -> None:
        """Establish async SQLite connection."""
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            # Enable WAL mode for better concurrency
            await self.connection.execute("PRAGMA journal_mode=WAL;")
            await self.connection.execute("PRAGMA synchronous=NORMAL;")
            await self.connection.execute("PRAGMA cache_size=10000;")
            await self.connection.commit()
            logger.info(f"Connected to async SQLite: {self.db_path}")
        except Exception as e:
            logger.error(f"Async SQLite connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close async SQLite connection."""
        if self.connection:
            await self.connection.close()
            self.connection = None

    async def initialize_schema(self) -> None:
        """Create SQLite tables asynchronously."""
        if not self.connection:
            await self.connect()

        try:
            # Ticks table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    flags INTEGER DEFAULT 0
                )
            """)

            # Candles table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    timestamp REAL,
                    PRIMARY KEY (symbol, timeframe, timestamp)
                )
            """)

            # Trade audit table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS trade_audit (
                    ticket INTEGER PRIMARY KEY,
                    symbol TEXT,
                    type TEXT,
                    volume REAL,
                    open_price REAL,
                    close_price REAL,
                    profit REAL,
                    open_time REAL,
                    close_time REAL,
                    strategy_reason TEXT
                )
            """)

            # Create indexes for performance
            await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, timestamp)")
            await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe ON candles(symbol, timeframe)")
            await self.connection.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_audit(symbol)")

            await self.connection.commit()
            logger.info("Async SQLite schema initialized")

        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise

    async def record_tick(self, tick: TickData) -> None:
        """Record a single tick asynchronously."""
        if not self.connection:
            await self.connect()

        try:
            await self.connection.execute(
                "INSERT INTO ticks (symbol, bid, ask, timestamp, flags) VALUES (?, ?, ?, ?, ?)",
                (tick.symbol, tick.bid, tick.ask, tick.timestamp, tick.flags)
            )
            await self.connection.commit()
        except Exception as e:
            logger.error(f"Tick recording failed: {e}")

    async def record_ticks_batch(self, ticks: List[TickData]) -> None:
        """Record multiple ticks in batch for performance."""
        if not self.connection:
            await self.connect()

        if not ticks:
            return

        try:
            data = [(t.symbol, t.bid, t.ask, t.timestamp, t.flags) for t in ticks]
            await self.connection.executemany(
                "INSERT INTO ticks (symbol, bid, ask, timestamp, flags) VALUES (?, ?, ?, ?, ?)",
                data
            )
            await self.connection.commit()
        except Exception as e:
            logger.error(f"Batch tick recording failed: {e}")

    async def record_candle(self, candle: CandleData) -> None:
        """Record a single candle asynchronously."""
        if not self.connection:
            await self.connect()

        try:
            await self.connection.execute(
                "INSERT OR REPLACE INTO candles (symbol, timeframe, open, high, low, close, volume, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (candle.symbol, candle.timeframe, candle.open_price, candle.high, candle.low, candle.close, candle.volume, candle.timestamp)
            )
            await self.connection.commit()
        except Exception as e:
            logger.error(f"Candle recording failed: {e}")

    async def record_candles_batch(self, candles: List[CandleData]) -> None:
        """Record multiple candles in batch."""
        if not self.connection:
            await self.connect()

        if not candles:
            return

        try:
            data = [(c.symbol, c.timeframe, c.open_price, c.high, c.low, c.close, c.volume, c.timestamp) for c in candles]
            await self.connection.executemany(
                "INSERT OR REPLACE INTO candles (symbol, timeframe, open, high, low, close, volume, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                data
            )
            await self.connection.commit()
        except Exception as e:
            logger.error(f"Batch candle recording failed: {e}")

    async def record_trade(self, trade: TradeData) -> None:
        """Record a trade asynchronously."""
        if not self.connection:
            await self.connect()

        try:
            await self.connection.execute(
                "INSERT OR REPLACE INTO trade_audit (ticket, symbol, type, volume, open_price, close_price, profit, open_time, close_time, strategy_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trade.ticket, trade.symbol, trade.trade_type, trade.volume, trade.open_price,
                 trade.close_price, trade.profit, trade.open_time, trade.close_time, trade.strategy_reason)
            )
            await self.connection.commit()
        except Exception as e:
            logger.error(f"Trade recording failed: {e}")

    async def update_trade_close(self, ticket: int, close_price: float, profit: float) -> None:
        """Update trade with close information."""
        if not self.connection:
            await self.connect()

        try:
            await self.connection.execute(
                "UPDATE trade_audit SET close_price = ?, profit = ?, close_time = ? WHERE ticket = ?",
                (close_price, profit, time.time(), ticket)
            )
            await self.connection.commit()
        except Exception as e:
            logger.error(f"Trade close update failed: {e}")

    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics from SQLite."""
        if not self.connection:
            await self.connect()

        stats = {
            "status": "ACTIVE",
            "pnl_daily": 0.0,
            "active_strategies": [],
            "cpu_usage": 0,
            "db_type": "AsyncSQLite"
        }

        try:
            # Calculate Daily PnL
            start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            cursor = await self.connection.execute(
                "SELECT SUM(profit) FROM trade_audit WHERE close_time >= ? AND profit IS NOT NULL",
                (start_of_day,)
            )
            result = await cursor.fetchone()
            stats["pnl_daily"] = round(result[0] if result and result[0] else 0.0, 2)

            # Get Active Strategies
            cursor = await self.connection.execute(
                "SELECT DISTINCT strategy_reason FROM trade_audit WHERE close_time IS NULL AND strategy_reason IS NOT NULL"
            )
            rows = await cursor.fetchall()
            stats["active_strategies"] = [r[0] for r in rows if r[0]]

        except Exception as e:
            logger.error(f"Stats fetch failed: {e}")

        return stats


class AsyncTimescaleManager(AsyncDatabaseManager):
    """
    Async TimescaleDB/PostgreSQL manager using asyncpg.

    Provides high-performance async operations for TimescaleDB with:
    - Connection pooling
    - Hypertable optimizations
    - Batch operations
    - Non-blocking I/O
    """

    def __init__(self, host: str = "localhost", port: int = 6543, user: str = "gravity_user",
                 password: str = "gravity_password", dbname: str = "gravity_market_memory"):
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg not available. Install with: pip install asyncpg")

        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": dbname
        }
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish asyncpg connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                **self.connection_params,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info(f"Connected to async TimescaleDB: {self.connection_params['host']}")
        except Exception as e:
            logger.error(f"Async TimescaleDB connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close asyncpg connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def initialize_schema(self) -> None:
        """Create TimescaleDB tables and hypertables asynchronously."""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            try:
                # Ticks hypertable
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS ticks (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        bid DOUBLE PRECISION NOT NULL,
                        ask DOUBLE PRECISION NOT NULL,
                        flags INTEGER DEFAULT 0
                    );
                """)

                # Convert to hypertable
                try:
                    await conn.execute("SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);")
                except asyncpg.exceptions.UndefinedFunctionError:
                    logger.warning("TimescaleDB extension not found. Using standard PostgreSQL table for 'ticks'.")
                except Exception as e:
                    logger.warning(f"Could not convert 'ticks' to hypertable: {e}")

                # Candles hypertable
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS candles (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        open DOUBLE PRECISION,
                        high DOUBLE PRECISION,
                        low DOUBLE PRECISION,
                        close DOUBLE PRECISION,
                        volume DOUBLE PRECISION,
                        PRIMARY KEY (time, symbol, timeframe)
                    );
                """)

                # Convert to hypertable
                try:
                    await conn.execute("SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);")
                except asyncpg.exceptions.UndefinedFunctionError:
                    logger.warning("TimescaleDB extension not found. Using standard PostgreSQL table for 'candles'.")
                except Exception as e:
                    logger.warning(f"Could not convert 'candles' to hypertable: {e}")

                # Trade audit table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trade_audit (
                        ticket BIGINT PRIMARY KEY,
                        symbol TEXT,
                        type TEXT,
                        volume DOUBLE PRECISION,
                        open_price DOUBLE PRECISION,
                        close_price DOUBLE PRECISION,
                        profit DOUBLE PRECISION,
                        open_time TIMESTAMPTZ,
                        close_time TIMESTAMPTZ,
                        strategy_reason TEXT
                    );
                """)

                # Create indexes
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, time DESC);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe ON candles(symbol, timeframe, time DESC);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trade_audit(symbol);")

                logger.info("Async TimescaleDB schema initialized")

            except Exception as e:
                logger.error(f"Schema initialization failed: {e}")
                raise

    async def record_tick(self, tick: TickData) -> None:
        """Record a single tick asynchronously."""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            try:
                await conn.execute(
                    "INSERT INTO ticks (time, symbol, bid, ask, flags) VALUES ($1, $2, $3, $4, $5)",
                    datetime.fromtimestamp(tick.timestamp), tick.symbol, tick.bid, tick.ask, tick.flags
                )
            except Exception as e:
                logger.error(f"Tick recording failed: {e}")

    async def record_ticks_batch(self, ticks: List[TickData]) -> None:
        """Record multiple ticks in batch."""
        if not self.pool:
            await self.connect()

        if not ticks:
            return

        async with self.pool.acquire() as conn:
            try:
                data = [(datetime.fromtimestamp(t.timestamp), t.symbol, t.bid, t.ask, t.flags) for t in ticks]
                await conn.executemany(
                    "INSERT INTO ticks (time, symbol, bid, ask, flags) VALUES ($1, $2, $3, $4, $5)",
                    data
                )
            except Exception as e:
                logger.error(f"Batch tick recording failed: {e}")

    async def record_candle(self, candle: CandleData) -> None:
        """Record a single candle asynchronously."""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO candles (time, symbol, timeframe, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, volume = EXCLUDED.volume
                    """,
                    datetime.fromtimestamp(candle.timestamp), candle.symbol, candle.timeframe,
                    candle.open_price, candle.high, candle.low, candle.close, candle.volume
                )
            except Exception as e:
                logger.error(f"Candle recording failed: {e}")

    async def record_candles_batch(self, candles: List[CandleData]) -> None:
        """Record multiple candles in batch."""
        if not self.pool:
            await self.connect()

        if not candles:
            return

        async with self.pool.acquire() as conn:
            try:
                data = [(datetime.fromtimestamp(c.timestamp), c.symbol, c.timeframe,
                        c.open_price, c.high, c.low, c.close, c.volume) for c in candles]
                await conn.executemany(
                    """
                    INSERT INTO candles (time, symbol, timeframe, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                        close = EXCLUDED.close, volume = EXCLUDED.volume
                    """,
                    data
                )
            except Exception as e:
                logger.error(f"Batch candle recording failed: {e}")

    async def record_trade(self, trade: TradeData) -> None:
        """Record a trade asynchronously."""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO trade_audit (ticket, symbol, type, volume, open_price, close_price, profit, open_time, close_time, strategy_reason)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (ticket) DO UPDATE SET
                        close_price = EXCLUDED.close_price, profit = EXCLUDED.profit, close_time = EXCLUDED.close_time
                    """,
                    trade.ticket, trade.symbol, trade.trade_type, trade.volume, trade.open_price,
                    trade.close_price, trade.profit, datetime.fromtimestamp(trade.open_time) if trade.open_time else None,
                    datetime.fromtimestamp(trade.close_time) if trade.close_time else None, trade.strategy_reason
                )
            except Exception as e:
                logger.error(f"Trade recording failed: {e}")

    async def update_trade_close(self, ticket: int, close_price: float, profit: float) -> None:
        """Update trade with close information."""
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            try:
                await conn.execute(
                    "UPDATE trade_audit SET close_price = $1, profit = $2, close_time = $3 WHERE ticket = $4",
                    close_price, profit, datetime.now(), ticket
                )
            except Exception as e:
                logger.error(f"Trade close update failed: {e}")

    async def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics from TimescaleDB."""
        if not self.pool:
            await self.connect()

        stats = {
            "status": "ACTIVE",
            "pnl_daily": 0.0,
            "active_strategies": [],
            "cpu_usage": 0,
            "db_type": "AsyncTimescaleDB"
        }

        async with self.pool.acquire() as conn:
            try:
                # Calculate Daily PnL
                result = await conn.fetchval(
                    "SELECT SUM(profit) FROM trade_audit WHERE close_time >= NOW() - INTERVAL '1 day' AND profit IS NOT NULL"
                )
                stats["pnl_daily"] = round(result if result else 0.0, 2)

                # Get Active Strategies
                rows = await conn.fetch(
                    "SELECT DISTINCT strategy_reason FROM trade_audit WHERE close_time IS NULL AND strategy_reason IS NOT NULL"
                )
                stats["active_strategies"] = [r['strategy_reason'] for r in rows if r['strategy_reason']]

            except Exception as e:
                logger.error(f"Stats fetch failed: {e}")

        return stats


async def get_async_database_manager(config: Optional[Dict[str, Any]] = None) -> AsyncDatabaseManager:
    """
    Factory function to return the appropriate async database manager.

    Args:
        config: Database configuration dictionary

    Returns:
        AsyncDatabaseManager instance
    """
    if config and config.get("use_timescale", False):
        # Check if DB credentials exist
        if not os.getenv("DB_HOST"):
            logger.info("No DB_HOST found in environment. Falling back to async SQLite.")
        elif ASYNCPG_AVAILABLE:
            try:
                logger.info("Initializing async TimescaleDB manager...")
                return AsyncTimescaleManager(
                    host=os.getenv("DB_HOST", config.get("timescale_host", "localhost")),
                    port=int(os.getenv("DB_PORT", config.get("timescale_port", 6543))),
                    user=os.getenv("DB_USER", config.get("timescale_user", "gravity_user")),
                    password=os.getenv("DB_PASSWORD", config.get("timescale_password", "gravity_password")),
                    dbname=os.getenv("DB_NAME", config.get("timescale_db", "gravity_market_memory"))
                )
            except Exception as e:
                logger.warning(f"Async TimescaleDB initialization failed: {e}. Falling back to async SQLite.")
        else:
            logger.warning("Async TimescaleDB requested but asyncpg not found. Falling back to async SQLite.")

    logger.info("Initializing async SQLite manager...")
    return AsyncSQLiteManager()


class AsyncDatabaseQueue:
    """
    Async queue for batching database operations.

    This class provides:
    - Background batch processing
    - Automatic flushing based on time/size thresholds
    - Non-blocking queue operations
    """

    def __init__(self, db_manager: AsyncDatabaseManager, batch_size: int = 100, flush_interval: float = 1.0):
        self.db_manager = db_manager
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.ticks_queue: List[TickData] = []
        self.candles_queue: List[CandleData] = []
        self.trades_queue: List[TradeData] = []

        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background processing task."""
        self.running = True
        self.task = asyncio.create_task(self._process_queue())
        logger.info("Async database queue started")

    async def stop(self) -> None:
        """Stop the background processing and flush remaining items."""
        self.running = False
        if self.task:
            await self.task

        # Final flush
        await self._flush_all()
        logger.info("Async database queue stopped")

    async def add_tick(self, tick: TickData) -> None:
        """Add tick to queue for batch processing."""
        self.ticks_queue.append(tick)
        if len(self.ticks_queue) >= self.batch_size:
            await self._flush_ticks()

    async def add_candle(self, candle: CandleData) -> None:
        """Add candle to queue for batch processing."""
        self.candles_queue.append(candle)
        if len(self.candles_queue) >= self.batch_size:
            await self._flush_candles()

    async def add_trade(self, trade: TradeData) -> None:
        """Add trade to queue (immediate processing for trades)."""
        await self.db_manager.record_trade(trade)

    async def _process_queue(self) -> None:
        """Background task to periodically flush queues."""
        while self.running:
            await asyncio.sleep(self.flush_interval)
            await self._flush_all()

    async def _flush_ticks(self) -> None:
        """Flush accumulated ticks."""
        if self.ticks_queue:
            try:
                await self.db_manager.record_ticks_batch(self.ticks_queue)
                self.ticks_queue.clear()
            except Exception as e:
                logger.error(f"Failed to flush ticks: {e}")

    async def _flush_candles(self) -> None:
        """Flush accumulated candles."""
        if self.candles_queue:
            try:
                await self.db_manager.record_candles_batch(self.candles_queue)
                self.candles_queue.clear()
            except Exception as e:
                logger.error(f"Failed to flush candles: {e}")

    async def _flush_all(self) -> None:
        """Flush all queues."""
        await self._flush_ticks()
        await self._flush_candles()