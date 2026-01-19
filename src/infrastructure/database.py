import sqlite3
import time
import logging
import os
from datetime import datetime
# ...existing code...
try:
    from .timescale_adapter import TimescaleMemory
    TIMESCALE_AVAILABLE = True
except ImportError:
    TIMESCALE_AVAILABLE = False

logger = logging.getLogger("MarketMemory")

def get_database_manager(config=None):
    """
    Factory function to return the appropriate database manager.
    """
    if config and config.get("use_timescale", False):
        # Check if DB credentials exist in Env Vars
        if not os.getenv("DB_HOST"):
            logger.info("No DB_HOST found in environment. Skipping TimescaleDB and using SQLite.")
        elif TIMESCALE_AVAILABLE:
            try:
                logger.info("Initializing TimescaleDB Memory...")
                return TimescaleMemory(
                    host=os.getenv("DB_HOST", config.get("timescale_host", "localhost")),
                    port=int(os.getenv("DB_PORT", config.get("timescale_port", 6543))),
                    user=os.getenv("DB_USER", config.get("timescale_user", "gravity_user")),
                    password=os.getenv("DB_PASSWORD", config.get("timescale_password", "gravity_password")),
                    dbname=os.getenv("DB_NAME", config.get("timescale_db", "gravity_market_memory"))
                )
            except Exception as e:
                logger.warning(f"TimescaleDB initialization failed: {e}. Falling back to SQLite.")
        else:
            logger.warning("TimescaleDB requested but dependencies (psycopg2) not found. Falling back to SQLite.")
    
    logger.info("Initializing SQLite Memory...")
    return MarketMemory()

class MarketMemory:
    """
    The Institutional Data Backbone.
    Stores high-frequency Tick Data and Candle Data in a structured SQL database.
    This allows for:
    1. Accurate Replay Simulations (Backtesting)
    2. Deep Learning Training on massive datasets
    3. Audit trails of every market movement
    """
    def __init__(self, db_path="data/market_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = None
        self.cursor = None
        self.connect()
        self.initialize_schema()

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            # Enable WAL mode for higher write concurrency (Institutional speed)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            logger.info(f"Connected to Market Memory: {self.db_path}")
        except Exception as e:
            logger.error(f"Database Connection Failed: {e}")

    def initialize_schema(self):
        """
        Creates the necessary tables if they don't exist.
        """
        try:
            # 1. Ticks Table (High Frequency)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    flags INTEGER
                )
            """)
            
            # 2. Candles Table (Aggregated)
            self.conn.execute("""
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
            
            # 3. Trade Log (Audit Trail)
            self.conn.execute("""
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
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Schema Initialization Failed: {e}")

    def record_tick(self, symbol, bid, ask, flags=0):
        """
        Saves a single tick. 
        In a real institutional setup, we would batch these.
        """
        try:
            self.conn.execute(
                "INSERT INTO ticks (symbol, bid, ask, timestamp, flags) VALUES (?, ?, ?, ?, ?)",
                (symbol, bid, ask, time.time(), flags)
            )
            # We commit periodically in the main loop, or use WAL auto-checkpoint
        except Exception as e:
            logger.error(f"Tick Record Error: {e}")

    def record_trade(self, ticket, symbol, order_type, volume, open_price, reason):
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO trade_audit (ticket, symbol, type, volume, open_price, open_time, strategy_reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ticket, symbol, order_type, volume, open_price, time.time(), reason)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Trade Record Error: {e}")

    def record_candle(self, symbol, timeframe, open_price, high, low, close, volume, timestamp):
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO candles (symbol, timeframe, open, high, low, close, volume, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, timeframe, open_price, high, low, close, volume, timestamp)
            )
        except Exception as e:
            logger.error(f"Candle Record Error: {e}")

    def bulk_insert_candles(self, data_list):
        """
        Bulk insert candles for speed.
        data_list: list of tuples (symbol, timeframe, open, high, low, close, volume, timestamp)
        """
        try:
            self.conn.executemany(
                "INSERT OR REPLACE INTO candles (symbol, timeframe, open, high, low, close, volume, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                data_list
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Bulk Insert Error: {e}")

    def close(self):
        if self.conn:
            self.conn.close()

    def get_dashboard_stats(self):
        """
        Fetches real-time stats for the dashboard.
        """
        stats = {
            "status": "ACTIVE",
            "pnl_daily": 0.0,
            "active_strategies": [],
            "cpu_usage": 0,
            "db_type": "SQLite"
        }
        try:
            # Calculate Daily PnL (Sum of profit from trades closed today)
            start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            cursor = self.conn.execute("SELECT SUM(profit) FROM trade_audit WHERE close_time >= ?", (start_of_day,))
            pnl = cursor.fetchone()[0]
            stats["pnl_daily"] = round(pnl if pnl else 0.0, 2)

            # Get Active Strategies (Distinct reasons from open trades)
            cursor = self.conn.execute("SELECT DISTINCT strategy_reason FROM trade_audit WHERE close_time IS NULL")
            rows = cursor.fetchall()
            stats["active_strategies"] = [r[0] for r in rows]

        except Exception as e:
            logger.error(f"Stats Fetch Error: {e}")
        
        return stats
