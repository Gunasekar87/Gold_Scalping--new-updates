import psycopg2
from psycopg2 import pool
import logging
import time
import os
from datetime import datetime

logger = logging.getLogger("TimescaleMemory")

class TimescaleMemory:
    """
    The Institutional Data Backbone (TimescaleDB Edition).
    Stores high-frequency Tick Data and Candle Data in a structured TimescaleDB database.
    
    Advantages over SQLite:
    1. High Throughput (Millions of inserts/sec)
    2. Time-series optimizations (Hypertables)
    3. Compression (90%+ storage savings)
    4. Parallel queries for massive backtesting
    """
    def __init__(self, host="localhost", port=6543, user="gravity_user", password="gravity_password", dbname="gravity_market_memory"):
        self.connection_params = {
            "host": host,
            "port": int(port),
            "user": user,
            "password": password,
            "dbname": dbname
        }
        self.pool = None
        self.connect()
        self.initialize_schema()

    def connect(self):
        try:
            # Create a connection pool for thread safety and performance
            self.pool = psycopg2.pool.SimpleConnectionPool(
                1, 20, **self.connection_params
            )
            logger.info(f"Connected to TimescaleDB: {self.connection_params['host']}")
        except Exception as e:
            logger.error(f"TimescaleDB Connection Failed: {e}")
            raise e # Re-raise to allow fallback logic in factory

    def get_connection(self):
        return self.pool.getconn()

    def release_connection(self, conn):
        self.pool.putconn(conn)

    def initialize_schema(self):
        """
        Creates the necessary tables and converts them to Hypertables.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # 1. Ticks Table (High Frequency)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ticks (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        bid DOUBLE PRECISION NOT NULL,
                        ask DOUBLE PRECISION NOT NULL,
                        flags INTEGER
                    );
                """)
                # Convert to Hypertable (partition by time)
                try:
                    cur.execute("SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);")
                except psycopg2.errors.UndefinedFunction:
                    conn.rollback()
                    logger.warning("TimescaleDB extension not found. Using standard PostgreSQL table for 'ticks'.")
                except Exception as e:
                    conn.rollback()
                    logger.warning(f"Could not convert 'ticks' to hypertable: {e}")
                
                # 2. Candles Table (Aggregated)
                cur.execute("""
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
                # Convert to Hypertable
                try:
                    cur.execute("SELECT create_hypertable('candles', 'time', if_not_exists => TRUE);")
                except psycopg2.errors.UndefinedFunction:
                    conn.rollback()
                    logger.warning("TimescaleDB extension not found. Using standard PostgreSQL table for 'candles'.")
                except Exception as e:
                    conn.rollback()
                    logger.warning(f"Could not convert 'candles' to hypertable: {e}")
                
                # 3. Trade Log (Audit Trail) - Standard Postgres Table
                cur.execute("DROP TABLE IF EXISTS trade_audit CASCADE;")
                cur.execute("""
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

                # 4. Order Book Snapshots (Level 2 Data)
                # Stores the full depth state as JSONB for flexibility
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                        time TIMESTAMPTZ NOT NULL,
                        symbol TEXT NOT NULL,
                        bids JSONB,
                        asks JSONB,
                        imbalance DOUBLE PRECISION
                    );
                """)
                # Convert to Hypertable
                cur.execute("SELECT create_hypertable('orderbook_snapshots', 'time', if_not_exists => TRUE);")

                conn.commit()
                logger.info("TimescaleDB Schema Initialized & Hypertables Created")
        except Exception as e:
            logger.error(f"Schema Initialization Failed: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def record_tick(self, symbol, bid, ask, flags=0):
        """
        Saves a single tick.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ticks (time, symbol, bid, ask, flags) VALUES (NOW(), %s, %s, %s, %s)",
                    (symbol, bid, ask, flags)
                )
            conn.commit()
        except Exception as e:
            logger.error(f"Tick Record Error: {e}")
        finally:
            self.release_connection(conn)

    def record_candle(self, symbol, timeframe, o, h, l, c, v):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO candles (time, symbol, timeframe, open, high, low, close, volume)
                    VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (time, symbol, timeframe) DO UPDATE 
                    SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume
                """, (symbol, timeframe, o, h, l, c, v))
            conn.commit()
        except Exception as e:
            logger.error(f"Candle Record Error: {e}")
        finally:
            self.release_connection(conn)

    def record_trade(self, ticket, symbol, trade_type, volume, open_price, reason):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trade_audit (ticket, symbol, type, volume, open_price, open_time, strategy_reason)
                    VALUES (%s, %s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT (ticket) DO NOTHING
                """, (ticket, symbol, trade_type, volume, open_price, reason))
            conn.commit()
        except Exception as e:
            logger.error(f"Trade Record Error: {e}")
        finally:
            self.release_connection(conn)

    def get_dashboard_stats(self):
        """
        Fetches real-time stats for the dashboard from TimescaleDB.
        """
        stats = {
            "status": "ACTIVE",
            "pnl_daily": 0.0,
            "active_strategies": [],
            "cpu_usage": 0,
            "db_type": "TimescaleDB"
        }
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Calculate Daily PnL
                cur.execute("SELECT SUM(profit) FROM trade_audit WHERE close_time >= NOW() - INTERVAL '1 day'")
                pnl = cur.fetchone()[0]
                stats["pnl_daily"] = round(pnl if pnl else 0.0, 2)

                # Get Active Strategies
                cur.execute("SELECT DISTINCT strategy_reason FROM trade_audit WHERE close_time IS NULL")
                rows = cur.fetchall()
                stats["active_strategies"] = [r[0] for r in rows]
        except Exception as e:
            logger.error(f"Stats Fetch Error: {e}")
        finally:
            self.release_connection(conn)
        
        return stats

    def record_order_book(self, symbol, bids, asks):
        """
        Stores a snapshot of the order book.
        bids/asks should be lists of [price, volume] or dictionaries, stored as JSON.
        """
        import json
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO order_book_snapshots (time, symbol, bids, asks) VALUES (NOW(), %s, %s, %s)",
                    (symbol, json.dumps(bids), json.dumps(asks))
                )
            conn.commit()
        except Exception as e:
            logger.error(f"Order Book Record Error: {e}")
        finally:
            self.release_connection(conn)

    def get_dashboard_stats(self):
        """
        Fetches real-time stats for the dashboard from TimescaleDB.
        """
        stats = {
            "status": "ACTIVE",
            "pnl_daily": 0.0,
            "active_strategies": [],
            "cpu_usage": 0,
            "db_type": "TimescaleDB"
        }
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Calculate Daily PnL
                cur.execute("SELECT SUM(profit) FROM trade_audit WHERE close_time >= NOW() - INTERVAL '1 day'")
                pnl = cur.fetchone()[0]
                stats["pnl_daily"] = round(pnl if pnl else 0.0, 2)

                # Get Active Strategies
                cur.execute("SELECT DISTINCT strategy_reason FROM trade_audit WHERE close_time IS NULL")
                rows = cur.fetchall()
                stats["active_strategies"] = [r[0] for r in rows]
        except Exception as e:
            logger.error(f"Stats Fetch Error: {e}")
        finally:
            self.release_connection(conn)
        
        return stats
