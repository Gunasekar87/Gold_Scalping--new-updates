"""
AETHER Trading Bot - Refactored Main Entry Point

This module serves as the main entry point for the AETHER trading system,
now refactored into modular components for better maintainability and testing.

The bot orchestrates:
- Market data processing
- AI-driven trading decisions
- Risk management and position handling
- Configuration validation and security

Author: AETHER Development Team
License: MIT
Version: 5.5.8
"""

import asyncio
import logging
import signal
import time
import os
import yaml
import json
import random
import threading
from collections import deque
from typing import Dict, Any, Optional
from pathlib import Path

# Import modular components
from .market_data import MarketDataManager
from .position_manager import PositionManager
from .risk_manager import RiskManager, ZoneConfig
from .trading_engine import TradingEngine, TradingConfig
from .config_validator import ConfigValidator
from .utils.trading_logger import TradingLogger, DecisionTracker

# Import async database
from .infrastructure.async_database import get_async_database_manager

# Import NEW Intelligence Layers
from .ai_core.oracle import Oracle
from .utils.news_filter import NewsFilter

# Import existing components (to be gradually migrated)

from .ai_core.ppo_guardian import PPOGuardian
from .ai_core.iron_shield import IronShield
from .ai_core.strategist import Strategist
from .infrastructure.database import get_database_manager
from .infrastructure.supabase_adapter import SupabaseAdapter
from .bridge.broker_factory import BrokerFactory

logger = logging.getLogger("AetherBot")

# Setup UI Logger for console output
ui_logger = logging.getLogger("AETHER_UI")
ui_logger.setLevel(logging.INFO)
if not ui_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    ui_logger.addHandler(console_handler)
    ui_logger.propagate = False

# Configure Trader Dashboard (Event-Driven Logging)
trader_logger = logging.getLogger("TRADER")
trader_logger.setLevel(logging.INFO)
if not trader_logger.handlers:
    trader_handler = logging.StreamHandler()
    trader_handler.setFormatter(logging.Formatter('%(message)s'))
    trader_logger.addHandler(trader_handler)
    trader_logger.propagate = False

# Suppress technical/internal logs to reduce noise
# Only show warnings and errors from these loggers
logging.getLogger("FRESHNESS").setLevel(logging.WARNING)
logging.getLogger("POS_MGMT").setLevel(logging.WARNING)
logging.getLogger("BUCKET").setLevel(logging.WARNING)
logging.getLogger("TP_CHECK").setLevel(logging.WARNING)
logging.getLogger("ZONE_CHECK").setLevel(logging.WARNING)
logging.getLogger("SYNC").setLevel(logging.WARNING)
logging.getLogger("PLAN").setLevel(logging.WARNING)
logging.getLogger("CLOSE").setLevel(logging.WARNING)


class AetherBot:
    """
    Main AETHER trading bot class with modular architecture.

    This class orchestrates all trading activities using specialized modules:
    - MarketDataManager: Data fetching and market state tracking
    - PositionManager: Position tracking and bucket management
    - RiskManager: Zone recovery and hedging logic
    - TradingEngine: Core trading decision and execution
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AETHER trading bot.

        Args:
            config: Complete configuration dictionary
        """
        self.config = config
        self.running = False

        # Core components
        self.broker = None
        self.market_data = None
        self.position_manager = None
        self.risk_manager = None
        self.trading_engine = None

        # Legacy components (to be migrated)
        self.nexus = None
        self.ppo_guardian = None
        self.council = None
        self.shield = None
        self.strategist = None
        self.memory_db = None
        self.supabase_adapter = None

        # Control flags
        self.shutdown_requested = False
        
        # Dashboard timer
        self.last_console_update = 0.0
        self.console_update_interval = 10.0  # Seconds
        
        # Decision tracking for rolling logs (reduces noise)
        self.decision_tracker = DecisionTracker()

        logger.info("AETHER Bot initialized")

    def validate_configuration(self) -> bool:
        """
        Validate configuration with security checks.

        Returns:
            True if configuration is valid and secure
        """
        validator = ConfigValidator()

        # Validate main config
        is_valid, errors, warnings = validator.validate_all()

        # Log warnings
        for warning in warnings:
            logger.warning(f"Config Warning: {warning}")

        # Log errors
        if errors:
            for error in errors:
                logger.error(f"Config Error: {error}")
            return False

        # Security validation
        is_secure, security_issues = validator.validate_secure_config_loading(self.config)
        if not is_secure:
            for issue in security_issues:
                logger.error(f"Security Issue: {issue}")
            return False

        logger.info("[OK] Configuration validation passed")
        return True

    async def initialize_components(self) -> bool:
        """
        Initialize all bot components.

        Returns:
            True if all components initialized successfully
        """
        try:
            # Initialize broker
            logger.info("1. Connecting to Broker...")
            broker_type = self.config.get('trading', {}).get('broker_type', 'MT5')
            credentials = self._extract_credentials()
            self.broker = BrokerFactory.get_broker(broker_type, credentials)

            if not self.broker.connect():
                logger.error(f"Failed to connect to {broker_type}")
                return False

            # Initialize modular components
            logger.info("2. Initializing Market Data & Position Manager...")
            timeframe = self.config.get('trading', {}).get('timeframe', 'M1')
            self.market_data = MarketDataManager(self.broker, timeframe, self.config)

            # v5.5.0: Pass broker to PositionManager for Architect
            self.position_manager = PositionManager(mt5_adapter=self.broker)
            
            # CRITICAL: Sync position state with broker to remove stale positions
            logger.info("3. Synchronizing Positions...")
            broker_positions = self.broker.get_positions()
            if broker_positions is not None:
                self.position_manager.cleanup_stale_positions(broker_positions)
                # Update bucket stats to mark empty buckets as closed
                self.position_manager._update_bucket_stats()
                self.position_manager.update_positions(broker_positions)
                logger.info(f"[SYNC] Position manager synchronized with broker: {len(broker_positions)} active positions")
            else:
                logger.warning("[SYNC] Failed to fetch initial positions from broker")

            zone_config = ZoneConfig(
                zone_pips=self.config.get('risk', {}).get('zone_recovery', {}).get('zone_pips', 25),
                tp_pips=self.config.get('risk', {}).get('zone_recovery', {}).get('tp_pips', 25),
                max_hedges=self.config.get('risk', {}).get('zone_recovery', {}).get('max_layers', 5)
            )
            self.risk_manager = RiskManager(zone_config)
            
            # [FIX] Initialize IronShield early and attach to RiskManager if needed
            # Or better, just initialize it here so we can pass it to TradingEngine
            self.shield = IronShield(
                initial_lot=self.config['risk']['initial_lot'],
                zone_pips=self.config['risk']['zone_recovery']['zone_pips'],
                tp_pips=self.config['risk']['zone_recovery']['tp_pips']
            )
            # Attach shield to risk_manager if it expects it (based on error log)
            self.risk_manager.shield = self.shield

            trading_config = TradingConfig(
                symbol=self.config.get('trading', {}).get('symbol', 'EURUSD'),
                initial_lot=self.config.get('risk', {}).get('initial_lot', 0.01),
                global_trade_cooldown=5.0, # Reduced for Continuous Scalping
                timeframe=self.config.get('trading', {}).get('timeframe', 'M1')
            )

            # Initialize async database
            logger.info("4. Connecting to Database...")
            db_manager = await get_async_database_manager(self.config.get('database'))

            logger.info("5. Initializing AI Core (This is the heavy part)...")
            
            # Initialize AI components needed for TradingEngine
            print(">>> [INIT] Loading PPO Guardian (Reinforcement Learning)...", flush=True)
            self.ppo_guardian = PPOGuardian()
            print(">>> [INIT] PPO Guardian Online.", flush=True)
            
            # Initialize Global Brain (Layer 9)
            print(">>> [INIT] Loading Global Brain (Macro Analysis)...", flush=True)
            from .ai_core.global_brain import GlobalBrain
            self.global_brain = GlobalBrain(self.market_data)
            print(">>> [INIT] Global Brain Online.", flush=True)

            # v5.5.0: Initialize TickPressureAnalyzer shared instance
            from .ai_core.tick_pressure import TickPressureAnalyzer
            self.tick_analyzer = TickPressureAnalyzer()

            # Initialize Oracle (Layer 4)
            print(">>> [INIT] Loading Oracle (Price Prediction)...", flush=True)
            # v5.5.0: Pass broker and tick_analyzer to Oracle
            # INTEGRATION FIX: Oracle will receive model_monitor from trading_engine after initialization
            self.oracle = Oracle(mt5_adapter=self.broker, tick_analyzer=self.tick_analyzer, global_brain=self.global_brain)
            print(">>> [INIT] Oracle Online.", flush=True)

            print(">>> [INIT] Starting Trading Engine...", flush=True)
            self.trading_engine = TradingEngine(trading_config, self.broker, self.market_data,
                                              self.position_manager, self.risk_manager, db_manager, self.ppo_guardian, self.global_brain,
                                              tick_analyzer=self.tick_analyzer)
            print(">>> [INIT] Trading Engine Ready.", flush=True)
            
            # INTEGRATION FIX: Pass model_monitor to Oracle after trading_engine is initialized
            if hasattr(self.trading_engine, 'model_monitor') and self.trading_engine.model_monitor:
                self.oracle.model_monitor = self.trading_engine.model_monitor
                logger.info("[INTEGRATION] Model monitor connected to Oracle")

            # Initialize database components
            print(">>> [INIT] Initializing Database...", flush=True)
            await self.trading_engine.initialize_database()
            print(">>> [INIT] Database Initialized.", flush=True)

            # Initialize legacy components (temporary)
            logger.info("6. Finalizing Setup...")
            print(">>> [INIT] Initializing Legacy Components...", flush=True)
            self._initialize_legacy_components()
            print(">>> [INIT] Legacy Components Initialized.", flush=True)

            logger.info("[OK] All components initialized successfully")
            return True

        except Exception as e:
            import sys
            sys.stderr.write(f"INIT ERROR: {e}\n")
            logger.error(f"Component initialization failed: {e}")
            return False

    def _extract_credentials(self) -> Dict[str, str]:
        """Extract broker credentials from config."""
        credentials = {}

        # Try to get from environment first (secure)
        credentials.update({
            'mt5_login': os.getenv('MT5_LOGIN'),
            'mt5_password': os.getenv('MT5_PASSWORD'),
            'mt5_server': os.getenv('MT5_SERVER'),
            'api_key': os.getenv('API_KEY'),
            'secret_key': os.getenv('SECRET_KEY')
        })

        # Fallback to config (less secure)
        if not credentials['mt5_login']:
            credentials.update({
                'mt5_login': self.config.get('mt5_login'),
                'mt5_password': self.config.get('mt5_password'),
                'mt5_server': self.config.get('mt5_server'),
                'api_key': self.config.get('api_key'),
                'secret_key': self.config.get('secret_key')
            })

        return credentials

    def _initialize_legacy_components(self) -> None:
        """Initialize legacy components (to be migrated to modular system)."""
        logger.info("Initializing legacy AI components (Nexus Brain, Iron Shield)...")

        # AI Components

        # self.ppo_guardian is already initialized in initialize_components

        # Risk and Strategy
        # self.shield is already initialized in initialize_components
        self.strategist = Strategist()

        # Infrastructure
        self.memory_db = get_database_manager(self.config.get('database'))
        self.supabase_adapter = SupabaseAdapter()

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Shutdown signal received ({signum}). Initiating graceful shutdown...")
            self.shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def run(self) -> None:
        """
        Main bot execution loop.
        """
        if not self.validate_configuration():
            logger.error("Configuration validation failed. Exiting.")
            return

        if not await self.initialize_components():
            logger.error("Component initialization failed. Exiting.")
            return

        self._setup_signal_handlers()
        self.running = True

        logger.info("[INFO] AETHER Trading System Online")
        print(">>> [SYSTEM] Bot is Running. Waiting for market data...", flush=True)
        logger.info("=" * 50)

        try:
            print(">>> [DEBUG] Entering Main Loop...", flush=True)
            last_heartbeat = time.time()
            while not self.shutdown_requested:
                # Heartbeat every 10 seconds (Log only, no print)
                if time.time() - last_heartbeat > 10.0:
                    # print(f">>> [SYSTEM] Heartbeat - Bot is alive. Time: {time.strftime('%H:%M:%S')}", flush=True)
                    last_heartbeat = time.time()

                await self._run_trading_cycle()

                # Adaptive sleep: Balanced latency for active positions
                has_positions = len(self.position_manager.active_positions) > 0
                if has_positions:
                    # Active positions mode: 100ms polling (10Hz) - fast enough for scalping without overloading broker
                    # Provides <100ms response time while reducing race conditions and API load
                    await asyncio.sleep(0.1) 
                else:
                    # Normal adaptive sleep when waiting for signals (0.1s - 1.0s)
                    sleep_time = self.market_data.get_adaptive_sleep_time()
                    await asyncio.sleep(min(sleep_time, 0.5)) # Cap at 0.5s to catch signals faster

        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
        finally:
            await self._shutdown()
            logger.info("AETHER System shutdown complete")

    async def _run_trading_cycle(self) -> None:
        """
        Execute one complete trading cycle.
        """
        try:
            logger.debug("[CYCLE] TRADING CYCLE START")

            # Trading engine handles position management AND new entries in single call
            await self._execute_trading_strategy()

            # Update dashboard and logs
            await self._update_dashboard()
            
            # Safe Garbage Collection
            # Only clean memory if we are FLAT (no open trades) to avoid lag spikes during trading
            # [OPTIMIZATION] Removed frequent GC calls. Python's cyclic GC is sufficient.
            # Only force collect if necessary during idle times (implemented elsewhere if needed).
            pass

            logger.debug("[SUCCESS] TRADING CYCLE COMPLETED")

        except Exception as e:
            print(f">>> [ERROR] Cycle Error: {e}", flush=True)
            logger.error(f"[FAIL] TRADING CYCLE ERROR: {e}")

    async def _execute_trading_strategy(self) -> None:
        """Execute trading strategy for new entries."""
        # Use the new trading engine
        await self.trading_engine.run_trading_cycle(
            self.strategist, self.shield, self.ppo_guardian, self.nexus, self.oracle
        )

    async def _update_dashboard(self) -> None:
        """Update dashboard with current status."""
        try:
            # Get dashboard stats
            stats = self.trading_engine.get_session_stats()

            # Update Console Dashboard (Periodic)
            current_time = time.time()
            if current_time - self.last_console_update >= self.console_update_interval:
                self._print_console_status()
                self.last_console_update = current_time

            # Update via Supabase if enabled
            if self.supabase_adapter and self.supabase_adapter.enabled:
                await self.supabase_adapter.push_system_status({
                    "status": "ACTIVE" if self.running else "STOPPED",
                    "trades_opened": stats.get("trades_opened", 0),
                    "trades_closed": stats.get("trades_closed", 0),
                    "total_profit": stats.get("total_profit", 0.0),
                    "timestamp": int(time.time() * 1000)
                })

        except Exception as e:
            logger.error(f"Dashboard update failed: {e}")

    def _print_console_status(self) -> None:
        """Print a clean status dashboard to the console."""
        # DISABLED: User requested to disable rolling logs
        return

    async def _shutdown(self) -> None:
        """Perform graceful shutdown."""
        logger.info("Performing graceful shutdown...")

        self.running = False

        # -------------------------------
        # End-of-session learning (daily)
        # -------------------------------
        # Goal: maximize profit while penalizing drawdown, without changing live weights intraday.
        try:
            enable_tuner = str(os.getenv("AETHER_ENABLE_SESSION_TUNING", "1")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if enable_tuner and getattr(self, "oracle", None) is not None and getattr(self.oracle, "tuner", None) is not None:
                stats = self.trading_engine.get_session_stats() if self.trading_engine else {}
                total_profit = float((stats or {}).get("total_profit", 0.0) or 0.0)
                max_dd_pct = float((stats or {}).get("max_drawdown_pct", 0.0) or 0.0)

                acct = self.broker.get_account_info() if self.broker else {}
                balance = float((acct or {}).get("balance", 0.0) or 0.0)
                # Penalty is scaled by starting/ending balance; if unknown, fall back to 0.
                if balance <= 0:
                    balance = 0.0

                try:
                    dd_lambda = float(os.getenv("AETHER_TUNER_DD_LAMBDA", "0.75"))
                except Exception:
                    dd_lambda = 0.75

                try:
                    cost_per_trade = float(os.getenv("AETHER_TUNER_COST_PER_TRADE", "0.0"))
                except Exception:
                    cost_per_trade = 0.0

                trades_closed = float((stats or {}).get("trades_closed", 0) or 0)

                # Risk-adjusted reward: profit minus drawdown penalty (scaled by balance) minus per-trade friction.
                dd_penalty = dd_lambda * (balance * max_dd_pct)
                reward = total_profit - dd_penalty - (cost_per_trade * trades_closed)

                params = getattr(self.oracle, "last_tuner_params", None)
                if isinstance(params, dict) and params:
                    self.oracle.tuner.report_result(params, pnl=float(reward))
                    logger.info(
                        f"[SESSION_TUNING] Reported reward=${reward:.2f} pnl=${total_profit:.2f} dd_pct={max_dd_pct*100:.2f}% dd_penalty=${dd_penalty:.2f} trades_closed={int(trades_closed)}"
                    )
                else:
                    logger.info("[SESSION_TUNING] Skipped: no tuner params recorded this session")
        except Exception as e:
            logger.warning(f"[SESSION_TUNING] Failed: {e}")

        # Optional: offline PPO evolution (explicitly gated)
        try:
            enable_ppo_evolve = str(os.getenv("AETHER_ENABLE_PPO_EVOLVE", "0")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if enable_ppo_evolve and self.ppo_guardian is not None:
                # Avoid evolving while trades are still open (incomplete episodes)
                open_positions = 0
                try:
                    pos = self.broker.get_positions() if self.broker else []
                    open_positions = len(pos) if pos else 0
                except Exception:
                    open_positions = 0

                if open_positions == 0:
                    ok = bool(self.ppo_guardian.evolve())
                    logger.info(f"[PPO_EVOLVE] Completed ok={ok} experiences={self.ppo_guardian.get_experience_count()}")
                else:
                    logger.info(f"[PPO_EVOLVE] Skipped (open_positions={open_positions})")
        except Exception as e:
            logger.warning(f"[PPO_EVOLVE] Failed: {e}")

        # Shutdown trading engine database
        if self.trading_engine:
            await self.trading_engine.shutdown_database()

        # Close database connections
        if self.memory_db:
            self.memory_db.close()

        # Disconnect broker
        if self.broker:
            self.broker.disconnect()

        logger.info("Shutdown complete")


def load_configuration() -> Dict[str, Any]:
    """
    Load and validate configuration from files.

    Returns:
        Configuration dictionary
    """
    config = {}

    # Load settings.yaml
    settings_file = Path("config/settings.yaml")
    if settings_file.exists():
        with open(settings_file, 'r') as f:
            config.update(yaml.safe_load(f) or {})

    # Load model_config.json
    model_config_file = Path("config/model_config.json")
    if model_config_file.exists():
        with open(model_config_file, 'r') as f:
            config['model_config'] = json.load(f)

    # Load user_config.json
    user_config_file = Path("config/user_config.json")
    if user_config_file.exists():
        with open(user_config_file, 'r') as f:
            config['user_config'] = json.load(f)

    return config


async def main():
    """
    Main entry point for the AETHER trading system.
    """
    # Load configuration
    config = load_configuration()

    # Create and run bot
    bot = AetherBot(config)
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
