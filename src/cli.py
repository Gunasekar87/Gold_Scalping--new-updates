"""
AETHER Trading System - Command Line Interface

This module handles the application startup, environment configuration,
and process optimization before launching the main trading bot.
"""

import sys
import os
import io
import logging
import asyncio
import gc
import codecs
from dotenv import load_dotenv

# Try to import psutil for process priority management
try:
    import psutil
except ImportError:
    psutil = None

# Import the main bot application
from src.main_bot import main as bot_main

def setup_environment():
    """Configure the runtime environment for optimal performance."""
    
    # 1. Force UTF-8 encoding for Windows Console
    if sys.platform == 'win32':
        # Method 1: Reconfigure stdout/stderr (Python 3.7+)
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            # Method 2: Wrap buffer if reconfigure fails
            try:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
            except Exception:
                pass

    # 2. Load Environment Variables
    # Look for secrets.env in config folder relative to CWD or package
    env_path = os.path.join(os.getcwd(), "config", "secrets.env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        # Fallback to package-relative path if installed
        pkg_env_path = os.path.join(os.path.dirname(__file__), "..", "config", "secrets.env")
        if os.path.exists(pkg_env_path):
            load_dotenv(pkg_env_path)

    # 3. Pre-load Heavy Libraries (Torch) - Clean startup
    from src.constants import SYSTEM_VERSION
    print(f"\\n🚀 AETHER v{SYSTEM_VERSION} | Initializing...", flush=True)
    try:
        import torch
    except ImportError:
        pass

    # Optional runtime trace to prove which code is actually executing.
    # Enable with: AETHER_RUNTIME_TRACE=1
    if str(os.getenv("AETHER_RUNTIME_TRACE", "0")).strip().lower() in ("1", "true", "yes", "on"):
        try:
            import src
            from src.ai_core import oracle as _oracle_mod
            from src.ai_core import contrastive_fusion as _fusion_mod
            print(f">>> [TRACE] src package: {getattr(src, '__file__', None)}", flush=True)
            print(f">>> [TRACE] oracle.py: {getattr(_oracle_mod, '__file__', None)}", flush=True)
            print(f">>> [TRACE] contrastive_fusion.py: {getattr(_fusion_mod, '__file__', None)}", flush=True)
        except Exception as e:
            print(f">>> [TRACE] Runtime trace failed: {e}", flush=True)

def optimize_process():
    """Apply OS-level process optimizations - Silent mode."""
    
    # 1. Set High Process Priority - Silent
    if psutil:
        try:
            p = psutil.Process(os.getpid())
            if sys.platform == 'win32':
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                p.nice(-10) # High priority on Linux
        except Exception:
            pass
    
    # 2. Tune Garbage Collector - Silent
    gc.set_threshold(5000, 10, 10)

def main():
    """CLI Entry Point."""
    try:
        setup_environment()
        optimize_process()
        
        # Launch the Async Event Loop
        if sys.platform == 'win32':
            # Set Windows Selector Event Loop Policy if needed
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        asyncio.run(bot_main())
        
    except KeyboardInterrupt:
        print("\\n>>> [SYSTEM] Shutdown requested by user.")
    except Exception as e:
        print(f"\\n>>> [FATAL] System Crash: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
