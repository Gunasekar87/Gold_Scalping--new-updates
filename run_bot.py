#!/usr/bin/env python3
"""
AETHER Trading Bot - Enhanced Launcher Script
Ensures latest code is always loaded by clearing cache before starting.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path

def clear_python_cache():
    """Remove all __pycache__ directories and .pyc files to ensure fresh code."""
    try:
        project_root = Path(__file__).parent
        cache_cleared = 0
        
        # Remove __pycache__ directories
        for pycache_dir in project_root.rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir)
                cache_cleared += 1
            except Exception:
                pass  # Silently skip if can't remove
        
        # Remove .pyc files
        for pyc_file in project_root.rglob("*.pyc"):
            try:
                pyc_file.unlink()
                cache_cleared += 1
            except Exception:
                pass
        
        # Remove .pyo files
        for pyo_file in project_root.rglob("*.pyo"):
            try:
                pyo_file.unlink()
                cache_cleared += 1
            except Exception:
                pass
        
        if cache_cleared > 0:
            print(f"🧹 Cleared {cache_cleared} cache items (ensuring fresh code)")
    except Exception as e:
        # Don't fail startup if cache clear fails
        print(f"⚠ Cache clear warning: {e}")

def verify_version():
    """Display bot version on startup."""
    try:
        constants_file = Path(__file__).parent / "src" / "constants.py"
        with open(constants_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'SYSTEM_VERSION' in line and '=' in line:
                    version = line.split('=')[1].strip().strip('"').strip("'")
                    # Extract just the version number
                    if 'SYSTEM_VERSION' in line:
                        version_parts = version.split('#')[0].strip()
                        print(f"🤖 AETHER Bot Version: {version_parts}")
                    break
    except Exception:
        pass  # Silently skip if can't read version

def check_git_status():
    """Check for uncommitted changes (optional)."""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=2
        )
        
        if result.returncode == 0 and result.stdout.strip():
            print("⚠ Warning: Running with uncommitted changes")
    except Exception:
        pass  # Silently skip if Git not available

# Add the current directory to sys.path to ensure src is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Pre-startup checks (fast and non-blocking)
    clear_python_cache()
    verify_version()
    check_git_status()
    
    # Start the bot
    from src.cli import main
    main()
    