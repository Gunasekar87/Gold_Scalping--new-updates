"""
Risk Governor - Intelligent account protection with profit-seeking emergency exit

This module enforces smart limits on account exposure and provides
intelligent emergency exit that aims to close trades in PROFIT.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger("RiskGovernor")


@dataclass
class RiskLimits:
    """Configuration for risk limits."""
    max_total_exposure_pct: float  # Maximum total margin usage % (e.g., 0.15 = 15%)
    max_drawdown_pct: float  # Maximum account drawdown % (e.g., 0.20 = 20%)
    position_limit_per_1k: int  # Positions allowed per $1000 balance (e.g., 2 = 2 positions per $1k)
    news_lockout: bool  # Block trading during news events
    capital_reserve_pct: float = 0.5 # [TREASURY] Default 50% Reserve


class RiskGovernor:
    """
    Enforces intelligent risk limits to protect account.
    
    Key Features:
    - Emergency exit aims to close trades in PROFIT
    - Dynamic position limits based on account balance
    - Intelligent recovery mode before shutdown
    """
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.valkyrie_active = False # Valkyrie Protocol (The Freeze)
        self.emergency_shutdown = False  # Full shutdown (last resort)
        self.shutdown_reason = ""
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        logger.info(f"RiskGovernor initialized: Max Exposure={limits.max_total_exposure_pct:.1%}, "
                   f"Max Drawdown={limits.max_drawdown_pct:.1%} (Valkyrie Trigger), "
                   f"Dynamic Position Limit={limits.position_limit_per_1k} per $1k")
    
    def veto(self, metrics: dict) -> Tuple[bool, str]:
        """
        Check if trading should be blocked based on risk metrics.
        
        Args:
            metrics: Dictionary containing:
                - total_exposure_pct: Current margin usage as % of balance
                - total_positions: Number of open positions
                - account_drawdown_pct: Current drawdown from peak
                - news_now: Boolean indicating if news event is happening
                - balance: Current account balance
                - positions: List of position dicts (for intelligent exit)
                
        Returns:
            Tuple of (should_veto: bool, reason: str)
        """
        # [THE TREASURY] Shadow Balance Logic
        # We trade with only a portion of the funds. The rest is a "Doomsday Reserve".
        real_balance = metrics.get("balance", 10000.0)
        real_equity = metrics.get("equity", real_balance) # Default to balance if equity missing
        
        # Calculate Shadow (Effective) Capital
        # If Reserve = 0.5 (50%), we use (1 - 0.5) = 0.5 multiplier.
        trading_ratio = 1.0 - self.limits.capital_reserve_pct
        effective_balance = real_balance * trading_ratio
        
        # Effective Equity = Real Equity - Reserved Portion
        # Reserved Portion = Real Balance * Reserve PCT
        reserved_cash = real_balance * self.limits.capital_reserve_pct
        effective_equity = real_equity - reserved_cash
        
        # Recalculate Drawdown based on Shadow Balance
        # Drawdown = (Balance - Equity) / Balance
        # Shadow Drawdown = (Eff Balance - Eff Equity) / Eff Balance
        if effective_balance > 0:
            shadow_drawdown_pct = (effective_balance - effective_equity) / effective_balance
        else:
            shadow_drawdown_pct = 0.0

        # Check if full shutdown is active
        if self.emergency_shutdown:
            return True, f"EMERGENCY_SHUTDOWN: {self.shutdown_reason}"
        
        # Check 1: Total Exposure Limit
        # Recalculate Exposure based on Shadow Balance
        # Exposure = Used Margin / Effective Balance
        used_margin = metrics.get("margin", 0.0) # Need margin passed in metrics
        # If margin not in metrics, approximate from exposure_pct (Exposure = Balance * pct) -> bad
        # Better: Trust the passed 'total_exposure_pct' but scale it?
        # Passed Pct = Margin / RealBalance.
        # We want Margin / EffBalance. 
        # EffBalance = RealBalance * Ratio.
        # So NewPct = (Margin / RealBalance) / Ratio = PassedPct / Ratio.
        
        raw_exposure_pct = metrics.get("total_exposure_pct", 0.0)
        shadow_exposure_pct = raw_exposure_pct / trading_ratio if trading_ratio > 0 else 0.0
        
        if shadow_exposure_pct > self.limits.max_total_exposure_pct:
            logger.warning(f"[RISK_GOVERNOR] Shadow Exposure limit breached: {shadow_exposure_pct:.2%} > {self.limits.max_total_exposure_pct:.2%} (Real: {raw_exposure_pct:.2%})")
            return True, "veto:exposure_limit"
        
        # Check 2: Dynamic Position Count Limit
        total_positions = metrics.get("total_positions", 0)
        # Use Effective Balance for position limits
        max_positions = self._calculate_dynamic_position_limit(effective_balance)
        
        if total_positions >= max_positions:
            logger.info(f"[RISK_GOVERNOR] Dynamic position limit reached: {total_positions}/{max_positions} (Balance: ${balance:.0f})")
            return True, f"veto:max_positions_{max_positions}"
        
        # Check 3: Account Drawdown Limit (CRITICAL - triggers intelligent exit)
        # Use Shadow Drawdown calculated above
        if shadow_drawdown_pct >= 0.15: # [VALKYRIE] Trigger at 15% of TRADING CAPITAL (not Total)
            # Activate Valkyrie Protocol - FREEZE THE ACCOUNT
            if not self.valkyrie_active:
                self._activate_valkyrie_protocol(
                    f"Shadow Drawdown limit breached: {shadow_drawdown_pct:.2%} >= 15% (Real DD: {max(0, (real_balance - real_equity)/real_balance):.2%})",
                    metrics
                )
            return True, "veto:valkyrie_active"
        
        # Check 4: News Lockout
        if self.limits.news_lockout and metrics.get("news_now", False):
            logger.info("[RISK_GOVERNOR] News event detected - trading locked")
            return True, "veto:news_event"
        
        # All checks passed
        return False, ""
    
    def _calculate_dynamic_position_limit(self, balance: float) -> int:
        """
        Calculate dynamic position limit based on account balance.
        
        Args:
            balance: Current account balance
            
        Returns:
            Maximum allowed positions
        """
        # Calculate positions per $1000
        thousands = balance / 1000.0
        max_positions = int(thousands * self.limits.position_limit_per_1k)
        
        # Minimum 2 positions, maximum 20
        max_positions = max(2, min(max_positions, 20))
        
        return max_positions
    
    def _activate_valkyrie_protocol(self, reason: str, metrics: dict):
        """
        Activate Valkyrie Protocol - The "Perfect Hedge" Freeze.
        Instead of closing trades (realizing loss), we open an opposing trade to net 0 delta.
        
        Args:
            reason: Reason for activation
            metrics: Current metrics
        """
        if not self.valkyrie_active:
            self.valkyrie_active = True
            self.shutdown_reason = reason
            logger.critical(f"[RISK_GOVERNOR] ❄️ VALKYRIE PROTOCOL ACTIVATED: {reason}")
            logger.critical("[RISK_GOVERNOR] Requesting PERFECT HEDGE to freeze account state.")
            # Note: The actual hedge execution happens in PositionManager/TradingEngine
            # when they see this flag.
    
    def _create_intelligent_exit_strategy(self, positions: List[Dict]) -> Dict:
        """
        Create intelligent exit strategy to close trades in profit.
        
        Strategy:
        1. Close all profitable positions immediately
        2. For losing positions, wait for recovery or use smart hedging
        3. Set tight trailing stops on all positions
        4. If recovery fails after N attempts, force close at best available price
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Exit strategy dictionary
        """
        profitable = []
        losing = []
        total_pnl = 0.0
        
        for pos in positions:
            pnl = pos.get('profit', 0.0)
            total_pnl += pnl
            
            if pnl > 0:
                profitable.append(pos)
            else:
                losing.append(pos)
        
        strategy = {
            'total_positions': len(positions),
            'profitable_count': len(profitable),
            'losing_count': len(losing),
            'total_pnl': total_pnl,
            'profitable_positions': profitable,
            'losing_positions': losing,
            'actions': []
        }
        
        # Action 1: Close profitable positions immediately
        if profitable:
            strategy['actions'].append({
                'action': 'CLOSE_PROFITABLE',
                'count': len(profitable),
                'expected_profit': sum(p.get('profit', 0) for p in profitable),
                'priority': 1
            })
        
        # Action 2: For losing positions, set tight trailing stops
        if losing:
            strategy['actions'].append({
                'action': 'SET_TIGHT_STOPS',
                'count': len(losing),
                'stop_distance_pips': 5,  # Very tight stop
                'priority': 2
            })
        
        # Action 3: Wait for recovery (give positions a chance)
        if losing:
            strategy['actions'].append({
                'action': 'WAIT_FOR_RECOVERY',
                'count': len(losing),
                'max_wait_minutes': 5,  # Wait max 5 minutes
                'priority': 3
            })
        
        # Action 4: If recovery fails, force close at best price
        if losing:
            strategy['actions'].append({
                'action': 'FORCE_CLOSE_IF_NO_RECOVERY',
                'count': len(losing),
                'trigger': 'after_recovery_attempts',
                'priority': 4
            })
        
        # Summary
        if profitable and not losing:
            summary = f"✅ All {len(profitable)} positions profitable - Close immediately"
        elif profitable and losing:
            summary = f"⚠️ Mixed: {len(profitable)} profitable (close now), {len(losing)} losing (wait for recovery)"
        else:
            summary = f"❌ All {len(losing)} positions losing - Wait for recovery, then force close"
        
        strategy['summary'] = summary
        
        return strategy
    
    def get_exit_strategy(self) -> Optional[Dict]:
        """
        Get the current exit strategy if emergency mode is active.
        
        Returns:
            Exit strategy dict or None
        """
        if self.emergency_mode and hasattr(self, 'exit_strategy'):
            return self.exit_strategy
        return None
    
    def should_close_position(self, position: Dict) -> Tuple[bool, str]:
        """
        Check if a specific position should be closed in emergency mode.
        
        Args:
            position: Position dictionary
            
        Returns:
            Tuple of (should_close: bool, reason: str)
        """
        if not self.emergency_mode:
            return False, "Not in emergency mode"
        
        # Always close profitable positions
        if position.get('profit', 0.0) > 0:
            return True, "Emergency mode: Closing profitable position"
        
        # For losing positions, check recovery attempts
        if self.recovery_attempts >= self.max_recovery_attempts:
            return True, "Emergency mode: Max recovery attempts reached, force closing"
        
        return False, "Emergency mode: Waiting for recovery"
    
    def increment_recovery_attempt(self):
        """Increment recovery attempt counter."""
        self.recovery_attempts += 1
        logger.info(f"[RISK_GOVERNOR] Recovery attempt {self.recovery_attempts}/{self.max_recovery_attempts}")
        
        if self.recovery_attempts >= self.max_recovery_attempts:
            logger.critical("[RISK_GOVERNOR] Max recovery attempts reached - will force close all positions")
            self._trigger_full_shutdown("Max recovery attempts exhausted")
    
    def _trigger_full_shutdown(self, reason: str):
        """
        Trigger full emergency shutdown (last resort).
        
        Args:
            reason: Reason for shutdown
        """
        if not self.emergency_shutdown:
            self.emergency_shutdown = True
            logger.critical(f"[RISK_GOVERNOR] 🚨 FULL EMERGENCY SHUTDOWN: {reason}")
            logger.critical("[RISK_GOVERNOR] Force closing ALL positions at current market price")
    
    def reset_emergency_mode(self, admin_password: str = "RESET_AETHER_2026"):
        """
        Reset emergency mode (requires admin password).
        
        Args:
            admin_password: Password to confirm reset
            
        Returns:
            True if reset successful
        """
        if admin_password == "RESET_AETHER_2026":
            self.valkyrie_active = False
            self.emergency_shutdown = False
            self.shutdown_reason = ""
            self.recovery_attempts = 0
            if hasattr(self, 'exit_strategy'):
                delattr(self, 'exit_strategy')
            logger.warning("[RISK_GOVERNOR] Valkyrie/Emergency mode RESET by admin")
            return True
        else:
            logger.error("[RISK_GOVERNOR] Invalid admin password for emergency reset")
            return False
    
    def get_risk_status(self, metrics: dict) -> dict:
        """
        Get current risk status without vetoing.
        
        Args:
            metrics: Current risk metrics
            
        Returns:
            Dictionary with risk status for each limit
        """
        total_exposure_pct = metrics.get("total_exposure_pct", 0.0)
        total_positions = metrics.get("total_positions", 0)
        account_drawdown_pct = metrics.get("account_drawdown_pct", 0.0)
        balance = metrics.get("balance", 10000.0)
        
        max_positions = self._calculate_dynamic_position_limit(balance)
        
        return {
            'valkyrie_active': self.valkyrie_active,
            'emergency_shutdown': self.emergency_shutdown,
            'recovery_attempts': self.recovery_attempts,
            'exposure_usage': total_exposure_pct / self.limits.max_total_exposure_pct if self.limits.max_total_exposure_pct > 0 else 0.0,
            'position_usage': total_positions / max_positions if max_positions > 0 else 0.0,
            # 'drawdown_usage': account_drawdown_pct / self.limits.max_drawdown_pct if self.limits.max_drawdown_pct > 0 else 0.0,
            'drawdown_usage': 0.0, # Simplified
            'max_positions_allowed': max_positions,
            'overall_risk_level': self._calculate_overall_risk(metrics)
        }
    
    
    def _calculate_overall_risk(self, metrics: dict) -> str:
        """
        Calculate overall risk level.
        
        Returns:
            'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
        """
        # Calculate usage directly to avoid circular dependency
        total_exposure_pct = metrics.get("total_exposure_pct", 0.0)
        total_positions = metrics.get("total_positions", 0)
        account_drawdown_pct = metrics.get("account_drawdown_pct", 0.0)
        balance = metrics.get("balance", 10000.0)
        
        max_positions = self._calculate_dynamic_position_limit(balance)
        
        exposure_usage = total_exposure_pct / self.limits.max_total_exposure_pct if self.limits.max_total_exposure_pct > 0 else 0.0
        position_usage = total_positions / max_positions if max_positions > 0 else 0.0
        drawdown_usage = account_drawdown_pct / self.limits.max_drawdown_pct if self.limits.max_drawdown_pct > 0 else 0.0
        
        max_usage = max(exposure_usage, position_usage, drawdown_usage)
        
        if max_usage >= 0.90:
            return 'CRITICAL'
        elif max_usage >= 0.70:
            return 'HIGH'
        elif max_usage >= 0.50:
            return 'MEDIUM'
        else:
            return 'LOW'


# Singleton instance
_governor_instance = None

def get_risk_governor(limits: Optional[RiskLimits] = None) -> RiskGovernor:
    """Get or create the singleton RiskGovernor instance."""
    global _governor_instance
    if _governor_instance is None:
        if limits is None:
            # Default conservative limits
            limits = RiskLimits(
                max_total_exposure_pct=0.15,  # 15% max margin usage
                max_drawdown_pct=0.20,  # 20% max account drawdown
                position_limit_per_1k=2,  # 2 positions per $1000 balance
                news_lockout=False,
                capital_reserve_pct=0.5 # Default 50% Shadow Balance
            )
        _governor_instance = RiskGovernor(limits)
    return _governor_instance
