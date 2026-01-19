"""
Trading Logger - Structured logging for clean, actionable trade information
Reduces log noise by only showing:
1. Initial trade entries with full trading plan
2. Trade exits with detailed summaries
3. Decision changes (not repetitive HOLD signals)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

# Use a specific logger for UI output that will be configured to show on console
logger = logging.getLogger("AETHER_UI")


class TradingLogger:
    """Structured logger for trading operations with minimal noise"""
    
    @staticmethod
    def log_initial_trade(bucket_id: str, data: Dict):
        """
        Log trade entry using trader dashboard (event-driven, no spam)
        """
        from .trader_dashboard import get_dashboard
        
        dashboard = get_dashboard()
        dashboard.trade_entry(
            action=data.get('action', 'UNKNOWN'),
            lots=float(data.get('lots', 0)),
            price=float(data.get('entry_price', 0)),
            reason=data.get('reasoning', data.get('reasons', 'N/A')),
            trade_type="ENTRY"
        )

    @staticmethod
    def log_active_status(symbol: str, bucket_id: str, positions: List[Dict], pnl_pips: float, tp_pips: float, next_hedge: Optional[Dict] = None, ai_notes: str = ""):
        """
        Log the ongoing status of an active bucket (Dashboard style)
        """
        # Calculate aggregate stats
        total_lots = sum(p.get('volume', 0) for p in positions)
        net_profit = sum(p.get('profit', 0) for p in positions)
        
        # Determine direction of the main/latest position
        direction = "MIXED"
        if positions:
            last_pos = positions[-1]
            direction = "BUY" if last_pos.get('type') == 0 else "SELL"
            
        msg = []
        msg.append(f">>> [STATUS] {symbol} | {direction} | Lots: {total_lots:.2f} | PnL: {net_profit:.2f} USD ({pnl_pips:.1f} pips)")
        msg.append(f"             Target: +{tp_pips:.1f} pips | Bucket: {bucket_id}")
        
        if next_hedge:
            msg.append(f"             Next Hedge: {next_hedge.get('type')} {next_hedge.get('lots')} lots @ {next_hedge.get('price'):.5f}")
            
        if ai_notes:
            msg.append(f"             AI View: {ai_notes}")
            
        # Use print for immediate console feedback (Dashboard feel)
        print("\n".join(msg), flush=True)
    
    @staticmethod
    def log_trade_close(symbol: str, exit_reason: str, duration: float, pnl_usd: float, pnl_pips: float, positions_closed: int, ai_analysis: Dict):
        """
        Log trade exit using trader dashboard (event-driven, no spam)
        """
        from .trader_dashboard import get_dashboard
        
        # Format duration
        m, s = divmod(duration, 60)
        h, m = divmod(m, 60)
        duration_str = f"{int(h)}h {int(m)}m" if h > 0 else f"{int(m)}m {int(s)}s"
        
        dashboard = get_dashboard()
        dashboard.trade_exit(
            num_positions=positions_closed,
            profit=pnl_usd,
            duration_str=duration_str,
            reason=exit_reason
        )
    
    @staticmethod
    def log_decision_change(symbol: str, old_decision: Dict, new_decision: Dict, reason: str):
        """
        Log when AI decision changes significantly
        
        Args:
            symbol: Trading symbol
            old_decision: Previous decision state
            new_decision: Current decision state
            reason: What changed and why
        """
        # This might be too noisy for the main terminal based on user request.
        # We'll log it to the file logger (standard logging) instead of UI logger.
        file_logger = logging.getLogger("TradingLogger")
        file_logger.info(f"[DECISION] {symbol} changed from {old_decision.get('action')} to {new_decision.get('action')} | {reason}")
    
    @staticmethod
    def log_error(context: str, error: Exception, additional_info: Optional[Dict] = None):
        """
        Log errors with context
        
        Args:
            context: What was being attempted
            error: Exception that occurred
            additional_info: Optional additional context
        """
        print(f"\n[ERROR] {context}")
        print(f"  Error: {str(error)}")
        if additional_info:
            for key, value in additional_info.items():
                print(f"  {key}: {value}")
        print()
    
    @staticmethod
    def log_system_event(event_type: str, message: str, data: Optional[Dict] = None):
        """
        Log system-level events (startup, shutdown, connection, etc.)
        
        Args:
            event_type: Type of event (STARTUP, SHUTDOWN, CONNECTION, etc.)
            message: Event description
            data: Optional event data
        """
        print(f"\n[{event_type}] {message}")
        if data:
            for key, value in data.items():
                print(f"  {key}: {value}")
        print()


class DecisionTracker:
    """
    Tracks AI decisions to detect meaningful changes
    Prevents repetitive HOLD signal logging
    """
    
    def __init__(self):
        self._last_decisions = {}  # symbol -> last_decision
        self._decision_history = {}  # symbol -> list of recent decisions
        
    def should_log_decision(self, symbol: str, current_decision: Dict) -> tuple[bool, Optional[str]]:
        """
        Determine if current decision should be logged
        
        Args:
            symbol: Trading symbol
            current_decision: Current AI decision with keys:
                - action: BUY/SELL/HOLD
                - confidence: Float 0-1
                - reasoning: String explanation
        
        Returns:
            Tuple of (should_log, change_reason)
        """
        last_decision = self._last_decisions.get(symbol)
        
        if not last_decision:
            # First decision for this symbol - log it
            self._update_decision(symbol, current_decision)
            return True, "Initial signal"
        
        # Check for significant changes
        action_changed = last_decision['action'] != current_decision['action']
        confidence_changed = abs(last_decision['confidence'] - current_decision['confidence']) > 0.15
        
        if action_changed:
            reason = f"Action changed: {last_decision['action']} -> {current_decision['action']}"
            self._update_decision(symbol, current_decision)
            return True, reason
        
        if confidence_changed:
            reason = f"Confidence shifted: {last_decision['confidence']:.3f} -> {current_decision['confidence']:.3f}"
            self._update_decision(symbol, current_decision)
            return True, reason
        
        # No significant change - don't log repetitive signals
        return False, None
    
    def _update_decision(self, symbol: str, decision: Dict):
        """Update stored decision for symbol"""
        self._last_decisions[symbol] = {
            'action': decision['action'],
            'confidence': decision['confidence'],
            'timestamp': datetime.now(),
            'reasoning': decision.get('reasoning', '')
        }
        
        # Maintain history (last 10 decisions)
        if symbol not in self._decision_history:
            self._decision_history[symbol] = []
        
        self._decision_history[symbol].append(decision.copy())
        
        # Keep only recent history
        if len(self._decision_history[symbol]) > 10:
            self._decision_history[symbol] = self._decision_history[symbol][-10:]
    
    def get_decision_summary(self, symbol: str) -> Optional[Dict]:
        """Get summary of recent decisions for a symbol"""
        if symbol not in self._decision_history:
            return None
        
        history = self._decision_history[symbol]
        
        return {
            'total_decisions': len(history),
            'last_action': history[-1]['action'] if history else None,
            'avg_confidence': sum(d['confidence'] for d in history) / len(history) if history else 0,
            'action_distribution': {
                'BUY': sum(1 for d in history if d['action'] == 'BUY'),
                'SELL': sum(1 for d in history if d['action'] == 'SELL'),
                'HOLD': sum(1 for d in history if d['action'] == 'HOLD')
            }
        }


def format_pips(value: float, symbol: str) -> str:
    """
    Format pip value correctly based on symbol type
    
    Args:
        value: Pip value to format
        symbol: Trading symbol (e.g., EURUSD, XAUUSD, USDJPY)
    
    Returns:
        Formatted string with correct decimal places
    """
    # JPY pairs and gold use 2 decimal places (100x multiplier)
    if 'JPY' in symbol or 'XAU' in symbol or 'XAG' in symbol:
        return f"{value:.2f}"
    # Major forex pairs use 1 decimal place (10000x multiplier)
    else:
        return f"{value:.1f}"
