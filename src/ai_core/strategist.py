import logging
import json

logger = logging.getLogger("STRATEGIST")

class Strategist:
    """
    The 'Boss' Agent.
    Analyzes the overall performance of the trading session and adjusts
    global risk parameters dynamically.
    
    ENHANCEMENT 4: Added Win Rate Tracking (Jan 4, 2026)
    """
    def __init__(self):
        self.global_risk_multiplier = 1.0
        self.session_profit = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.total_trades = 0
        
        # ENHANCEMENT 4: Win Rate Tracking
        self.trade_outcomes = []  # Track wins/losses (1 = win, 0 = loss)
        self.max_history = 50  # Keep last 50 trades for win rate calculation
        self.recent_win_rate = 0.5  # Start neutral

    def update_stats(self, profit):
        """
        Ingests the result of a closed trade/bucket.
        
        ENHANCEMENT 4: Now tracks individual trade outcomes
        """
        self.session_profit += profit
        self.total_trades += 1
        
        # ENHANCEMENT 4: Track outcome (1 for win, 0 for loss)
        outcome = 1 if profit > 0 else 0
        self.trade_outcomes.append(outcome)
        
        # Keep only last N trades
        if len(self.trade_outcomes) > self.max_history:
            self.trade_outcomes.pop(0)
        
        if profit > 0:
            self.gross_profit += profit
        else:
            self.gross_loss += abs(profit)
            
        self._review_performance()

    def get_win_rate(self):
        """
        Calculate current win rate from recent trades.
        
        Returns:
            float: Win rate from 0.0 to 1.0
        """
        if not self.trade_outcomes:
            return 0.5  # Neutral if no data
        
        wins = sum(self.trade_outcomes)
        total = len(self.trade_outcomes)
        return wins / total if total > 0 else 0.5

    def _review_performance(self):
        """
        The 'Thinking' Process.
        Evaluates the Profit Factor and Win Rate, then adjusts Risk Multiplier using Kelly Criterion.
        
        ENHANCEMENT 4: Now uses both Profit Factor AND Win Rate
        """
        # Avoid division by zero
        if self.gross_loss == 0:
            pf = 10.0 if self.gross_profit > 0 else 1.0
        else:
            pf = self.gross_profit / self.gross_loss
            
        # ENHANCEMENT 4: Calculate Win Rate
        win_rate = self.get_win_rate()
        self.recent_win_rate = win_rate  # Store for external access
        
        # --- ENHANCED KELLY CRITERION IMPLEMENTATION ---
        # Formula: f* = (bp - q) / b
        # where:
        #   p = win rate
        #   q = loss rate (1 - p)
        #   b = avg_win / avg_loss (approximated by profit factor)
        
        # Calculate target multiplier based on BOTH metrics
        target_multiplier = 1.0
        
        # Primary: Profit Factor Analysis
        if pf > 2.0:
            pf_mult = 2.0  # High confidence
        elif pf > 1.5:
            pf_mult = 1.5
        elif pf < 0.8:
            pf_mult = 0.5  # Defensive mode
        elif pf < 0.5:
            pf_mult = 0.1  # Survival mode
        else:
            pf_mult = 1.0
        
        # ENHANCEMENT 4: Secondary - Win Rate Analysis
        if win_rate > 0.6:
            wr_mult = 1.3  # Good win rate, increase risk
        elif win_rate > 0.55:
            wr_mult = 1.1  # Slightly above average
        elif win_rate < 0.4:
            wr_mult = 0.7  # Poor win rate, reduce risk
        elif win_rate < 0.45:
            wr_mult = 0.85  # Below average
        else:
            wr_mult = 1.0  # Neutral
        
        # ENHANCEMENT 4: Combined Kelly Criterion
        # If BOTH metrics are strong, amplify. If conflicting, be conservative.
        if pf > 1.5 and win_rate > 0.6:
            # Both metrics strong - high confidence
            target_multiplier = min(pf_mult * wr_mult, 2.5)  # Cap at 2.5x
        elif pf < 0.8 or win_rate < 0.4:
            # Either metric weak - defensive
            target_multiplier = min(pf_mult, wr_mult)  # Take the more conservative
        else:
            # Mixed signals - average them
            target_multiplier = (pf_mult + wr_mult) / 2.0
        
        # Smooth transition (don't jump from 1.0 to 2.0 instantly)
        old_multiplier = self.global_risk_multiplier
        self.global_risk_multiplier = (self.global_risk_multiplier * 0.8) + (target_multiplier * 0.2)
        
        # ENHANCEMENT 4: Enhanced logging with win rate
        logger.info(
            f"Strategist Review: PF={pf:.2f} | WinRate={win_rate:.1%} ({sum(self.trade_outcomes)}/{len(self.trade_outcomes)}) "
            f"-> Risk Multiplier={self.global_risk_multiplier:.2f}"
        )

    def get_risk_multiplier(self):
        """Get current risk multiplier."""
        return self.global_risk_multiplier
    
    def get_performance_summary(self):
        """
        Get comprehensive performance summary.
        
        Returns:
            dict: Performance metrics
        """
        return {
            'total_trades': self.total_trades,
            'session_profit': self.session_profit,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'profit_factor': self.gross_profit / self.gross_loss if self.gross_loss > 0 else 999.0,
            'win_rate': self.get_win_rate(),
            'risk_multiplier': self.global_risk_multiplier,
            'recent_trades': len(self.trade_outcomes)
        }
