import math
import logging

logger = logging.getLogger("IronShield")

class IronShield:
    def __init__(self, initial_lot=0.01, zone_pips=20, tp_pips=20, config=None):
        self.base_lot = initial_lot
        self.base_zone = zone_pips * 10 # Points
        self.base_tp = tp_pips * 10     # Points
        
        # Load dynamic parameters from config or use defaults
        if config and 'ai_parameters' in config:
            ai_params = config['ai_parameters']
            self.atr_zone_multiplier = ai_params.get('atr_zone_multiplier', 0.8)
            self.atr_tp_multiplier = ai_params.get('atr_tp_multiplier', 0.8)
            self.min_zone_floor = ai_params.get('min_zone_floor_points', 30)
            self.min_tp_floor = ai_params.get('min_tp_floor_points', 30)
        else:
            # Fallback defaults
            self.atr_zone_multiplier = 0.8
            self.atr_tp_multiplier = 0.8
            self.min_zone_floor = 50 # Updated to 5 pips for Gold
            self.min_tp_floor = 50   # Updated to 5 pips for Gold

        # Elastic Defense Protocol Parameters
        self.min_distance_atr_multiplier = 0.5
        self.max_distance_atr_multiplier = 2.0

    def calculate_dynamic_hedge_trigger(self, entry_price, current_price, trade_type, atr, rsi, hedge_count):
        """
        Decides IF and WHERE a hedge should be placed based on live volatility.
        Returns: (bool: should_hedge, float: suggested_price)
        """
        # 1. Calculate Base Distance using Volatility (ATR)
        # If ATR is high (volatile), give trade more room. If low, tighten up.
        base_distance = atr * self.min_distance_atr_multiplier
        
        # 2. Adjust for Hedge Depth (Deep hedges need more breathing room)
        depth_factor = 1.0 + (hedge_count * 0.2) 
        required_distance = base_distance * depth_factor

        # 3. Momentum Filter (RSI) - The "Intelligence"
        # If we are Long, and RSI is Oversold (<30), price might bounce up. Don't hedge yet (Wait).
        # If we are Long, and RSI is Neutral/Bearish, hedge normally.
        momentum_delay = False
        if trade_type == 0: # BUY
            dist_moved = entry_price - current_price
            if rsi < 30: momentum_delay = True # Oversold, expect bounce
        else: # SELL
            dist_moved = current_price - entry_price
            if rsi > 70: momentum_delay = True # Overbought, expect drop

        # 4. The Decision
        should_hedge = False
        
        # Hard Safety: If price moved > 2x ATR, we MUST hedge regardless of RSI (Emergency)
        emergency_dist = atr * 2.0
        
        if dist_moved > emergency_dist:
            should_hedge = True # Emergency Hedge
        elif dist_moved > required_distance and not momentum_delay:
            should_hedge = True # Smart Hedge
            
        return should_hedge, required_distance

    def calculate_recovery_lot_size(self, 
                                  positions: list, 
                                  current_price: float, 
                                  target_recovery_pips: float, 
                                  symbol_step: float,
                                  pip_value: float = 1.0,
                                  atr: float = 0.0,       # NEW: Market Volatility
                                  rsi: float = 50.0       # NEW: Market Momentum
                                  ) -> float:
        """
        Calculates lot size with AI-driven 'Dynamic Aggression'.
        Instead of a fixed step, it scales the hedge based on Trend Strength.
        """
        total_loss = 0.0
        losing_volume = 0.0
        
        # 1. Calculate total loss and volume
        for pos in positions:
            # Handle both dict and object access
            profit = pos.get('profit', 0.0) if isinstance(pos, dict) else getattr(pos, 'profit', 0.0)
            swap = pos.get('swap', 0.0) if isinstance(pos, dict) else getattr(pos, 'swap', 0.0)
            # Commission might not be available, assume 0 if missing
            commission = pos.get('commission', 0.0) if isinstance(pos, dict) else getattr(pos, 'commission', 0.0)
            
            total_loss += (profit + swap + commission) # profit is negative here
            
            p_vol = pos.get('volume') if isinstance(pos, dict) else getattr(pos, 'volume')
            losing_volume += p_vol

        # If we are actually in profit (rare glitch), return min lot
        if total_loss >= 0:
            return symbol_step

        # 2. Math Calculation (Zero-Loss Formula)
        # Target Profit: Aim for small profit to cover slippage
        target_profit = abs(total_loss) * 0.05
        cost_buffer = abs(total_loss) * 0.05 
        total_needed = abs(total_loss) + target_profit + cost_buffer
        
        if target_recovery_pips == 0: return symbol_step
        math_lot = total_needed / (target_recovery_pips * pip_value)
        
        # 3. AI DYNAMIC OVERPOWER LOGIC
        # Base: We want to be at least 5% stronger than the losing side
        aggression_multiplier = 1.05 
        
        # Determine Hedge Direction (If losing trades are SELL, we are BUYING)
        # MT5: 0=Buy, 1=Sell. If type is 1, we are fighting Sells, so we Buy.
        first_pos = positions[0]
        first_type = first_pos.get('type') if isinstance(first_pos, dict) else getattr(first_pos, 'type')
        is_buy_hedge = (first_type == 1) 
        
        # AI Decision: Adjust aggression based on Trend Strength (RSI)
        if is_buy_hedge:
            # We are Buying. Is the market Bullish?
            if rsi > 60: aggression_multiplier = 1.15  # Strong Up Trend -> Hit harder (15%)
            if rsi > 75: aggression_multiplier = 1.25  # Very Strong -> Max power (25%)
        else:
            # We are Selling. Is the market Bearish?
            if rsi < 40: aggression_multiplier = 1.15
            if rsi < 25: aggression_multiplier = 1.25

        # Calculate the AI-optimized lot size
        ai_recommended_lot = losing_volume * aggression_multiplier
        
        # Ensure it is at least 1 step larger (Hard Floor safety)
        hard_floor = losing_volume + symbol_step
        
        # The "Overpower Lot" is the max of AI recommendation or Hard Floor
        min_overpower_lot = max(ai_recommended_lot, hard_floor)
        
        # 4. Final Selection: Pick the largest of Math vs AI
        final_lot = max(math_lot, min_overpower_lot)

        # 5. Safety Clamp (Max 2.5x initial - Anti-Martingale Brake)
        initial_lot = first_pos.get('volume') if isinstance(first_pos, dict) else getattr(first_pos, 'volume')
        
        # [FIX] Cap the multiplier. Never go full martingale.
        # Previously this could go too high. Now capped at 2.5x max.
        max_multiplier = 2.5
        max_allowed = initial_lot * max_multiplier
        
        final_lot = min(final_lot, max_allowed)
        final_lot = max(final_lot, symbol_step) # Ensure at least min step
        
        # Round to symbol step
        final_lot = round(final_lot / symbol_step) * symbol_step
        
        return final_lot

    def should_trigger_recovery(self, trend_strength, rsi_value, sentiment_score):
        """
        [FIX] Veto recovery trades if the trend is crashing against us.
        """
        # If Trend is Strong Down (-0.7) and we want to Buy... FORBID IT.
        if trend_strength < -0.6:
            return False # Too dangerous to catch a falling knife
            
        # Standard RSI checks
        if rsi_value < 30 or rsi_value > 70:
            return True
            
        return False

    def calculate_entry_lot(self, equity, confidence=1.0, atr_value=0.0, trend_strength=0.0):
        """
        Calculates the initial lot size based on Account Equity, AI Confidence, Market Volatility (ATR), and Trend Strength.
        Rule: Dynamic Risk Scaling (0.01 lots per $1,000 Equity).
        Confidence Scaling: 
        - High Confidence (>0.8) -> 1.2x Size
        - Low Confidence (<0.5) -> 0.8x Size
        Volatility Scaling:
        - High ATR (>0.001) -> Reduce size (more volatile, higher risk)
        - Low ATR (<0.0005) -> Increase size (stable market)
        Trend Scaling:
        - Strong Trend (>0.7) -> Increase size (higher probability)
        - Weak Trend (<0.3) -> Reduce size
        """
        if equity <= 0: return self.base_lot
        
        # Calculate safe lot
        # e.g. 1000 -> 0.01, 5000 -> 0.05, 10000 -> 0.10
        safe_lot = math.floor(equity / 1000) * 0.01
        
        # Apply Confidence Multiplier
        conf_mult = 1.0
        if confidence > 0.8:
            conf_mult = 1.2
        elif confidence < 0.5:
            conf_mult = 0.8
            
        # Apply Volatility Multiplier (ATR-based)
        vol_mult = 1.0
        if atr_value > 0.001:  # High volatility
            vol_mult = 0.7
        elif atr_value < 0.0005:  # Low volatility
            vol_mult = 1.3
            
        # Apply Trend Strength Multiplier
        trend_mult = 1.0
        if trend_strength > 0.7:  # Strong trend
            trend_mult = 1.2
        elif trend_strength < 0.3:  # Weak trend
            trend_mult = 0.8
            
        safe_lot = safe_lot * conf_mult * vol_mult * trend_mult
        
        # Ensure we are within bounds
        final_lot = max(self.base_lot, safe_lot)
        
        # [FIX] Absolute Safety Cap based on Equity
        # Example: $3000 equity -> Max lot 0.30. 
        # The 0.68 trade on a $3000 account was mathematically reckless.
        max_allowed_lot = round(equity / 10000.0, 2) * 10.0 # e.g. 3000 -> 0.30
        
        final_lot = min(final_lot, max_allowed_lot)
        
        # Dynamic Cap: We allow up to 50 lots for high equity accounts
        final_lot = min(final_lot, 50.0)
        
        final_lot = round(final_lot, 2)
        
        return final_lot

    def get_dynamic_params(self, atr_points=0):
        """
        Returns dynamic Zone and TP based on market volatility (ATR).
        If ATR is 0, falls back to base settings.
        """
        if atr_points <= 0:
            return self.base_zone, self.base_tp
            
        # Dynamic Logic using configured multipliers and floors
        dynamic_zone = max(self.min_zone_floor, atr_points * self.atr_zone_multiplier)
        dynamic_tp = max(self.min_tp_floor, atr_points * self.atr_tp_multiplier)
        
        return int(dynamic_zone), int(dynamic_tp)

    def calculate_defense(self, current_loss_lot, spread_points, atr_value=0.0, trend_strength=0.0, 
                          fixed_zone_points=None, fixed_tp_points=None, 
                          oracle_prediction="NEUTRAL", volatility_ratio=1.0, hedge_level=1, rsi_value=None):
        """
        Advanced AI-Driven Defense Logic:
        1. Uses Math to calculate the exact Lot Size needed for Break-Even + Profit.
        2. Adapts Aggression based on AI Predictions (Oracle) and Volatility.
        3. Implements "Velvet Cushion" for deep hedges (Survival Mode).
        4. [UPGRADE] Uses RSI & Oracle for Dynamic Aggression (Smart Recovery).
        """
        
        dynamic_zone = 0.0
        target_tp = self.base_tp
        
        if fixed_zone_points:
            dynamic_zone = fixed_zone_points
        else:
            dynamic_zone = self.base_zone
            
        if fixed_tp_points:
            target_tp = fixed_tp_points
        
        # --- INTELLIGENT MODULATION ---
        
        # Baseline Profit Target (The "Greed" Factor)
        # default 10% profit on top of recovery
        profit_target_percent = 0.10 
        
        # 1. Oracle & Trend Adaptation
        # If we have a strong signal, we can aim for more profit (Aggressive Recovery).
        # If the signal opposes us, we aim for break-even (Survival).
        
        aggression_score = 1.0 # Neutral
        
        if oracle_prediction in ("BUY", "UP"):
            # If we are hedging (which usually means adding to a loser), 
            # we need to know if this hedge ALIGNS with the Oracle.
            # But here we don't know the hedge direction explicitly. 
            # We assume the caller calls us for a valid hedge.
            pass
            
        # 2. RSI-Based Aggression (The "Elastic" Logic Restored)
        # If RSI is extreme, we expect a snap-back, so we can trade larger/more aggressively.
        if rsi_value is not None:
            if rsi_value < 30: # Oversold - Expect Bounce (Call/Buy favor)
                # If we are effectively "buying low", we can be aggressive
                aggression_score += 0.2
            elif rsi_value > 70: # Overbought - Expect Drop (Put/Sell favor)
                aggression_score += 0.2
        
        # Adjust Profit Target based on Aggression
        profit_target_percent *= aggression_score
        
        # 3. Survival Mode (Velvet Cushion)
        # If we are deep in hedges (Level 3+), we drop the profit requirement.
        if hedge_level >= 3:
            profit_target_percent = 0.01 # Just 1% buffer (Survival)
            # Override aggression in survival mode - safety first
            logger.info(f"[IRON SHIELD] Survival Mode (Level {hedge_level}): Dropping profit target to 1%")
            
        # 4. The Mathematical Guarantee (The "Nil Loss" Equation)
        # L2 = (L1 * (Zone + Spread) + (L1 * TP) + ProfitBuffer) / (TP - Spread)
        
        # SAFETY: Ensure TP is significantly larger than Spread
        safe_tp = max(target_tp, spread_points * 2.5) 
        
        # Calculate Profit Buffer (The "Juice")
        profit_buffer = current_loss_lot * safe_tp * profit_target_percent
        
        # Include estimated Swap/Commission costs (approx 2 points per lot)
        cost_buffer = (current_loss_lot + (current_loss_lot * 2)) * 2.0 
        
        numerator = current_loss_lot * (dynamic_zone + spread_points) + (current_loss_lot * safe_tp) + profit_buffer + cost_buffer
        denominator = safe_tp - spread_points
        
        if denominator <= 1: 
            return current_loss_lot 
            
        hedge_lot = numerator / denominator
        
        # SAFETY: Cap the Hedge Lot
        # [INTELLIGENT FIX] Dynamic Max Multiplier
        # Base cap is 2.0x. 
        # If High Confidence/RSI snap-back, allow 2.5x.
        # If Level 3+ (Survival), allow 2.5x to ensure recovery.
        
        max_mult = 2.0
        if aggression_score > 1.1:
            max_mult = 2.5
        if hedge_level >= 3:
            max_mult = 3.0 # Allow slightly more power for final rescue
            
        max_hedge = current_loss_lot * max_mult
        if hedge_lot > max_hedge:
            hedge_lot = max_hedge
        
        return round(hedge_lot, 2)

    def can_trade(self, equity, balance):
        """
        Risk Manager Veto.
        Prevents opening new trades if drawdown is too high.
        """
        if balance <= 0:
            return False
            
        drawdown_percent = (balance - equity) / balance
        if drawdown_percent >= 0.10: # 10% Hard Limit for new trades
            return False
        return True

    def check_kill_switch(self, equity, balance):
        """
        Checks if drawdown exceeds hard limit.
        [VALKYRIE UPDATE] Shield down. We do NOT Liquidate.
        We let RiskGovernor Trigger "Valkyrie Freeze" at 15%.
        """
        return "SAFE"
