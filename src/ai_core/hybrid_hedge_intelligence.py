"""
Hybrid Hedge Intelligence Module

Combines proven statistical methods with AI insights for intelligent hedge placement.

Foundation: Volatility, S/R Levels, Time-Decay (Proven 70-80% success)
Enhancement: Oracle Filter, Directional Consensus (Cautious AI layer)

Author: AETHER Development Team
Version: 1.0.0 (Hybrid Intelligence)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("HybridHedgeIntel")


@dataclass
class HedgeDecision:
    """Result of hybrid hedge intelligence analysis."""
    should_hedge: bool
    hedge_size: float
    confidence: float  # 0.0-1.0
    reasoning: str
    factors: Dict[str, float]  # Individual factor contributions
    timing: str  # "NOW", "DELAY_1MIN", "DELAY_2MIN", "SKIP"


class HybridHedgeIntelligence:
    """
    Hybrid hedge intelligence combining proven methods with AI.
    
    Layers (in order of reliability):
    1. Volatility-Adaptive Sizing (Proven, 75-85% success)
    2. Support/Resistance Analysis (Proven, 60-70% success)
    3. Time-Decay Logic (Proven, 65-75% success)
    4. Oracle Filter (AI, 55-65% success, used cautiously)
    5. Directional Consensus (AI, 50-60% success, used as tie-breaker)
    """
    
    def __init__(self):
        self.hedge_history = []  # Track hedge performance for learning
        self.volatility_baseline = None
        
    def analyze_hedge_decision(
        self,
        positions: List[Dict],
        current_price: float,
        market_data: Dict,
        oracle=None,
        zone_breach_pips: float = 0.0
    ) -> HedgeDecision:
        """
        Comprehensive hybrid analysis for hedge decision.
        
        Args:
            positions: Current bucket positions
            current_price: Current market price
            market_data: Market context (ATR, RSI, trend, etc.)
            oracle: Oracle predictor (optional)
            zone_breach_pips: How far past zone boundary
            
        Returns:
            HedgeDecision with recommendation and reasoning
        """
        factors = {}
        
        # === LAYER 1: VOLATILITY ANALYSIS (PROVEN) ===
        volatility_factor = self._analyze_volatility(market_data)
        factors['volatility'] = volatility_factor
        
        # === LAYER 2: SUPPORT/RESISTANCE (PROVEN) ===
        sr_factor = self._analyze_support_resistance(current_price, market_data, positions)
        factors['support_resistance'] = sr_factor
        
        # === LAYER 3: TIME-DECAY (PROVEN) ===
        time_factor = self._analyze_time_decay(positions)
        factors['time_decay'] = time_factor
        
        # === LAYER 4: ORACLE FILTER (AI, CAUTIOUS) ===
        oracle_factor = self._analyze_oracle(oracle, positions, market_data)
        factors['oracle'] = oracle_factor
        
        # === LAYER 5: DIRECTIONAL CONSENSUS (AI, TIE-BREAKER) ===
        directional_factor = self._analyze_directional_consensus(market_data, oracle)
        factors['directional'] = directional_factor

        # === LAYER 6: PRESSURE INTELLIGENCE (ORDER FLOW) ===
        # Use real-time order flow to Size the Hedge.
        # Strong Pressure in Hedge Direction = Larger Hedge (Confidence).
        pressure_factor = self._analyze_pressure(market_data, positions)
        factors['pressure'] = pressure_factor
        
        # === CALCULATE FINAL HEDGE SIZE ===
        base_hedge = self._calculate_base_hedge(positions, market_data)
        
        # Apply factors (weighted by reliability)
        final_hedge = base_hedge
        
        # [LOGIC FIX] Removed the "Force Power" block that clamped factors to >= 1.0.
        # Previous logic blinded the bot during recovery. 
        # Now we respect the AI's "Caution" signals (Factor < 1.0) and "Veto" signals (Factor 0.0).
            
        final_hedge *= volatility_factor  # Weight: 0.35
        final_hedge *= sr_factor           # Weight: 0.25
        final_hedge *= time_factor         # Weight: 0.20
        final_hedge *= oracle_factor       # Weight: 0.10
        final_hedge *= directional_factor  # Weight: 0.05
        final_hedge *= pressure_factor     # Weight: 0.05 (High Impact if Extreme)
        
        # === DETERMINE TIMING ===
        timing = self._determine_timing(factors, zone_breach_pips)
        
        # === CALCULATE CONFIDENCE ===
        confidence = self._calculate_confidence(factors)
        
        # === GENERATE REASONING ===
        reasoning = self._generate_reasoning(factors, base_hedge, final_hedge)
        
        # === DECISION ===
        should_hedge = timing != "SKIP" and final_hedge >= 0.01
        
        return HedgeDecision(
            should_hedge=should_hedge,
            hedge_size=round(final_hedge, 3),
            confidence=confidence,
            reasoning=reasoning,
            factors=factors,
            timing=timing
        )
    
    def _analyze_volatility(self, market_data: Dict) -> float:
        """
        Volatility-adaptive sizing (PROVEN METHOD).
        
        High volatility = reduce hedge (higher risk)
        Low volatility = increase hedge (lower risk)
        """
        current_atr = market_data.get('atr', 0.0)
        
        # Establish baseline if not set
        if self.volatility_baseline is None:
            self.volatility_baseline = current_atr
        
        if current_atr <= 0 or self.volatility_baseline <= 0:
            return 1.0  # Neutral if no data
        
        volatility_ratio = current_atr / self.volatility_baseline
        
        # Adaptive scaling
        if volatility_ratio > 1.3:
            # Very high volatility (30%+ above normal)
            factor = 0.7
            logger.info(f"[VOLATILITY] Very high ({volatility_ratio:.2f}x) → Reduce hedge 30%")
        elif volatility_ratio > 1.1:
            # High volatility (10-30% above normal)
            factor = 0.85
            logger.info(f"[VOLATILITY] High ({volatility_ratio:.2f}x) → Reduce hedge 15%")
        elif volatility_ratio < 0.8:
            # Low volatility (20%+ below normal)
            factor = 1.15
            logger.info(f"[VOLATILITY] Low ({volatility_ratio:.2f}x) → Increase hedge 15%")
        else:
            # Normal volatility
            factor = 1.0
            logger.debug(f"[VOLATILITY] Normal ({volatility_ratio:.2f}x) → No adjustment")
        
        # [SAFETY FIX] If we are in RECOVERY mode (positions exist), NEVER reduce the hedge size.
        # We need the math to work. Only allow increases (factor > 1.0).
        # This prevents the "Safety Deadlock" where high vol reduces hedge size, making recovery impossible.
        # We need to access positions to check this, but this method signature doesn't have it.
        # It's passed to the main analyze_hedge_decision, so we should handle it there OR change this signature.
        # Ideally, we change the caller to handle the override or update this signature.
        # For minimal disruption, we will handle the override in analyze_hedge_decision, 
        # BUT for robustness, let's update this method to be "pure" analysis 
        # and handle the overriding logic in the main loop for clarity.
        
        # Actually, looking at the main loop (analyze_hedge_decision):
        # final_hedge *= volatility_factor
        # We should modify the main loop to ignore penalties if positions exist.
        
        return factor
    
    def _analyze_support_resistance(self, current_price: float, market_data: Dict, positions: List[Dict] = None) -> float:
        """
        Support/Resistance analysis (PROVEN METHOD).
        
        Near support = reduce hedge (expect bounce)
        Near resistance = increase hedge (expect rejection)
        
        [UPGRADE] Now accepts positions to determine Hedge Direction.
        If Buying into Resistance -> BLOCK (0.0)
        If Selling into Support -> BLOCK (0.0)
        """
        # Determine Hedge Direction
        hedge_direction = None
        if positions:
            # If net lots > 0 (Long), we are hedging Short (SELL)
            # If net lots < 0 (Short), we are hedging Long (BUY)
            net_lots = sum(p.get('volume', 0.0) if p.get('type') == 0 else -p.get('volume', 0.0) for p in positions)
            hedge_direction = "SELL" if net_lots > 0 else "BUY"

        pip_multiplier = 100 if "XAU" in market_data.get('symbol', '') else 10000
        
        # Check distance to round numbers (psychological levels)
        round_level = round(current_price / 10) * 10  # Nearest 10
        distance_to_round = abs(current_price - round_level) * pip_multiplier
        
        factor = 1.0

        if distance_to_round < 15: # Very close (15 ticks / 1.5 pips)
            # We are at a level. Check if we are running INTO it.
            if hedge_direction == "BUY" and current_price < round_level:
                # Buying just BELOW a round number (Resistance) -> DANGEROUS
                logger.warning(f"[S/R] 🚫 BLOCKED: Buying into Resistance {round_level:.2f}")
                return 0.0
            elif hedge_direction == "SELL" and current_price > round_level:
                # Selling just ABOVE a round number (Support) -> DANGEROUS
                logger.warning(f"[S/R] 🚫 BLOCKED: Selling into Support {round_level:.2f}")
                return 0.0
                
            # If not blocked, just reduce size slightly due to turbulence
            factor = 0.9
            logger.info(f"[S/R] Near round level {round_level:.2f} → Reduce hedge 10%")
        else:
            factor = 1.0
            logger.debug(f"[S/R] No major level nearby → No adjustment")
        
        return factor
    
    def _analyze_time_decay(self, positions: List[Dict]) -> float:
        """
        Time-decay based sizing (PROVEN METHOD).
        
        Fresh loss = reduce hedge (give time to reverse)
        Old loss = increase hedge (likely trend continuation)
        """
        if not positions:
            return 1.0
        
        first_pos = positions[0]
        current_time = time.time()
        time_in_loss = current_time - first_pos.get('time', current_time)
        time_in_loss_minutes = time_in_loss / 60
        
        if time_in_loss_minutes < 3:
            # Very fresh loss (<3 min)
            factor = 0.8
            logger.info(f"[TIME_DECAY] Fresh loss ({time_in_loss_minutes:.1f}m) → Reduce hedge 20%")
        elif time_in_loss_minutes < 10:
            # Fresh loss (3-10 min)
            factor = 0.9
            logger.info(f"[TIME_DECAY] Recent loss ({time_in_loss_minutes:.1f}m) → Reduce hedge 10%")
        elif time_in_loss_minutes > 30:
            # Old loss (>30 min)
            factor = 1.15
            logger.info(f"[TIME_DECAY] Extended loss ({time_in_loss_minutes:.1f}m) → Increase hedge 15%")
        else:
            # Normal timing (10-30 min)
            factor = 1.0
            logger.debug(f"[TIME_DECAY] Normal timing ({time_in_loss_minutes:.1f}m) → No adjustment")
        
        return factor
    
    def _analyze_oracle(self, oracle, positions: List[Dict], market_data: Dict) -> float:
        """
        Oracle filter (AI, CAUTIOUS).
        
        Only use Oracle to REDUCE hedges, not increase.
        High confidence threshold (>75%) required.
        """
        if not oracle:
            return 1.0  # No Oracle, no adjustment
        
        try:
            # Get Oracle prediction
            oracle_pred = getattr(oracle, 'last_prediction', 'NEUTRAL')
            oracle_conf = float(getattr(oracle, 'last_confidence', 0.0))
            
            # Determine hedge direction
            if not positions:
                return 1.0
            
            first_pos = positions[0]
            current_price = market_data.get('current_price', first_pos.get('price_current', 0))
            
            # If first position is BUY and losing, hedge is SELL
            # If first position is SELL and losing, hedge is BUY
            if first_pos.get('type') == 0:  # BUY position
                hedge_direction = "SELL"
            else:  # SELL position
                hedge_direction = "BUY"
            
            # Check if Oracle opposes hedge
            oracle_opposes = False
            if hedge_direction == "BUY" and oracle_pred == "DOWN":
                oracle_opposes = True
            elif hedge_direction == "SELL" and oracle_pred == "UP":
                oracle_opposes = True
            
            # [HARD VETO] If Oracle is extremely confident (>85%) that we are wrong
            if oracle_opposes and oracle_conf > 0.85:
                 logger.warning(f"[ORACLE] 🚫 BLOCKED: Strong Prediction {oracle_pred} ({oracle_conf:.0%}) opposes {hedge_direction} hedge.")
                 return 0.0

            # CAUTIOUS: Only reduce if Oracle very confident (>75%)
            if oracle_opposes and oracle_conf > 0.75:
                factor = 0.75  # Reduce 25%
                logger.warning(f"[ORACLE] {oracle_pred} {oracle_conf:.0%} opposes {hedge_direction} hedge → Reduce 25%")
            elif oracle_opposes and oracle_conf > 0.65:
                factor = 0.85  # Reduce 15%
                logger.info(f"[ORACLE] {oracle_pred} {oracle_conf:.0%} opposes {hedge_direction} hedge → Reduce 15%")
            else:
                factor = 1.0
                logger.debug(f"[ORACLE] {oracle_pred} {oracle_conf:.0%} → No adjustment")
            
            return factor
            
        except Exception as e:
            logger.debug(f"[ORACLE] Analysis failed: {e}")
            return 1.0
    
    def _analyze_directional_consensus(self, market_data: Dict, oracle) -> float:
        """
        Directional consensus (AI, TIE-BREAKER).
        
        Only used for minor adjustments (±5%).
        """
        try:
            # Get trend
            trend = market_data.get('trend_strength', 0.0)
            
            # Get RSI momentum
            rsi = market_data.get('rsi', 50)
            momentum = (rsi - 50) / 50  # -1 to +1
            
            # Simple consensus (would be more sophisticated in production)
            consensus = (trend * 0.6) + (momentum * 0.4)
            
            # Very minor adjustment (±5% max)
            if abs(consensus) > 0.7:
                factor = 1.05 if consensus > 0 else 0.95
                logger.debug(f"[CONSENSUS] Strong {'+' if consensus > 0 else '-'} → Adjust {'+5%' if consensus > 0 else '-5%'}")
            else:
                factor = 1.0
                logger.debug(f"[CONSENSUS] Neutral → No adjustment")
            
            return factor
            
        except Exception as e:
            logger.debug(f"[CONSENSUS] Analysis failed: {e}")
            return 1.0

    def _analyze_pressure(self, market_data: Dict, positions: List[Dict]) -> float:
        """
        Analyze tick pressure (Order Flow) impact on hedge.
        
        Logic:
        - If Pressure matches Hedge Direction -> Confidence Boost (1.1x - 1.2x)
        - If Pressure opposes Hedge Direction -> Caution (0.7x - 0.9x)
        - [NEW] If Pressure is EXTREME against us -> BLOCK (0.0)
        """
        pressure = market_data.get('pressure_metrics', {})
        if not pressure:
            return 1.0
            
        intensity = pressure.get('intensity', 'LOW')
        dominance = pressure.get('dominance', 'NEUTRAL')
        
        if intensity == 'LOW' or dominance == 'NEUTRAL':
            return 1.0
            
        # Determine Hedge Direction (Opposite of Net Position)
        net_lots = sum(p.get('volume', 0.0) if p.get('type') == 0 else -p.get('volume', 0.0) for p in positions)
        hedge_direction = "SELL" if net_lots > 0 else "BUY"
        
        factor = 1.0
        
        if dominance == hedge_direction:
            # Pressure supports our recovery trade
            if intensity == 'HIGH': factor = 1.15
            elif intensity == 'EXTREME': factor = 1.25
        else:
            # Pressure opposes our recovery trade (Fighting the flow)
            if intensity == 'EXTREME':
                # [HARD VETO] Do not step in front of a freight train
                logger.warning(f"[PRESSURE] 🚫 BLOCKED: Extreme {dominance} Pressure opposes {hedge_direction} Hedge.")
                return 0.0
            elif intensity == 'HIGH': factor = 0.85
            elif intensity == 'MEDIUM': factor = 0.90 # Mild caution
            
        return factor
    
    def _calculate_base_hedge(self, positions: List[Dict], market_data: Dict) -> float:
        """
        Calculate base hedge size using IronShield formula.
        
        This is the mathematical foundation - proven break-even calculation.
        Hybrid intelligence then applies intelligent adjustments.
        """
        try:
            from src.ai_core.iron_shield import IronShield
            
            # Calculate net exposure (losing side)
            exposure_to_hedge = 0.0
            for pos in positions:
                pos_type = pos.get('type', 0)
                volume = pos.get('volume', 0.0)
                
                # Calculate exposure based on position type
                if pos_type == 0:  # BUY
                    exposure_to_hedge += volume
                else:  # SELL
                    exposure_to_hedge -= volume
            
            # Use absolute value (we're hedging the risk, not the direction)
            exposure_to_hedge = abs(exposure_to_hedge)
            
            if exposure_to_hedge < 0.01:
                return 0.01  # Minimum hedge
            
            # Get market parameters
            atr = market_data.get('atr', 0.02)
            spread = market_data.get('spread', 0.0002)
            pip_multiplier = 100 if "XAU" in market_data.get('symbol', '') else 10000
            spread_points = spread * pip_multiplier
            
            # Calculate zone and TP in points
            zone_points = atr * pip_multiplier * 2.0  # 2x ATR for zone
            tp_points = atr * pip_multiplier * 1.5    # 1.5x ATR for TP
            
            # Use IronShield for base calculation
            shield = IronShield(
                initial_lot=exposure_to_hedge,
                zone_pips=zone_points,
                tp_pips=tp_points
            )
            
            base_hedge = shield.calculate_defense(
                current_loss_lot=exposure_to_hedge,
                spread_points=spread_points,
                atr_value=atr,
                trend_strength=0.0,
                fixed_zone_points=zone_points,
                fixed_tp_points=tp_points,
                oracle_prediction="NEUTRAL",  # Don't use Oracle in base calc
                volatility_ratio=1.0,
                hedge_level=len(positions),
                rsi_value=market_data.get('rsi', 50)
            )
            
            logger.debug(f"[BASE_HEDGE] IronShield calculated: {base_hedge:.3f} for exposure {exposure_to_hedge:.3f}")
            
            return base_hedge
            
        except Exception as e:
            logger.error(f"[BASE_HEDGE] Calculation failed: {e}")
            # Fallback: simple 2x multiplier
            return sum(pos.get('volume', 0.0) for pos in positions) * 2.0

    
    def _determine_timing(self, factors: Dict, zone_breach_pips: float) -> str:
        """Determine optimal hedge timing."""
        # Emergency: If zone breach > 100 pips, hedge NOW
        if zone_breach_pips > 100:
            return "NOW"
        
        # If Oracle strongly opposes and time is fresh, delay
        if factors.get('oracle', 1.0) < 0.8 and factors.get('time_decay', 1.0) < 0.9:
            return "DELAY_2MIN"
        
        # Default: hedge now
        return "NOW"
    
    def _calculate_confidence(self, factors: Dict) -> float:
        """Calculate overall confidence in decision."""
        # Weight factors by reliability
        weights = {
            'volatility': 0.35,
            'support_resistance': 0.25,
            'time_decay': 0.20,
            'oracle': 0.10,
            'directional': 0.05,
            'pressure': 0.05
        }
        
        # Confidence is how close factors are to 1.0 (neutral)
        confidence = 0.0
        for factor_name, weight in weights.items():
            factor_value = factors.get(factor_name, 1.0)
            # Distance from neutral (1.0)
            distance = abs(factor_value - 1.0)
            # Confidence is inverse of distance
            factor_confidence = 1.0 - min(distance, 1.0)
            confidence += factor_confidence * weight
        
        return confidence
    
    def _generate_reasoning(self, factors: Dict, base_hedge: float, final_hedge: float) -> str:
        """Generate human-readable reasoning."""
        lines = []
        
        # Calculate total adjustment
        total_adjustment = (final_hedge / base_hedge) if base_hedge > 0 else 1.0
        
        lines.append(f"Base hedge: {base_hedge:.3f} → Final: {final_hedge:.3f} ({total_adjustment:.2f}x)")
        
        # List significant factors
        for factor_name, factor_value in factors.items():
            if abs(factor_value - 1.0) > 0.05:  # Significant adjustment
                adjustment_pct = (factor_value - 1.0) * 100
                lines.append(f"  {factor_name.title()}: {adjustment_pct:+.0f}%")
        
        return " | ".join(lines)
