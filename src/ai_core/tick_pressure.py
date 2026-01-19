import time
from collections import deque

class TickPressureAnalyzer:
    """
    "Holographic" Market View: Simulates Order Flow (Level 2) using Tick Velocity.
    Detects Institutional Aggression vs Retail Noise.
    
    ENHANCEMENT 2: Added Order Flow Imbalance Analysis (Jan 4, 2026)
    """
    def __init__(self, window_seconds=5):
        self.window_seconds = window_seconds
        self.ticks = deque() # Stores (price, time)
        
        # ENHANCEMENT 2: Order Flow Imbalance Tracking
        self.buy_volume_buffer = deque(maxlen=100)
        self.sell_volume_buffer = deque(maxlen=100)
        self.max_buffer_size = 100
        
    def add_tick(self, tick):
        """
        Add a new tick to the analyzer.
        Args:
            tick: Dictionary containing 'bid', 'ask', 'time'
        """
        if not tick:
            return
            
        # Use Bid price for pressure analysis.
        # IMPORTANT: Use local wall-clock time for timing/velocity.
        # MT5 tick timestamps are often coarse (seconds) which can make duration ~0
        # and artificially saturate velocity/pressure.
        price = tick.get('bid', 0.0)
        current_time = time.time()
        
        if price > 0:
            self.ticks.append((price, current_time))
            self._cleanup(current_time)
        
    def _cleanup(self, current_time):
        while self.ticks and (current_time - self.ticks[0][1] > self.window_seconds):
            self.ticks.popleft()
            
    def get_pressure_metrics(self, point_value=0.01):
        """
        Calculates Tick Pressure (Aggression).
        Returns: Dictionary with pressure metrics
        """
        if len(self.ticks) < 5:
            return {
                'pressure_score': 0.0,
                'intensity': 'LOW',
                'dominance': 'NEUTRAL',
                'state': 'INSUFFICIENT_DATA'
            }
            
        start_price = self.ticks[0][0]
        end_price = self.ticks[-1][0]
        price_change = end_price - start_price
        
        tick_count = len(self.ticks)
        duration = self.ticks[-1][1] - self.ticks[0][1]
        if duration <= 0: duration = 0.001
        
        # Velocity = Ticks per Second (Speed of orders)
        velocity = tick_count / duration
        
        # Normalized Price Change (in Points)
        price_delta_points = price_change / point_value
        
        # Pressure = Price Delta * Velocity
        # Example: +10 points * 5 ticks/sec = +50 (Strong Buy)
        # Example: +1 points * 50 ticks/sec = +50 (Massive Absorption/Buy)
        pressure = price_delta_points * velocity
        
        state = "NEUTRAL"
        intensity = "NORMAL"
        dominance = "NEUTRAL"
        
        # Thresholds (Tuned for Gold)
        if pressure > 50.0: 
            state = "INSTITUTIONAL_BUY"
            intensity = "HIGH"
            dominance = "BUY"
        elif pressure > 15.0: 
            state = "STRONG_BUY"
            intensity = "MEDIUM"
            dominance = "BUY"
        elif pressure < -50.0: 
            state = "INSTITUTIONAL_SELL"
            intensity = "HIGH"
            dominance = "SELL"
        elif pressure < -15.0: 
            state = "STRONG_SELL"
            intensity = "MEDIUM"
            dominance = "SELL"
        elif velocity > 10.0 and abs(price_delta_points) < 2.0: 
            state = "ABSORPTION_FIGHT" # High volume, no movement (Trap)
            intensity = "HIGH"
            dominance = "NEUTRAL"
        elif velocity < 1.0:
            state = "LOW_LIQUIDITY"
            intensity = "LOW"
            
        return {
            'pressure_score': pressure,
            'intensity': intensity,
            'dominance': dominance,
            'state': state,
            'velocity': velocity
        }
    
    # ============================================================================
    # ENHANCEMENT 2: Order Flow Imbalance Analysis
    # Added: January 4, 2026
    # Purpose: Detect buy vs sell aggressor volume for better entry timing
    # ============================================================================
    
    def analyze_order_flow(self, tick):
        """
        Analyze buy vs sell volume imbalance from tick data.
        
        Detects aggressor side (who crossed the spread) and tracks volume imbalance.
        
        Args:
            tick: Dictionary with 'last', 'bid', 'ask', 'volume' keys
            
        Returns:
            Dictionary with:
            - order_flow_imbalance: -1 to 1 (negative = sell pressure, positive = buy pressure)
            - buy_pressure: Ratio of buy volume (0-1)
            - sell_pressure: Ratio of sell volume (0-1)
        """
        if not tick:
            return {
                'order_flow_imbalance': 0.0,
                'buy_pressure': 0.5,
                'sell_pressure': 0.5
            }
        
        try:
            last_price = tick.get('last', tick.get('bid', 0))
            bid = tick.get('bid', 0)
            ask = tick.get('ask', 0)
            volume = tick.get('volume', 1)
            
            # Detect aggressor side (who crossed the spread)
            # If last price >= ask, it's a buy aggressor (market buy order)
            # If last price <= bid, it's a sell aggressor (market sell order)
            
            if last_price >= ask and ask > 0:
                # Buy aggressor - someone hit the ask
                self.buy_volume_buffer.append(volume)
            elif last_price <= bid and bid > 0:
                # Sell aggressor - someone hit the bid
                self.sell_volume_buffer.append(volume)
            else:
                # Mid-price trade or limit order fill - neutral
                # Add half to each side
                self.buy_volume_buffer.append(volume / 2)
                self.sell_volume_buffer.append(volume / 2)
            
            # Calculate imbalance from buffers
            buy_vol = sum(self.buy_volume_buffer) if self.buy_volume_buffer else 0
            sell_vol = sum(self.sell_volume_buffer) if self.sell_volume_buffer else 0
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                # Imbalance: -1 (all sell) to +1 (all buy)
                imbalance = (buy_vol - sell_vol) / total_vol
                buy_pressure = buy_vol / total_vol
                sell_pressure = sell_vol / total_vol
            else:
                imbalance = 0.0
                buy_pressure = 0.5
                sell_pressure = 0.5
            
            return {
                'order_flow_imbalance': float(imbalance),
                'buy_pressure': float(buy_pressure),
                'sell_pressure': float(sell_pressure),
                'buy_volume': float(buy_vol),
                'sell_volume': float(sell_vol)
            }
            
        except Exception as e:
            # On error, return neutral
            return {
                'order_flow_imbalance': 0.0,
                'buy_pressure': 0.5,
                'sell_pressure': 0.5
            }
    
    # ============================================================================
    # ENHANCEMENT 3: The Physicist & The Chemist (Unified Field Theory)
    # Added: January 8, 2026
    # Purpose: Deterministic prediction using Hydro-Thermodynamics
    # ============================================================================

    def calculate_vpin(self) -> float:
        """
        [THE CHEMIST] Calculate Volume-Synchronized Probability of Informed Trading.
        Approximation: VPIN = |V_buy - V_sell| / Total_Volume
        
        Returns:
            VPIN score (0.0 to 1.0). >0.4 indicates Toxic/Informed Flow.
        """
        buy_vol = sum(self.buy_volume_buffer) if self.buy_volume_buffer else 0
        sell_vol = sum(self.sell_volume_buffer) if self.sell_volume_buffer else 0
        total_vol = buy_vol + sell_vol
        
        if total_vol <= 0:
            return 0.0
            
        return abs(buy_vol - sell_vol) / total_vol

    def calculate_reynolds_number(self, spread_points: float, point_value: float = 0.01) -> float:
        """
        [THE PHYSICIST] Calculate Reynolds Number (Re) for Phase Transition detection.
        Re = (Volume * Volatility) / Viscosity(Spread)
        
        Args:
            spread_points: Current spread in points
            point_value: value of a point
            
        Returns:
            Re score. High Re (>1000) = Turbulent (Breakout). Low Re = Laminar (Range).
        """
        if spread_points <= 0 or not self.ticks:
            return 0.0
            
        # 1. Characteristic Volume (Velocity)
        # Using ticks/sec as proxy for flow velocity u
        duration = self.ticks[-1][1] - self.ticks[0][1]
        if duration <= 0: duration = 0.001
        velocity = len(self.ticks) / duration
        
        # 2. Characteristic Length (Volatility/Turbulence)
        # StdDev of prices in window
        prices = [p[0] for p in self.ticks]
        if len(prices) > 2:
            import numpy as np
            volatility = np.std(prices) / point_value
        else:
            volatility = 0.0
            
        # 3. Viscosity (Spread)
        viscosity = spread_points
        
        # Re = (Velocity * Volatility) / Viscosity
        # Scaling factor 100 to make numbers readable (e.g., 0-5000)
        re = (velocity * volatility * 100) / viscosity
        
        return float(re)

    def calculate_navier_stokes_pressure(self, order_flow_imbalance: float, velocity: float) -> float:
        """
        [THE PHYSICIST] Calculate Net Force (Pressure Gradient).
        F = Mass * Acceleration
        Approximation: Force = Imbalance * Velocity
        """
        return order_flow_imbalance * velocity

    def calculate_reaction_rate(self) -> float:
        """
        [THE CHEMIST] Calculate Liquidity Consumption Rate.
        Reaction Rate = Trade Frequency / Time
        (Simple proxy: Ticks per second, which is roughly velocity)
        """
        if len(self.ticks) < 2: 
            return 0.0
            
        duration = self.ticks[-1][1] - self.ticks[0][1]
        if duration <= 0: return 0.0
        
        return len(self.ticks) / duration

    def get_combined_analysis(self, tick, point_value=0.01):
        """
        Get all Physics & Chemistry metrics in one call.
        """
        pressure = self.get_pressure_metrics(point_value)
        order_flow = self.analyze_order_flow(tick)
        
        # Extract needed data for Physics
        ask = tick.get('ask', 0)
        bid = tick.get('bid', 0)
        spread = (ask - bid) if (ask > 0 and bid > 0) else 0.0001
        spread_points = spread / point_value
        
        # [THE PHYSICIST]
        re = self.calculate_reynolds_number(spread_points, point_value)
        ns_pressure = self.calculate_navier_stokes_pressure(
            order_flow['order_flow_imbalance'], 
            pressure['velocity']
        )
        
        # [THE CHEMIST]
        vpin = self.calculate_vpin()
        reaction = self.calculate_reaction_rate()
        
        combined = {
            **pressure,
            **order_flow,
            'physics': {
                'reynolds_number': re,
                'navier_stokes_force': ns_pressure,
                'regime': 'TURBULENT' if re > 500 else 'LAMINAR' # Threshold to be tuned
            },
            'chemistry': {
                'vpin': vpin,
                'reaction_rate': reaction,
                'is_toxic': vpin > 0.4
            }
        }
        
        return combined
