from collections import deque
from typing import Optional

class RollingStats:
    def __init__(self, window=60):
        self.window = window
        self.buf = deque(maxlen=window)

    def push(self, x: float):
        self.buf.append(float(x))

    def mean(self) -> float:
        if not self.buf: return 0.0
        return sum(self.buf) / max(1, len(self.buf))

    def std(self) -> float:
        if not self.buf: return 0.0
        m = self.mean()
        return (sum((x - m)**2 for x in self.buf) / max(1, len(self.buf))) ** 0.5

def spread_atr(spread_points: float, atr_points: float) -> float:
    atr_points = max(1e-9, float(atr_points))
    return float(spread_points) / atr_points

def zscore(x: float, mean: float, std: float) -> float:
    std = max(1e-9, std)
    return (float(x) - float(mean)) / std

def simple_breakout_quality(high: float, low: float, close: float) -> float:
    rng = max(1e-9, high - low)
    clv = (close - low) / rng  # 0..1
    return max(0.0, min(1.0, clv))

def simple_regime_trend(slope: float, atr: float) -> float:
    atr = max(1e-9, atr)
    norm = min(1.0, abs(slope) / atr)  # crude normalization
    return norm


def linear_regression_slope(values: list[float]) -> float:
    """Return slope per index using simple OLS on x=0..n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    sum_x = (n - 1) * n / 2
    sum_xx = (n - 1) * n * (2 * n - 1) / 6
    sum_y = float(sum(values))
    sum_xy = float(sum(i * v for i, v in enumerate(values)))
    denom = (n * sum_xx - sum_x * sum_x)
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def simple_structure_break(candles: list[dict], lookback: int = 20) -> float:
    """Detect a simple range break using real candles.

    Returns:
        +1.0 if last close > prior lookback high
        -1.0 if last close < prior lookback low
        0.0 otherwise
    """
    if not candles or len(candles) < 3:
        return 0.0

    # Use the most recent candle as the "break" candle, compare to prior range.
    last = candles[-1]
    prior = candles[:-1]
    lb = max(2, min(int(lookback), len(prior)))
    window = prior[-lb:]

    prior_high = max(c['high'] for c in window)
    prior_low = min(c['low'] for c in window)
    close = float(last.get('close'))

    if close > prior_high:
        return 1.0
    if close < prior_low:
        return -1.0
    return 0.0


# ============================================================================
# ENHANCEMENT 1: Advanced Feature Engineering
# Added: January 4, 2026
# Purpose: Expand feature set for better AI predictions
# ============================================================================

def calculate_volume_features(candles: list[dict]) -> dict:
    """
    Calculate volume-based features for better entry timing.
    
    Args:
        candles: List of candle dictionaries with 'tick_volume' key
        
    Returns:
        Dictionary with volume features:
        - volume_sma_ratio: Current volume / 20-period average
        - volume_spike: 1 if volume > 1.5x average, else 0
        - volume_trend: Linear regression slope of last 10 volumes
    """
    if not candles or len(candles) < 20:
        return {
            'volume_sma_ratio': 1.0,
            'volume_spike': 0,
            'volume_trend': 0.0
        }
    
    try:
        volumes = [float(c.get('tick_volume', 1)) for c in candles]
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-20:]) / 20.0
        
        # Avoid division by zero
        if avg_volume == 0:
            avg_volume = 1.0
        
        volume_sma_ratio = current_volume / avg_volume
        volume_spike = 1 if current_volume > (avg_volume * 1.5) else 0
        
        # Volume trend (last 10 candles)
        if len(volumes) >= 10:
            volume_trend = linear_regression_slope(volumes[-10:])
        else:
            volume_trend = 0.0
        
        return {
            'volume_sma_ratio': float(volume_sma_ratio),
            'volume_spike': int(volume_spike),
            'volume_trend': float(volume_trend)
        }
    except Exception:
        return {
            'volume_sma_ratio': 1.0,
            'volume_spike': 0,
            'volume_trend': 0.0
        }


def calculate_price_momentum(candles: list[dict]) -> dict:
    """
    Calculate momentum indicators for trend strength assessment.
    
    Args:
        candles: List of candle dictionaries with 'close' key
        
    Returns:
        Dictionary with momentum features:
        - roc_5: 5-period Rate of Change
        - roc_10: 10-period Rate of Change
        - momentum_strength: Combined momentum score
    """
    if not candles or len(candles) < 11:
        return {
            'roc_5': 0.0,
            'roc_10': 0.0,
            'momentum_strength': 0.0
        }
    
    try:
        closes = [float(c.get('close', 0)) for c in candles]
        
        # Rate of Change calculations
        roc_5 = 0.0
        roc_10 = 0.0
        
        if len(closes) > 5 and closes[-6] != 0:
            roc_5 = (closes[-1] - closes[-6]) / closes[-6]
        
        if len(closes) > 10 and closes[-11] != 0:
            roc_10 = (closes[-1] - closes[-11]) / closes[-11]
        
        # Combined momentum strength
        momentum_strength = abs(roc_5) + abs(roc_10)
        
        return {
            'roc_5': float(roc_5),
            'roc_10': float(roc_10),
            'momentum_strength': float(momentum_strength)
        }
    except Exception:
        return {
            'roc_5': 0.0,
            'roc_10': 0.0,
            'momentum_strength': 0.0
        }


def calculate_volatility_features(candles: list[dict]) -> dict:
    """
    Calculate volatility metrics for risk assessment.
    
    Args:
        candles: List of candle dictionaries with OHLC data
        
    Returns:
        Dictionary with volatility features:
        - atr_normalized: ATR / current close (volatility as % of price)
        - volatility_percentile: 80th percentile of true range
        - range_expansion: Current range / average range
    """
    if not candles or len(candles) < 20:
        return {
            'atr_normalized': 0.0,
            'volatility_percentile': 0.0,
            'range_expansion': 1.0
        }
    
    try:
        # Get last 20 candles for analysis
        recent = candles[-20:]
        
        highs = [float(c.get('high', 0)) for c in recent]
        lows = [float(c.get('low', 0)) for c in recent]
        closes = [float(c.get('close', 0)) for c in recent]
        
        # Calculate True Range for each candle
        true_ranges = []
        for i in range(1, len(recent)):
            h = highs[i]
            l = lows[i]
            prev_close = closes[i-1]
            
            tr = max(
                h - l,
                abs(h - prev_close),
                abs(l - prev_close)
            )
            true_ranges.append(tr)
        
        if not true_ranges or closes[-1] == 0:
            return {
                'atr_normalized': 0.0,
                'volatility_percentile': 0.0,
                'range_expansion': 1.0
            }
        
        # ATR (Average True Range)
        atr = sum(true_ranges) / len(true_ranges)
        atr_normalized = atr / closes[-1]
        
        # Volatility percentile (80th percentile of TR)
        sorted_tr = sorted(true_ranges)
        percentile_idx = int(len(sorted_tr) * 0.8)
        volatility_percentile = sorted_tr[percentile_idx] if percentile_idx < len(sorted_tr) else sorted_tr[-1]
        
        # Range expansion (current range vs average range)
        current_range = highs[-1] - lows[-1]
        avg_range = sum(h - l for h, l in zip(highs, lows)) / len(highs)
        
        if avg_range == 0:
            range_expansion = 1.0
        else:
            range_expansion = current_range / avg_range
        
        return {
            'atr_normalized': float(atr_normalized),
            'volatility_percentile': float(volatility_percentile),
            'range_expansion': float(range_expansion)
        }
    except Exception:
        return {
            'atr_normalized': 0.0,
            'volatility_percentile': 0.0,
            'range_expansion': 1.0
        }


def calculate_price_position(candles: list[dict]) -> dict:
    """
    Calculate where price is positioned within recent range.
    
    Args:
        candles: List of candle dictionaries
        
    Returns:
        Dictionary with position features:
        - price_position_pct: Where close is in 20-bar range (0-1)
        - distance_from_high: % distance from 20-bar high
        - distance_from_low: % distance from 20-bar low
    """
    if not candles or len(candles) < 20:
        return {
            'price_position_pct': 0.5,
            'distance_from_high': 0.0,
            'distance_from_low': 0.0
        }
    
    try:
        recent = candles[-20:]
        
        highs = [float(c.get('high', 0)) for c in recent]
        lows = [float(c.get('low', 0)) for c in recent]
        current_close = float(recent[-1].get('close', 0))
        
        high_20 = max(highs)
        low_20 = min(lows)
        
        if high_20 == low_20:
            return {
                'price_position_pct': 0.5,
                'distance_from_high': 0.0,
                'distance_from_low': 0.0
            }
        
        # Position in range (0 = at low, 1 = at high)
        price_position_pct = (current_close - low_20) / (high_20 - low_20)
        
        # Distance from extremes
        distance_from_high = (high_20 - current_close) / current_close if current_close != 0 else 0.0
        distance_from_low = (current_close - low_20) / current_close if current_close != 0 else 0.0
        
        return {
            'price_position_pct': float(price_position_pct),
            'distance_from_high': float(distance_from_high),
            'distance_from_low': float(distance_from_low)
        }
    except Exception:
        return {
            'price_position_pct': 0.5,
            'distance_from_high': 0.0,
            'distance_from_low': 0.0
        }


def calculate_all_enhanced_features(candles: list[dict]) -> dict:
    """
    Calculate all enhanced features in one call.
    
    Args:
        candles: List of candle dictionaries
        
    Returns:
        Dictionary with all enhanced features combined
    """
    features = {}
    
    # Volume features
    features.update(calculate_volume_features(candles))
    
    # Momentum features
    features.update(calculate_price_momentum(candles))
    
    # Volatility features
    features.update(calculate_volatility_features(candles))
    
    # Price position features
    features.update(calculate_price_position(candles))
    
    return features
