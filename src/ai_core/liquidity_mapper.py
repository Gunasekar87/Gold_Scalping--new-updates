"""
Liquidity Mapper - AI Layer 12
===============================
Maps institutional order zones and liquidity pools.

This module identifies key support/resistance levels where institutional
orders are likely clustered, helping to avoid "no-man's land" entries.

Author: AETHER Development Team
Version: 1.0
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("LiquidityMapper")


class ZoneType(Enum):
    """Liquidity zone types."""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    NEUTRAL = "neutral"


@dataclass
class LiquidityZone:
    """Represents a liquidity zone."""
    price: float
    strength: float  # 0.0 to 1.0
    zone_type: ZoneType
    volume: float
    touches: int  # Number of times price touched this level


@dataclass
class LiquidityMap:
    """Complete liquidity analysis."""
    zones: List[LiquidityZone]
    current_position: str  # 'AT_SUPPORT', 'AT_RESISTANCE', 'BETWEEN', 'ABOVE_ALL', 'BELOW_ALL'
    nearest_support: Optional[LiquidityZone]
    nearest_resistance: Optional[LiquidityZone]
    recommendation: str


class LiquidityMapper:
    """
    Maps liquidity zones using volume profile analysis.
    
    Methods:
    1. Volume clustering (high volume = liquidity zone)
    2. Price level touch count (multiple touches = strong level)
    3. Wick rejection analysis (wicks = liquidity absorption)
    """
    
    def __init__(self):
        # Zone cache
        self._zone_cache = {}
        self._cache_time = 0
        self._cache_ttl = 300  # 5 minutes
        
        # Configuration
        self.min_touches = 2  # Minimum touches to be significant
        self.zone_proximity_pct = 0.001  # 0.1% proximity = at zone
        self.min_strength = 0.5  # Minimum strength to consider
        
        logger.info("[LIQUIDITY MAPPER] AI Layer 12 Online")
    
    def map_liquidity(self, candles: List[Dict], current_price: float) -> LiquidityMap:
        """
        Map liquidity zones from candle data.
        
        Args:
            candles: Recent candle data (at least 100)
            current_price: Current market price
            
        Returns:
            LiquidityMap with zone analysis
        """
        if not candles or len(candles) < 20:
            return LiquidityMap(
                zones=[],
                current_position='UNKNOWN',
                nearest_support=None,
                nearest_resistance=None,
                recommendation="Insufficient data"
            )
        
        # Detect liquidity zones
        zones = self._detect_zones(candles)
        
        # Filter by strength
        significant_zones = [z for z in zones if z.strength >= self.min_strength]
        
        # Sort by price
        significant_zones.sort(key=lambda z: z.price)
        
        # Determine current position
        current_position = self._determine_position(significant_zones, current_price)
        
        # Find nearest zones
        nearest_support = self._find_nearest_support(significant_zones, current_price)
        nearest_resistance = self._find_nearest_resistance(significant_zones, current_price)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            current_position, nearest_support, nearest_resistance, current_price
        )
        
        return LiquidityMap(
            zones=significant_zones,
            current_position=current_position,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            recommendation=recommendation
        )
    
    def _detect_zones(self, candles: List[Dict]) -> List[LiquidityZone]:
        """
        Detect liquidity zones from candle data.
        
        Returns:
            List of LiquidityZone objects
        """
        zones = []
        
        # Method 1: High/Low clustering
        price_levels = defaultdict(lambda: {'touches': 0, 'volume': 0.0, 'type': []})
        
        for candle in candles[-100:]:  # Last 100 candles
            try:
                high = float(candle.get('high', 0))
                low = float(candle.get('low', 0))
                volume = float(candle.get('tick_volume', 1))
                
                # Round to nearest significant level (e.g., $1 for Gold)
                high_level = round(high)
                low_level = round(low)
                
                # Record touches
                price_levels[high_level]['touches'] += 1
                price_levels[high_level]['volume'] += volume
                price_levels[high_level]['type'].append('resistance')
                
                price_levels[low_level]['touches'] += 1
                price_levels[low_level]['volume'] += volume
                price_levels[low_level]['type'].append('support')
            except (KeyError, ValueError, TypeError):
                continue
        
        # Convert to LiquidityZone objects
        for price, data in price_levels.items():
            if data['touches'] >= self.min_touches:
                # Determine zone type
                support_count = data['type'].count('support')
                resistance_count = data['type'].count('resistance')
                
                if support_count > resistance_count:
                    zone_type = ZoneType.SUPPORT
                elif resistance_count > support_count:
                    zone_type = ZoneType.RESISTANCE
                else:
                    zone_type = ZoneType.NEUTRAL
                
                # Calculate strength (based on touches and volume)
                touch_strength = min(data['touches'] / 10.0, 1.0)  # Max at 10 touches
                volume_strength = min(data['volume'] / 1000.0, 1.0)  # Normalize volume
                strength = (touch_strength * 0.7) + (volume_strength * 0.3)
                
                zones.append(LiquidityZone(
                    price=float(price),
                    strength=strength,
                    zone_type=zone_type,
                    volume=data['volume'],
                    touches=data['touches']
                ))
        
        return zones
    
    def _determine_position(self, zones: List[LiquidityZone], 
                           current_price: float) -> str:
        """Determine current price position relative to zones."""
        if not zones:
            return 'UNKNOWN'
        
        # Check if at a zone
        for zone in zones:
            distance_pct = abs(current_price - zone.price) / current_price
            if distance_pct < self.zone_proximity_pct:
                if zone.zone_type == ZoneType.SUPPORT:
                    return 'AT_SUPPORT'
                elif zone.zone_type == ZoneType.RESISTANCE:
                    return 'AT_RESISTANCE'
                else:
                    return 'AT_NEUTRAL_ZONE'
        
        # Check if above all or below all
        highest_zone = max(zones, key=lambda z: z.price)
        lowest_zone = min(zones, key=lambda z: z.price)
        
        if current_price > highest_zone.price:
            return 'ABOVE_ALL'
        elif current_price < lowest_zone.price:
            return 'BELOW_ALL'
        else:
            return 'BETWEEN'
    
    def _find_nearest_support(self, zones: List[LiquidityZone], 
                             current_price: float) -> Optional[LiquidityZone]:
        """Find nearest support zone below current price."""
        support_zones = [z for z in zones 
                        if z.zone_type == ZoneType.SUPPORT and z.price < current_price]
        
        if support_zones:
            return max(support_zones, key=lambda z: z.price)
        return None
    
    def _find_nearest_resistance(self, zones: List[LiquidityZone],
                                current_price: float) -> Optional[LiquidityZone]:
        """Find nearest resistance zone above current price."""
        resistance_zones = [z for z in zones 
                           if z.zone_type == ZoneType.RESISTANCE and z.price > current_price]
        
        if resistance_zones:
            return min(resistance_zones, key=lambda z: z.price)
        return None
    
    def _generate_recommendation(self, position: str, 
                                nearest_support: Optional[LiquidityZone],
                                nearest_resistance: Optional[LiquidityZone],
                                current_price: float) -> str:
        """Generate trading recommendation based on liquidity map."""
        if position == 'AT_SUPPORT':
            return "At support - good for BUY entries"
        elif position == 'AT_RESISTANCE':
            return "At resistance - good for SELL entries"
        elif position == 'BETWEEN':
            # Calculate distance to nearest zones
            if nearest_support and nearest_resistance:
                support_dist = current_price - nearest_support.price
                resistance_dist = nearest_resistance.price - current_price
                total_dist = support_dist + resistance_dist
                
                # If in middle third (no-man's land)
                if 0.33 < (support_dist / total_dist) < 0.67:
                    return "In no-man's land - wait for support/resistance"
                elif support_dist < resistance_dist:
                    return f"Near support (${nearest_support.price:.2f}) - consider BUY"
                else:
                    return f"Near resistance (${nearest_resistance.price:.2f}) - consider SELL"
            return "Between zones - use caution"
        elif position == 'ABOVE_ALL':
            return "Above all resistance - strong bullish, but watch for reversal"
        elif position == 'BELOW_ALL':
            return "Below all support - strong bearish, but watch for reversal"
        else:
            return "Normal operation"
    
    def should_avoid_entry(self, current_price: float, 
                          trade_direction: str,
                          liquidity_map: LiquidityMap) -> Tuple[bool, str]:
        """
        Determine if entry should be avoided based on liquidity.
        
        Args:
            current_price: Current market price
            trade_direction: "BUY" or "SELL"
            liquidity_map: Current liquidity map
            
        Returns:
            (should_avoid, reason)
        """
        # Avoid entries in no-man's land
        if liquidity_map.current_position == 'BETWEEN':
            if liquidity_map.nearest_support and liquidity_map.nearest_resistance:
                support_dist = current_price - liquidity_map.nearest_support.price
                resistance_dist = liquidity_map.nearest_resistance.price - current_price
                total_dist = support_dist + resistance_dist
                
                # If in middle third
                if 0.33 < (support_dist / total_dist) < 0.67:
                    return True, "In no-man's land between liquidity zones"
        
        # Avoid BUY at strong resistance
        if trade_direction == "BUY" and liquidity_map.current_position == 'AT_RESISTANCE':
            if liquidity_map.nearest_resistance and liquidity_map.nearest_resistance.strength > 0.7:
                return True, f"At strong resistance (${liquidity_map.nearest_resistance.price:.2f})"
        
        # Avoid SELL at strong support
        if trade_direction == "SELL" and liquidity_map.current_position == 'AT_SUPPORT':
            if liquidity_map.nearest_support and liquidity_map.nearest_support.strength > 0.7:
                return True, f"At strong support (${liquidity_map.nearest_support.price:.2f})"
        
        return False, "Liquidity OK"
