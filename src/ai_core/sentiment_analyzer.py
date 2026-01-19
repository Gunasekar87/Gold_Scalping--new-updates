"""
Sentiment Analyzer - AI Layer 10
=================================
Analyzes market sentiment from order flow and tick pressure.

This module provides real-time sentiment analysis to enhance trading decisions
by detecting bullish/bearish bias in the market.

Author: AETHER Development Team
Version: 1.0
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("SentimentAnalyzer")


class SentimentState(Enum):
    """Market sentiment states."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class SentimentSignal:
    """Sentiment analysis result."""
    sentiment: SentimentState
    score: float  # -1.0 (bearish) to +1.0 (bullish)
    confidence: float  # 0.0 to 1.0
    source: str  # 'order_flow', 'tick_pressure', 'combined'
    details: str


class SentimentAnalyzer:
    """
    Analyzes market sentiment from multiple sources.
    
    Sources:
    1. Order Book Imbalance (OBI) trends
    2. Tick Pressure (buy vs sell aggression)
    3. Volume-weighted sentiment
    """
    
    def __init__(self, tick_pressure_analyzer=None):
        self.tick_pressure = tick_pressure_analyzer
        
        # Sentiment history (rolling 5-minute window)
        self._sentiment_history = deque(maxlen=300)  # 5 min at 1 sample/sec
        
        # OBI history
        self._obi_history = deque(maxlen=100)
        
        # Configuration
        self.strong_threshold = 0.7  # >0.7 = strong bullish, <-0.7 = strong bearish
        self.neutral_threshold = 0.2  # -0.2 to 0.2 = neutral
        
        logger.info("[SENTIMENT ANALYZER] AI Layer 10 Online")
    
    def analyze(self, tick_data: Dict, obi_score: float = 0.0) -> SentimentSignal:
        """
        Analyze current market sentiment.
        
        Args:
            tick_data: Current tick information
            obi_score: Order Book Imbalance score (-1 to 1)
            
        Returns:
            SentimentSignal with analysis results
        """
        # Collect sentiment from multiple sources
        sources = []
        
        # Source 1: Order Book Imbalance
        if obi_score != 0.0:
            self._obi_history.append(obi_score)
            obi_sentiment = self._analyze_obi_trend()
            sources.append(('obi', obi_sentiment, 0.4))  # 40% weight
        
        # Source 2: Tick Pressure
        if self.tick_pressure:
            pressure_sentiment = self._analyze_tick_pressure()
            sources.append(('pressure', pressure_sentiment, 0.6))  # 60% weight
        
        # Combine sources with weighted average
        if sources:
            total_weight = sum(w for _, _, w in sources)
            combined_score = sum(score * w for _, score, w in sources) / total_weight
            
            # Calculate confidence based on agreement
            confidence = self._calculate_confidence(sources)
            
            # Determine sentiment state
            sentiment_state = self._score_to_state(combined_score)
            
            # Build details
            details = self._build_details(sources, combined_score)
            
            # Store in history
            self._sentiment_history.append({
                'time': time.time(),
                'score': combined_score,
                'state': sentiment_state
            })
            
            return SentimentSignal(
                sentiment=sentiment_state,
                score=combined_score,
                confidence=confidence,
                source='combined',
                details=details
            )
        else:
            # No data available
            return SentimentSignal(
                sentiment=SentimentState.NEUTRAL,
                score=0.0,
                confidence=0.0,
                source='none',
                details="Insufficient data"
            )
    
    def _analyze_obi_trend(self) -> float:
        """
        Analyze Order Book Imbalance trend.
        
        Returns:
            Sentiment score (-1 to 1)
        """
        if len(self._obi_history) < 5:
            return 0.0
        
        # Calculate moving average of OBI
        recent_obi = list(self._obi_history)[-20:]  # Last 20 samples
        avg_obi = sum(recent_obi) / len(recent_obi)
        
        # Calculate trend (is OBI increasing or decreasing?)
        if len(recent_obi) >= 10:
            first_half = sum(recent_obi[:10]) / 10
            second_half = sum(recent_obi[10:]) / (len(recent_obi) - 10)
            trend = second_half - first_half
        else:
            trend = 0.0
        
        # Combine average and trend
        sentiment = (avg_obi * 0.7) + (trend * 0.3)
        
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, sentiment))
    
    def _analyze_tick_pressure(self) -> float:
        """
        Analyze tick pressure sentiment.
        
        Returns:
            Sentiment score (-1 to 1)
        """
        try:
            metrics = self.tick_pressure.get_pressure_metrics()
            
            # Get pressure score
            pressure_score = metrics.get('pressure_score', 0.0)
            
            # Normalize to [-1, 1]
            # Assuming pressure_score is typically in range [-20, 20]
            normalized = pressure_score / 20.0
            
            return max(-1.0, min(1.0, normalized))
        except Exception as e:
            logger.debug(f"[SENTIMENT] Error analyzing tick pressure: {e}")
            return 0.0
    
    def _calculate_confidence(self, sources: List[Tuple[str, float, float]]) -> float:
        """
        Calculate confidence based on agreement between sources.
        
        High confidence = sources agree
        Low confidence = sources disagree
        """
        if len(sources) < 2:
            return 0.5  # Medium confidence with single source
        
        scores = [score for _, score, _ in sources]
        
        # Calculate standard deviation
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Low std dev = high agreement = high confidence
        # High std dev = low agreement = low confidence
        confidence = 1.0 - min(std_dev, 1.0)
        
        return confidence
    
    def _score_to_state(self, score: float) -> SentimentState:
        """Convert sentiment score to state."""
        if score > self.strong_threshold:
            return SentimentState.STRONG_BULLISH
        elif score > self.neutral_threshold:
            return SentimentState.BULLISH
        elif score < -self.strong_threshold:
            return SentimentState.STRONG_BEARISH
        elif score < -self.neutral_threshold:
            return SentimentState.BEARISH
        else:
            return SentimentState.NEUTRAL
    
    def _build_details(self, sources: List[Tuple[str, float, float]], 
                      combined_score: float) -> str:
        """Build human-readable details string."""
        parts = []
        
        for source_name, score, weight in sources:
            parts.append(f"{source_name}={score:.2f}({weight:.0%})")
        
        return f"Combined: {combined_score:.2f} | " + ", ".join(parts)
    
    def get_sentiment_alignment(self, trade_direction: str) -> Tuple[bool, float]:
        """
        Check if trade direction aligns with current sentiment.
        
        Args:
            trade_direction: "BUY" or "SELL"
            
        Returns:
            (is_aligned, alignment_strength)
        """
        if not self._sentiment_history:
            return True, 0.5  # Neutral - allow trade
        
        # Get recent sentiment
        recent = self._sentiment_history[-1]
        score = recent['score']
        
        # Check alignment
        if trade_direction == "BUY":
            # Bullish sentiment aligns with BUY
            is_aligned = score > -self.neutral_threshold
            alignment_strength = max(0.0, score)  # 0 to 1
        else:  # SELL
            # Bearish sentiment aligns with SELL
            is_aligned = score < self.neutral_threshold
            alignment_strength = max(0.0, -score)  # 0 to 1
        
        return is_aligned, alignment_strength
    
    def should_boost_confidence(self, trade_direction: str, 
                               base_confidence: float) -> float:
        """
        Boost trade confidence if sentiment aligns.
        
        Args:
            trade_direction: "BUY" or "SELL"
            base_confidence: Original confidence (0-1)
            
        Returns:
            Boosted confidence
        """
        is_aligned, alignment_strength = self.get_sentiment_alignment(trade_direction)
        
        if is_aligned and alignment_strength > 0.5:
            # Strong alignment - boost confidence by up to 10%
            boost = alignment_strength * 0.1
            boosted = min(1.0, base_confidence + boost)
            
            logger.info(f"[SENTIMENT] {trade_direction} aligns with sentiment "
                       f"(strength: {alignment_strength:.0%}) - Boosting confidence "
                       f"{base_confidence:.0%} → {boosted:.0%}")
            
            return boosted
        
        return base_confidence
