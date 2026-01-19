"""
Model Monitor - AI Performance Tracking
========================================
Tracks AI model prediction accuracy in real-time and detects degradation.

ENHANCEMENT 8: Added January 4, 2026
Purpose: Monitor AI model performance and trigger retraining when needed

Author: AETHER Development Team
Version: 1.0
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("ModelMonitor")


class ModelMonitor:
    """
    Track AI model prediction accuracy in real-time.
    
    ENHANCEMENT 8: Monitors predictions vs actual outcomes
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize model monitor.
        
        Args:
            max_history: Maximum number of predictions to track
        """
        self.predictions = deque(maxlen=max_history)
        self.max_history = max_history
        
        # Performance thresholds
        self.accuracy_threshold = 0.52  # Retrain if below 52%
        self.min_samples = 100  # Need at least 100 predictions
        
        # Tracking
        self.last_accuracy = 0.0
        self.last_check_time = 0.0
        
        logger.info(f"[MODEL MONITOR] Initialized (threshold: {self.accuracy_threshold:.1%}, min_samples: {self.min_samples})")
    
    def record_prediction(self, prediction: str, confidence: float, metadata: Optional[Dict] = None):
        """
        Record a prediction when signal is generated.
        
        Args:
            prediction: Predicted direction ('UP', 'DOWN', 'NEUTRAL')
            confidence: Prediction confidence (0.0-1.0)
            metadata: Optional additional data
        """
        entry = {
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': time.time(),
            'actual': None,  # Will be filled later
            'metadata': metadata or {}
        }
        
        self.predictions.append(entry)
        
        logger.debug(f"[MODEL MONITOR] Recorded prediction: {prediction} ({confidence:.2f})")
    
    def record_outcome(self, timestamp: float, actual_direction: str, profit: float = 0.0):
        """
        Record actual outcome after trade closes.
        
        Args:
            timestamp: Timestamp of the original prediction
            actual_direction: Actual direction ('UP' if profit > 0, 'DOWN' if profit < 0)
            profit: Actual profit/loss
        """
        # Find matching prediction (within 5 minutes)
        for pred in reversed(self.predictions):
            if pred['actual'] is None:  # Not yet matched
                time_diff = abs(pred['timestamp'] - timestamp)
                
                if time_diff < 300:  # Within 5 minutes
                    pred['actual'] = actual_direction
                    pred['profit'] = profit
                    pred['matched_at'] = time.time()
                    
                    logger.debug(f"[MODEL MONITOR] Matched outcome: {actual_direction} (profit: {profit:.2f})")
                    break
    
    def get_accuracy(self) -> float:
        """
        Calculate prediction accuracy.
        
        Returns:
            Accuracy as a percentage (0.0-1.0)
        """
        correct = 0
        total = 0
        
        for pred in self.predictions:
            if pred['actual'] is not None:
                total += 1
                
                # Check if prediction matches actual
                if pred['prediction'] == pred['actual']:
                    correct += 1
                # Also count NEUTRAL as half-correct if it avoided a loss
                elif pred['prediction'] == 'NEUTRAL' and abs(pred.get('profit', 0)) < 0.01:
                    correct += 0.5
        
        accuracy = correct / total if total > 0 else 0.0
        self.last_accuracy = accuracy
        
        return accuracy
    
    def get_confidence_calibration(self) -> Dict[str, float]:
        """
        Check if confidence scores are well-calibrated.
        
        Returns:
            Dict with calibration metrics
        """
        # Group predictions by confidence buckets
        buckets = {
            'low': {'correct': 0, 'total': 0},      # 0.0-0.6
            'medium': {'correct': 0, 'total': 0},   # 0.6-0.75
            'high': {'correct': 0, 'total': 0}      # 0.75-1.0
        }
        
        for pred in self.predictions:
            if pred['actual'] is None:
                continue
            
            conf = pred['confidence']
            correct = (pred['prediction'] == pred['actual'])
            
            if conf < 0.6:
                bucket = 'low'
            elif conf < 0.75:
                bucket = 'medium'
            else:
                bucket = 'high'
            
            buckets[bucket]['total'] += 1
            if correct:
                buckets[bucket]['correct'] += 1
        
        # Calculate accuracy per bucket
        calibration = {}
        for bucket_name, bucket_data in buckets.items():
            if bucket_data['total'] > 0:
                calibration[f'{bucket_name}_accuracy'] = bucket_data['correct'] / bucket_data['total']
            else:
                calibration[f'{bucket_name}_accuracy'] = 0.0
        
        return calibration
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if model needs retraining.
        
        Returns:
            (should_retrain, reason)
        """
        # Need minimum samples
        matched_count = sum(1 for p in self.predictions if p['actual'] is not None)
        
        if matched_count < self.min_samples:
            return False, f"Insufficient data ({matched_count}/{self.min_samples})"
        
        # Check accuracy
        accuracy = self.get_accuracy()
        
        if accuracy < self.accuracy_threshold:
            return True, f"Accuracy degraded to {accuracy:.2%} (threshold: {self.accuracy_threshold:.2%})"
        
        # Check confidence calibration
        calibration = self.get_confidence_calibration()
        
        # High confidence predictions should be accurate
        if calibration.get('high_accuracy', 1.0) < 0.65:
            return True, f"High-confidence predictions only {calibration['high_accuracy']:.2%} accurate"
        
        return False, f"Model performing well ({accuracy:.2%})"
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dict with performance metrics
        """
        matched_count = sum(1 for p in self.predictions if p['actual'] is not None)
        accuracy = self.get_accuracy()
        calibration = self.get_confidence_calibration()
        
        # Calculate average confidence for correct vs incorrect predictions
        correct_confidences = [p['confidence'] for p in self.predictions 
                              if p['actual'] is not None and p['prediction'] == p['actual']]
        incorrect_confidences = [p['confidence'] for p in self.predictions 
                                if p['actual'] is not None and p['prediction'] != p['actual']]
        
        return {
            'total_predictions': len(self.predictions),
            'matched_predictions': matched_count,
            'accuracy': accuracy,
            'calibration': calibration,
            'avg_confidence_correct': sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0,
            'avg_confidence_incorrect': sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0.0,
            'should_retrain': self.should_retrain()[0],
            'retrain_reason': self.should_retrain()[1]
        }
    
    def log_performance_summary(self):
        """Log performance summary to console."""
        summary = self.get_performance_summary()
        
        logger.info(
            f"[MODEL MONITOR] Performance Summary:\n"
            f"  Predictions: {summary['total_predictions']} (Matched: {summary['matched_predictions']})\n"
            f"  Accuracy: {summary['accuracy']:.2%}\n"
            f"  Calibration: Low={summary['calibration'].get('low_accuracy', 0):.2%} | "
            f"Med={summary['calibration'].get('medium_accuracy', 0):.2%} | "
            f"High={summary['calibration'].get('high_accuracy', 0):.2%}\n"
            f"  Avg Confidence: Correct={summary['avg_confidence_correct']:.2f} | "
            f"Incorrect={summary['avg_confidence_incorrect']:.2f}\n"
            f"  Status: {summary['retrain_reason']}"
        )
