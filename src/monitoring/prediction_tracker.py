"""
Prediction Accuracy Tracker - Real-time validation of AI predictions

This module tracks every prediction made by the Oracle/Nexus and compares
it against actual market movement to calculate accuracy metrics.

Author: AETHER Development Team
Version: 1.0.0
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger("PredictionTracker")


@dataclass
class Prediction:
    """Single prediction record."""
    timestamp: float
    prediction: str  # "UP", "DOWN", "NEUTRAL"
    confidence: float
    current_price: float
    symbol: str
    horizon_candles: int = 5  # How many candles ahead we're predicting
    
    # Actual outcome (filled later)
    actual_direction: Optional[str] = None
    actual_price_change: Optional[float] = None
    actual_price_end: Optional[float] = None
    outcome_timestamp: Optional[float] = None
    correct: Optional[bool] = None
    
    # Metadata
    model: str = "Oracle"
    regime: Optional[str] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None


class PredictionTracker:
    """
    Tracks AI predictions and validates them against actual market movement.
    
    Provides real-time accuracy metrics and generates performance reports.
    """
    
    def __init__(self, log_dir: str = "logs/predictions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for pending predictions
        self.pending_predictions: Dict[str, List[Prediction]] = {}
        
        # Accuracy metrics
        self.total_predictions = 0
        self.correct_predictions = 0
        self.by_direction = {
            "UP": {"total": 0, "correct": 0},
            "DOWN": {"total": 0, "correct": 0},
            "NEUTRAL": {"total": 0, "correct": 0}
        }
        
        # Load existing predictions
        self._load_today_predictions()
        
        logger.info(f"PredictionTracker initialized. Log dir: {self.log_dir}")
    
    def record_prediction(
        self,
        symbol: str,
        prediction: str,
        confidence: float,
        current_price: float,
        horizon_candles: int = 5,
        model: str = "Oracle",
        regime: Optional[str] = None,
        rsi: Optional[float] = None,
        atr: Optional[float] = None
    ) -> None:
        """
        Record a new prediction.
        
        Args:
            symbol: Trading symbol
            prediction: "UP", "DOWN", or "NEUTRAL"
            confidence: Prediction confidence (0-1)
            current_price: Current market price
            horizon_candles: Number of candles ahead to predict
            model: Model name (Oracle, Nexus, etc.)
            regime: Market regime
            rsi: RSI value
            atr: ATR value
        """
        pred = Prediction(
            timestamp=time.time(),
            prediction=prediction,
            confidence=confidence,
            current_price=current_price,
            symbol=symbol,
            horizon_candles=horizon_candles,
            model=model,
            regime=regime,
            rsi=rsi,
            atr=atr
        )
        
        # Add to pending list
        if symbol not in self.pending_predictions:
            self.pending_predictions[symbol] = []
        
        self.pending_predictions[symbol].append(pred)
        
        # Save to disk immediately
        self._save_prediction(pred)
        
        logger.debug(
            f"[PREDICTION] {model} predicts {prediction} "
            f"(conf={confidence:.2f}) for {symbol} @ {current_price:.5f}"
        )
    
    def validate_predictions(
        self,
        symbol: str,
        current_price: float,
        current_time: float
    ) -> None:
        """
        Check if any pending predictions can now be validated.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_time: Current timestamp
        """
        if symbol not in self.pending_predictions:
            return
        
        validated = []
        
        for pred in self.pending_predictions[symbol]:
            # Check if enough time has passed (horizon_candles * 60 seconds for M1)
            time_elapsed = current_time - pred.timestamp
            required_time = pred.horizon_candles * 60  # M1 timeframe
            
            if time_elapsed >= required_time:
                # Validate this prediction
                price_change = current_price - pred.current_price
                pct_change = (price_change / pred.current_price) * 100
                
                # Determine actual direction (threshold: 0.01% to avoid noise)
                if pct_change > 0.01:
                    actual_direction = "UP"
                elif pct_change < -0.01:
                    actual_direction = "DOWN"
                else:
                    actual_direction = "NEUTRAL"
                
                # Check if prediction was correct
                correct = (pred.prediction == actual_direction)
                
                # Update prediction
                pred.actual_direction = actual_direction
                pred.actual_price_change = price_change
                pred.actual_price_end = current_price
                pred.outcome_timestamp = current_time
                pred.correct = correct
                
                # Update metrics
                self.total_predictions += 1
                if correct:
                    self.correct_predictions += 1
                
                self.by_direction[pred.prediction]["total"] += 1
                if correct:
                    self.by_direction[pred.prediction]["correct"] += 1
                
                # Save validated prediction
                self._save_validated_prediction(pred)
                
                # Mark for removal from pending
                validated.append(pred)
                
                logger.info(
                    f"[VALIDATION] {pred.model} prediction "
                    f"{'✓ CORRECT' if correct else '✗ WRONG'}: "
                    f"Predicted {pred.prediction}, Actual {actual_direction} "
                    f"({pct_change:+.3f}% in {time_elapsed/60:.1f}min)"
                )
        
        # Remove validated predictions
        for pred in validated:
            self.pending_predictions[symbol].remove(pred)
    
    def get_accuracy_report(self) -> Dict:
        """
        Generate current accuracy metrics.
        
        Returns:
            Dictionary with accuracy statistics
        """
        overall_accuracy = (
            (self.correct_predictions / self.total_predictions * 100)
            if self.total_predictions > 0 else 0.0
        )
        
        direction_accuracy = {}
        for direction, stats in self.by_direction.items():
            if stats["total"] > 0:
                direction_accuracy[direction] = {
                    "accuracy": (stats["correct"] / stats["total"] * 100),
                    "total": stats["total"],
                    "correct": stats["correct"]
                }
            else:
                direction_accuracy[direction] = {
                    "accuracy": 0.0,
                    "total": 0,
                    "correct": 0
                }
        
        return {
            "overall_accuracy": overall_accuracy,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "by_direction": direction_accuracy,
            "pending_validations": sum(
                len(preds) for preds in self.pending_predictions.values()
            )
        }
    
    def print_report(self) -> None:
        """Print formatted accuracy report to console."""
        report = self.get_accuracy_report()
        
        print("\n" + "="*70)
        print("📊 PREDICTION ACCURACY REPORT")
        print("="*70)
        print(f"Overall Accuracy:    {report['overall_accuracy']:.1f}%")
        print(f"Total Predictions:   {report['total_predictions']}")
        print(f"Correct:             {report['correct_predictions']}")
        print(f"Pending Validation:  {report['pending_validations']}")
        print("\nBy Direction:")
        for direction, stats in report['by_direction'].items():
            print(
                f"  {direction:8s}: {stats['accuracy']:5.1f}% "
                f"({stats['correct']}/{stats['total']})"
            )
        print("="*70 + "\n")
    
    def _save_prediction(self, pred: Prediction) -> None:
        """Save prediction to daily log file."""
        today = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"predictions_{today}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(asdict(pred)) + '\n')
    
    def _save_validated_prediction(self, pred: Prediction) -> None:
        """Save validated prediction to results file."""
        today = datetime.now().strftime("%Y%m%d")
        results_file = self.log_dir / f"validated_{today}.jsonl"
        
        with open(results_file, 'a') as f:
            f.write(json.dumps(asdict(pred)) + '\n')
    
    def _load_today_predictions(self) -> None:
        """Load today's validated predictions to restore metrics."""
        today = datetime.now().strftime("%Y%m%d")
        results_file = self.log_dir / f"validated_{today}.jsonl"
        
        if not results_file.exists():
            return
        
        try:
            with open(results_file, 'r') as f:
                for line in f:
                    pred_dict = json.loads(line.strip())
                    if pred_dict.get('correct') is not None:
                        self.total_predictions += 1
                        if pred_dict['correct']:
                            self.correct_predictions += 1
                        
                        direction = pred_dict['prediction']
                        self.by_direction[direction]["total"] += 1
                        if pred_dict['correct']:
                            self.by_direction[direction]["correct"] += 1
            
            logger.info(
                f"Loaded {self.total_predictions} validated predictions from today"
            )
        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")


# Global singleton instance
_tracker_instance = None


def get_prediction_tracker() -> PredictionTracker:
    """Get or create the singleton PredictionTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PredictionTracker()
    return _tracker_instance
