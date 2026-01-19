import logging
import json
import os
import time
import random
from typing import Dict, Any, List, Optional

logger = logging.getLogger("BayesianTuner")

class BayesianOptimizer:
    """
    Dynamic Hyperparameter Tuner using Bayesian Optimization.
    Adapts AI parameters based on real-time PnL feedback.
    """
    def __init__(
        self,
        state_file: str = "data/optimizer_state.json",
        enable_exploration: Optional[bool] = None,
        max_history: int = 200,
    ):
        self.state_file = state_file
        self.max_history = max_history
        if enable_exploration is None:
            enable_exploration = os.getenv("AETHER_TUNER_EXPLORE", "0").strip().lower() in {"1", "true", "yes"}
        self.enable_exploration = enable_exploration
        self.param_space = {
            "velocity_threshold": (0.55, 0.75), # Oracle Velocity Gate
            "min_viable_move_mult": (1.5, 3.0), # Oracle Spread Multiplier
            "transformer_heads": [2, 4, 8],     # NexusBrain Architecture
            "risk_reward_ratio": (1.0, 3.0)     # Risk Manager
        }
        self.current_params = self._load_defaults()
        self.history: List[Dict[str, Any]] = []
        self._load_state()

    def _load_defaults(self) -> Dict[str, Any]:
        return {
            "velocity_threshold": 0.65,
            "min_viable_move_mult": 2.0,
            "transformer_heads": 4,
            "risk_reward_ratio": 1.5
        }

    def _load_state(self) -> None:
        """Best-effort load of persisted optimizer state."""
        try:
            if not os.path.exists(self.state_file):
                return
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            params = state.get("current_params")
            history = state.get("history")
            if isinstance(params, dict):
                self.current_params.update(params)
            if isinstance(history, list):
                self.history = history[-self.max_history :]
        except Exception as e:
            logger.warning(f"[BAYESIAN] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Best-effort persist of optimizer state."""
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            payload = {
                "current_params": self.current_params,
                "history": self.history[-self.max_history :],
                "timestamp": time.time(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"[BAYESIAN] Failed to save state: {e}")

    def suggest_params(self) -> Dict[str, Any]:
        """
        Returns the current optimal parameters.
        In a full implementation, this uses Gaussian Processes (e.g., Optuna).
        For now, it returns the stable defaults with slight exploration.
        """
        # Exploration logic (Epsilon-Greedy)
        # NOTE: Exploration is DISABLED by default in live trading to avoid randomness/overfitting.
        if self.enable_exploration and random.random() < 0.1: # 10% chance to explore
            # Perturb velocity threshold slightly
            new_vel = self.current_params["velocity_threshold"] + random.uniform(-0.05, 0.05)
            new_vel = max(0.55, min(0.75, new_vel))
            
            logger.info(f"[BAYESIAN] Exploring new velocity threshold: {new_vel:.2f}")
            return {**self.current_params, "velocity_threshold": new_vel}
            
        return self.current_params

    def report_result(self, params: Dict[str, Any], pnl: float):
        """
        Feed back the result of a trading session/trade to the optimizer.
        """
        self.history.append({"params": params, "pnl": pnl, "timestamp": time.time()})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]
        # Persist for post-analysis; learning updates are intentionally conservative.
        self._save_state()
