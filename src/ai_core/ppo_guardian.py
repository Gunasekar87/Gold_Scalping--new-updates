import os
import numpy as np
import logging
import json
import time
import traceback

# Suppress TensorFlow warnings before importing stable_baselines3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from stable_baselines3 import PPO
    from gymnasium import Env, spaces
    PPO_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger("PPOGuardian")
    logger.warning(f"PPO import failed: {e}. Using fallback PPO Guardian.")
    PPO_AVAILABLE = False
    # Fallback dummy classes
    class Env:
        pass
    class spaces:
        Box = None

# [PATCH] Fix for FloatSchedule/ConstantSchedule deserialization error in newer SB3 versions
# This restores the missing class so old models can load without warnings
try:
    import stable_baselines3.common.utils
    
    # Define FloatSchedule with robust initialization
    class FloatSchedule:
        """Fixed FloatSchedule for SB3 compatibility."""
        def __init__(self, start_value=1.0):
            # Ensure start_value is always set, even if called with no args
            self.start_value = float(start_value) if start_value is not None else 1.0
            
        def __call__(self, progress=0.0):
            # Return start_value, ensuring it exists
            return getattr(self, 'start_value', 1.0)
        
        def __repr__(self):
            return f"FloatSchedule(start_value={self.start_value})"
    
    # Define ConstantSchedule with robust initialization
    class ConstantSchedule:
        """Fixed ConstantSchedule for SB3 compatibility."""
        def __init__(self, value=1.0):
            # Ensure value is always set
            self.value = float(value) if value is not None else 1.0
            # Also set start_value for compatibility
            self.start_value = self.value
            
        def __call__(self, progress=0.0):
            # Return value, ensuring it exists
            return getattr(self, 'value', getattr(self, 'start_value', 1.0))
        
        def __repr__(self):
            return f"ConstantSchedule(value={self.value})"
    
    # Monkey-patch into stable_baselines3.common.utils
    if not hasattr(stable_baselines3.common.utils, 'FloatSchedule'):
        stable_baselines3.common.utils.FloatSchedule = FloatSchedule
        
    if not hasattr(stable_baselines3.common.utils, 'ConstantSchedule'):
        stable_baselines3.common.utils.ConstantSchedule = ConstantSchedule
    
    # Also patch into the module's __dict__ for pickle compatibility
    import sys
    if 'stable_baselines3.common.utils' in sys.modules:
        sys.modules['stable_baselines3.common.utils'].FloatSchedule = FloatSchedule
        sys.modules['stable_baselines3.common.utils'].ConstantSchedule = ConstantSchedule
        
except (ImportError, AttributeError) as e:
    logger = logging.getLogger("PPOGuardian")
    logger.warning(f"SB3 compatibility patch failed (non-critical): {e}")
    # Continue without patches - old models may show deserialization warnings but will still load

logger = logging.getLogger("PPOGuardian")

class AetherTradingEnv(Env):
    """
    The 'Gym' where the AI learns to trade.
    It simulates the market environment for the PPO agent.
    """
    def __init__(self):
        super(AetherTradingEnv, self).__init__()
        # Action: [Hedge_Multiplier, Zone_Width_Modifier]
        # Hedge Multiplier: 1.0 to 1.5 (Aggression)
        # Zone Modifier: 0.5 to 2.0 (Flexibility)
        self.action_space = spaces.Box(low=np.array([1.0, 0.5]), high=np.array([1.5, 2.0]), dtype=np.float32)
        
        # Observation: [Current_Drawdown_Pips, ATR_Volatility, Trend_Strength, Nexus_Prediction]
        # Nexus_Prediction: -1 (Sell), 0 (Neutral), 1 (Buy)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Memory Buffer for Replay
        self.memory = []
        self.current_step = 0

    def set_memory(self, memory_data):
        self.memory = memory_data
        self.current_step = 0

    def step(self, action):
        # If we are training (evolving), we replay memory
        if len(self.memory) > 0 and self.current_step < len(self.memory):
            data = self.memory[self.current_step]
            self.current_step += 1
            
            obs = np.array(data['obs'], dtype=np.float32)
            reward = data['reward']
            done = (self.current_step >= len(self.memory))
            
            return obs, reward, done, False, {}
            
        # Default / Inference mode
        return np.zeros(4), 0, False, False, {}

    def reset(self, seed=None):
        if len(self.memory) > 0:
            self.current_step = 0
            return np.array(self.memory[0]['obs'], dtype=np.float32), {}
        return np.zeros(4), {}

class PPOGuardian:
    def __init__(self, model_path="models/ppo_guardian.zip"):
        logger.info("Initializing PPO Neural Network...")
        self.model_path = model_path
        self.env = AetherTradingEnv()
        self.memory_file = "data/brain_memory.json"
        
        # ENHANCEMENT 3: Auto-Training Configuration
        self.auto_train_interval = 100  # Train every 100 trades
        self.trades_since_training = 0
        self.min_experiences_for_training = 50  # Minimum experiences needed
        self.last_training_time = 0.0
        
        # Ensure data and models directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if os.path.exists(model_path):
            logger.info(f"Loading trained PPO brain from {model_path}")
            try:
                self.model = PPO.load(model_path, env=self.env)
            except ValueError as e:
                if "Observation spaces do not match" in str(e):
                    logger.warning("[BRAIN UPGRADE] 3 inputs -> 4 inputs. Resetting PPO Brain to adapt to Hive Mind...")
                    # Delete old model and start fresh
                    os.remove(model_path)
                    self.model = PPO("MlpPolicy", self.env, verbose=1)
                    self.model.save(model_path)
                    logger.info("[BRAIN RESET] New Hive Mind Architecture Active.")
                else:
                    raise e
        else:
            logger.warning("No trained model found. Initializing fresh PPO brain (Exploration Mode).")
            self.model = PPO("MlpPolicy", self.env, verbose=1)
            # Save immediately to establish the file
            self.model.save(model_path)
            logger.info(f"Fresh Brain Saved to {model_path}")
            
        logger.info("PPO Guardian Online: Level 5 Self-Learning Active.")
        logger.info(f"[AUTO-TRAIN] Enabled: Will retrain every {self.auto_train_interval} trades")

    def get_dynamic_zone(self, drawdown_pips, atr, trend_strength, nexus_prediction=0.0):
        """
        Asks the AI: "Given this drawdown, volatility, and FUTURE PREDICTION, how wide should the zone be?"
        """
        obs = np.array([drawdown_pips, atr, trend_strength, nexus_prediction], dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Action[0] is Hedge Multiplier, Action[1] is Zone Width Modifier
        return float(action[0]), float(action[1])

    def get_exit_score(self, drawdown_pips, atr, trend_strength, nexus_prediction=0.0):
        """
        Futuristic AI exit scoring for high-intelligence scalping.

        Enhanced with:
        - ATR-normalized analysis
        - Time-based decay factors
        - Volatility-adjusted thresholds
        - Multi-factor intelligence scoring

        Returns a score from 0.0 (hold) to 1.0 (strong exit signal).
        """
        try:
            # Normalize drawdown by ATR for intelligent analysis
            atr_pips = atr * 10000
            normalized_drawdown = drawdown_pips / max(atr_pips, 0.1)  # Avoid division by zero

            # Scalping intelligence factors
            volatility_factor = min(atr_pips / 10.0, 2.0)  # ATR relative to 10 pips
            trend_factor = abs(trend_strength)

            # Multi-factor analysis for scalping
            factors = {
                'drawdown_ratio': normalized_drawdown,
                'volatility': volatility_factor,
                'trend': trend_factor,
                'nexus_confidence': abs(nexus_prediction)
            }

            # Intelligent weighting for scalping decisions
            weights = {
                'drawdown_ratio': 0.5,  # Most important for exits
                'volatility': 0.2,      # Volatility affects timing
                'trend': 0.2,           # Trend strength
                'nexus_confidence': 0.1  # AI prediction confidence
            }

            # Calculate weighted score
            intelligence_score = sum(
                factors[key] * weights[key] for key in factors.keys()
            )

            # Apply sigmoid transformation for smooth 0-1 scaling
            # Sigmoid provides better probability distribution
            exit_score = 1.0 / (1.0 + np.exp(-intelligence_score))

            # Scalping adjustments
            if normalized_drawdown > 1.5:  # 1.5 ATR drawdown
                exit_score = max(exit_score, 0.8)  # Force high exit probability
            elif normalized_drawdown < 0.3:  # Small movement
                exit_score = min(exit_score, 0.2)  # Lower exit probability

            # Log intelligence factors for transparency
            logger.debug(f"[AI_EXIT] Drawdown: {drawdown_pips:.1f}pips | ATR: {atr_pips:.1f}pips | Score: {exit_score:.3f}")

            return float(exit_score)

        except Exception as e:
            logger.warning(f"Exit score calculation failed: {e}. Returning neutral 0.5")
            return 0.5

    def remember(self, obs, action, reward):
        """
        Stores the experience in the short-term memory.
        
        ENHANCEMENT 3: Auto-triggers training every N trades
        """
        # Ensure obs is a list of exactly 4 elements
        if isinstance(obs, np.ndarray):
            obs = obs.tolist()
        if not isinstance(obs, list) or len(obs) != 4:
            logger.warning(f"Invalid obs shape: {obs}. Expected 4 elements. Skipping memory.")
            return
            
        entry = {
            "obs": obs,
            "action": action if isinstance(action, list) else action.tolist() if isinstance(action, np.ndarray) else [action, action],
            "reward": reward,
            "timestamp": time.time()
        }
        
        # Load existing memory
        memory = []
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                try:
                    memory = json.load(f)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to load PPO memory: {e}, starting fresh")
        
        memory.append(entry)
        
        # Keep only last 1000 experiences
        if len(memory) > 1000:
            memory = memory[-1000:]
            
        with open(self.memory_file, "w") as f:
            json.dump(memory, f)
            
        logger.info(f"Brain Memory Updated. Total Experiences: {len(memory)}")
        
        # ENHANCEMENT 3: Auto-Training Trigger
        self.trades_since_training += 1
        
        # Check if we should auto-train
        if (self.trades_since_training >= self.auto_train_interval and 
            len(memory) >= self.min_experiences_for_training):
            
            logger.info(f"[AUTO-TRAIN] Triggering automatic training ({self.trades_since_training} trades since last training)")
            
            # Trigger evolution
            success = self.evolve()
            
            if success:
                self.trades_since_training = 0
                self.last_training_time = time.time()
                logger.info(f"[AUTO-TRAIN] Training completed successfully. Counter reset.")
            else:
                logger.warning(f"[AUTO-TRAIN] Training failed. Will retry at next interval.")
        else:
            # Log progress towards next training
            trades_remaining = self.auto_train_interval - self.trades_since_training
            if self.trades_since_training % 10 == 0:  # Log every 10 trades
                logger.debug(f"[AUTO-TRAIN] Progress: {self.trades_since_training}/{self.auto_train_interval} trades ({trades_remaining} until next training)")

    def evolve(self):
        """
        Triggers a training session on recent data.
        """
        if not os.path.exists(self.memory_file):
            return False
            
        with open(self.memory_file, "r") as f:
            memory = json.load(f)
        
        # CRITICAL: Filter out old 3D experiences (we now use 4D: [Drawdown, ATR, Trend, Nexus])
        valid_memory = []
        for exp in memory:
            if isinstance(exp.get('obs'), list) and len(exp['obs']) == 4:
                valid_memory.append(exp)
            else:
                logger.debug(f"Filtered out old 3D experience: {exp.get('obs', [])}")
        
        # If we filtered out experiences, save the cleaned memory
        if len(valid_memory) < len(memory):
            logger.info(f"[CLEANUP] MEMORY CLEANUP: Removed {len(memory) - len(valid_memory)} outdated 3D experiences")
            with open(self.memory_file, "w") as f:
                json.dump(valid_memory, f, indent=2)
            memory = valid_memory
            
        if len(memory) < 1: # Learn immediately after EVERY trade (Continuous Learning)
            logger.info(f"[NEURO] NEUROPLASTICITY: Accumulating Experience ({len(memory)}/1 trades)... Evolution Pending.")
            return False
            
        logger.info(f"[NEURO] NEUROPLASTICITY: Evolving Brain on {len(memory)} recent experiences...")
        
        # Load memory into environment
        self.env.set_memory(memory)
        
        # 1. Benchmark: What would the AI do in a "High Volatility" scenario right now?
        # Obs: [Drawdown=50 pips, ATR=High (0.0020), Trend=Strong (1.0), Nexus=Neutral (0.0)]
        benchmark_obs = np.array([50.0, 0.0020, 1.0, 0.0], dtype=np.float32)
        action_before, _ = self.model.predict(benchmark_obs, deterministic=True)
        
        try:
            # Train on the replayed memory
            # We set total_timesteps to match memory size roughly
            self.model.learn(total_timesteps=len(memory) * 5) 
            self.model.save(self.model_path)
            
            # 2. Post-Training Check
            action_after, _ = self.model.predict(benchmark_obs, deterministic=True)
            
            # 3. Log the Shift
            change_msg = (
                f"[EVOLUTION COMPLETE] Learning Impact:\n"
                f"   --------------------------------------------------\n"
                f"   [SCENARIO: High Volatility & Drawdown]\n"
                f"   * Previous Policy: Hedge Mult {action_before[0]:.2f} | Zone Mod {action_before[1]:.2f}\n"
                f"   * New Policy:      Hedge Mult {action_after[0]:.2f} | Zone Mod {action_after[1]:.2f}\n"
                f"   * Interpretation:  AI has become {'More Aggressive' if action_after[0] > action_before[0] else 'More Conservative'} "
                f"and {'Widened' if action_after[1] > action_before[1] else 'Tightened'} Zones.\n"
                f"   --------------------------------------------------"
            )
            logger.info(change_msg)
            return True
        except Exception as e:
            logger.error(f"Evolution Failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def get_experience_count(self):
        """
        Returns the total number of experiences stored in memory.
        """
        if not os.path.exists(self.memory_file):
            return 0
        try:
            with open(self.memory_file, "r") as f:
                memory = json.load(f)
                return len(memory)
        except (json.JSONDecodeError, ValueError, IOError) as e:
            logger.warning(f"Failed to read PPO memory count: {e}")
            return 0
    
    def get_position_size_multiplier(self, atr, trend_strength, confidence, current_equity):
        """
        PPO-driven position sizing multiplier for initial trades.
        
        Uses RL agent to optimize lot size based on:
        - Market volatility (ATR)
        - Trend strength
        - AI confidence
        - Account equity
        
        Args:
            atr: Average True Range (volatility measure)
            trend_strength: Trend strength from 0 to 1
            confidence: Signal confidence from AI (0 to 1)
            current_equity: Current account equity
            
        Returns:
            float: Multiplier from 0.5 to 1.5 to adjust base lot size
        """
        try:
            # Normalize inputs for PPO
            atr_pips = atr * 10000  # Convert to pips
            normalized_atr = min(atr_pips / 50.0, 3.0)  # Normalize to 0-3 range
            
            # Create observation for PPO
            # [drawdown (0 for entry), ATR, trend_strength, confidence]
            obs = np.array([0.0, normalized_atr, trend_strength, confidence], dtype=np.float32)
            
            # Get PPO action
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Use hedge multiplier as position size adjustment
            # Action[0] ranges from 1.0 to 1.5
            size_multiplier = float(action[0])
            
            # Apply confidence scaling
            # Low confidence → reduce size, High confidence → keep/increase size
            confidence_factor = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
            adjusted_multiplier = size_multiplier * confidence_factor
            
            # Volatility safety: reduce size in high volatility
            if atr_pips > 100:  # High volatility
                volatility_factor = 0.8
            elif atr_pips > 50:  # Medium volatility
                volatility_factor = 0.9
            else:  # Low volatility
                volatility_factor = 1.0
                
            final_multiplier = adjusted_multiplier * volatility_factor
            
            # Clamp to reasonable range (0.5x to 1.5x)
            final_multiplier = max(0.5, min(1.5, final_multiplier))
            
            logger.debug(f"[PPO_SIZING] ATR: {atr_pips:.1f}pips | Conf: {confidence:.2f} | Multiplier: {final_multiplier:.2f}")
            
            return float(final_multiplier)
            
        except Exception as e:
            logger.warning(f"PPO position sizing failed: {e}. Using default 1.0")
            return 1.0
    
    def should_exit_bucket(self, net_pnl_usd, position_age_seconds, atr, num_positions, 
                          total_volume, account_equity):
        """
        PPO-driven bucket exit decision for hedged positions.
        
        Uses RL agent to decide when to close entire bucket based on:
        - Net P&L (break-even or profit target)
        - Position age (scalping time limits)
        - Market volatility (ATR)
        - Bucket complexity (number of positions)
        
        Args:
            net_pnl_usd: Net profit/loss in USD for entire bucket
            position_age_seconds: Age of oldest position in seconds
            atr: Current ATR (volatility measure)
            num_positions: Number of positions in bucket
            total_volume: Total volume across all positions
            account_equity: Current account equity
            
        Returns:
            Tuple of (should_exit: bool, confidence: float, reason: str)
        """
        try:
            # Calculate key metrics
            atr_pips = atr * 10000
            position_age_minutes = position_age_seconds / 60.0
            
            # Calculate P&L in pips equivalent
            pip_value_per_lot = 10.0  # $10 per pip for standard lot
            net_pips = net_pnl_usd / (total_volume * pip_value_per_lot) if total_volume > 0 else 0
            
            # Normalize for PPO observation
            normalized_drawdown = abs(net_pips) / max(atr_pips, 1.0)
            normalized_atr = min(atr_pips / 50.0, 3.0)
            
            # PPO observation
            obs = np.array([
                normalized_drawdown,
                normalized_atr,
                0.5,  # Neutral trend (bucket is hedged)
                0.0   # No direction bias for hedged bucket
            ], dtype=np.float32)
            
            # Get AI exit score
            exit_score = self.get_exit_score(
                abs(net_pips),
                atr,
                0.5,  # Neutral trend for hedged positions
                0.0   # No prediction bias
            )
            
            # Multi-factor decision logic
            reasons = []
            
            # Factor 1: Profit Target (Primary for hedged buckets)
            if net_pnl_usd > 0:
                # Calculate minimum profit target (0.15 ATR per lot)
                min_profit_target = (0.15 * atr_pips) * total_volume * pip_value_per_lot
                
                if net_pnl_usd >= min_profit_target:
                    reasons.append(f"Profit target reached: ${net_pnl_usd:.2f} >= ${min_profit_target:.2f}")
                    exit_score += 0.3
                elif net_pnl_usd > 0:
                    reasons.append(f"Positive P&L: ${net_pnl_usd:.2f}")
                    exit_score += 0.1
            
            # Factor 2: Time-based Exit (Scalping limit)
            if num_positions == 1:
                time_limit = 3.0  # 3 minutes for single position
            else:
                time_limit = 5.0  # 5 minutes for hedged bucket
                
            if position_age_minutes > time_limit:
                reasons.append(f"Time limit exceeded: {position_age_minutes:.1f}min > {time_limit}min")
                exit_score += 0.2
            
            # Factor 3: Emergency Exit (Only if EXCEEDING max hedges)
            if num_positions > 4:
                reasons.append(f"Max hedge limit exceeded: {num_positions} positions")
                exit_score += 0.4
            
            # Factor 4: Break-even with age (Hedged buckets)
            if num_positions > 1 and net_pnl_usd >= 0 and position_age_minutes > 2.0:
                reasons.append(f"Break-even achieved for {num_positions}-position bucket")
                exit_score += 0.2
            
            # Decision threshold
            should_exit = exit_score > 0.6
            
            reason = " | ".join(reasons) if reasons else "Monitoring bucket"
            
            if should_exit:
                logger.info(f"[PPO_EXIT] [EXIT] SIGNAL: Score={exit_score:.2f} | {reason}")
            else:
                logger.debug(f"[PPO_EXIT] Holding: Score={exit_score:.2f} | Net P&L: ${net_pnl_usd:.2f} | Age: {position_age_minutes:.1f}min")
            
            return should_exit, float(exit_score), reason
            
        except Exception as e:
            logger.warning(f"PPO bucket exit decision failed: {e}. Using conservative hold")
            return False, 0.0, f"Error: {str(e)[:50]}"
    
    def dream_mode(self, epochs=50):
        """
        Intensive Offline Training (Dream Mode).
        Replays memory multiple times to reinforce learning when the market is closed.
        """
        logger.info("[DREAM MODE] Entering Deep Offline Learning...")
        if not os.path.exists(self.memory_file):
            logger.warning("No memory to dream on.")
            return

        with open(self.memory_file, "r") as f:
            try:
                memory = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load PPO memory for dreaming: {e}")
                return
            
        if len(memory) < 10:
            logger.info("Not enough memories to dream.")
            return

        self.env.set_memory(memory)
        
        # Train for more timesteps (Deep Sleep)
        try:
            self.model.learn(total_timesteps=len(memory) * epochs)
            self.model.save(self.model_path)
            logger.info(f"[DREAM MODE] Complete. Processed {len(memory) * epochs} simulated scenarios.")
        except Exception as e:
            logger.error(f"Dream Mode Interrupted: {e}")
