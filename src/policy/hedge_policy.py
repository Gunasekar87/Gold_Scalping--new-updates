from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HedgeConfig:
    min_confidence: float
    max_spread_atr: float
    tp_atr_mult: float
    sl_atr_mult: float
    max_stack: int

class HedgePolicy:
    def __init__(self, cfg: HedgeConfig):
        self.cfg = cfg

    def decide(self, features: Dict[str, float], context: Dict[str, Any]) -> Dict[str, Any]:
        reasons = []

        # Hard gates
        spread_atr_val = features.get("spread_atr", 1e9)
        if spread_atr_val > self.cfg.max_spread_atr:
            reasons.append(f"fail:spread_atr={spread_atr_val:.3f}>{self.cfg.max_spread_atr}")
            return {"hedge": False, "confidence": 0.0, "reasons": reasons}

        open_hedges = int(context.get("open_hedges", 0))
        if open_hedges >= self.cfg.max_stack:
            reasons.append(f"fail:max_stack={open_hedges}")
            return {"hedge": False, "confidence": 0.0, "reasons": reasons}

        # Score (weights can be tuned offline)
        w = {
            "regime_trend": 0.20,
            "breakout_quality": 0.20,
            "structure_break": 0.20,
            "drawdown_urgency": 0.25,
            "volume_quality": 0.15,
        }
        vz = float(features.get("vol_z", 0.0))
        vz_clipped = max(-2.0, min(2.0, vz))
        volume_quality = max(0.0, min(1.0, 0.5 + 0.5 * (vz_clipped / 2.0)))

        score = (
            w["regime_trend"]     * float(features.get("regime_trend", 0.0)) +
            w["breakout_quality"] * float(features.get("breakout_quality", 0.0)) +
            w["structure_break"]  * float(features.get("structure_break", 0.0)) +
            w["drawdown_urgency"] * float(features.get("drawdown_urgency", 0.0)) +
            w["volume_quality"]   * volume_quality
        )

        reasons.append(
            f"ok:score={score:.3f} rt={features.get('regime_trend',0.0):.2f} "
            f"bq={features.get('breakout_quality',0.0):.2f} sb={features.get('structure_break',0.0):.2f} "
            f"du={features.get('drawdown_urgency',0.0):.2f} vz={vz:.2f} sa={features.get('spread_atr',0.0):.3f}"
        )

        if score < self.cfg.min_confidence:
            reasons.append(f"fail:min_conf={self.cfg.min_confidence}")
            return {"hedge": False, "confidence": float(score), "reasons": reasons}

        atr  = max(1e-9, float(context["atr"]))
        side = context["side"]
        px   = float(context["price"])

        tp = px - self.cfg.tp_atr_mult * atr if side == "sell" else px + self.cfg.tp_atr_mult * atr
        sl = px + self.cfg.sl_atr_mult * atr if side == "sell" else px - self.cfg.sl_atr_mult * atr

        return {
            "hedge": True,
            "confidence": float(score),
            "tp_price": float(tp),
            "sl_price": float(sl),
            "reasons": reasons,
        }
