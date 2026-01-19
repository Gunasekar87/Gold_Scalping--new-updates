import json, os, time, threading
from dataclasses import asdict, dataclass
from typing import Any, Dict

@dataclass
class DecisionRecord:
    ts: float
    symbol: str
    action: str
    side: str
    price: float
    lots: float
    features: Dict[str, Any]
    context: Dict[str, Any]
    decision: Dict[str, Any]

class TelemetryWriter:
    def __init__(self, root: str = "logs/decisions"):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, record: DecisionRecord):
        day = time.strftime("%Y%m%d")
        path = os.path.join(self.root, f"{day}.jsonl")
        try:
            with self.lock, open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except Exception:
            pass # Never crash on logging
