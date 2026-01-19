"""
Data Normalization Utilities

This module provides reusable functions for standardizing data structures across the application.
It specifically handles the conversion of custom objects (like Position dataclasses or namedtuples)
into standard dictionaries to ensure compatibility with all system components.
"""

from typing import List, Dict, Any, Union
from dataclasses import asdict, is_dataclass

def normalize_position(pos: Any) -> Dict[str, Any]:
    """
    Convert a single position object/dict into a standard dictionary.
    Handles Dataclasses, NamedTuples, and existing Dictionaries.
    Falls back to vars() or returns the object if conversion fails.
    """
    if hasattr(pos, '_asdict'):
        # NamedTuple or compatible
        return pos._asdict()
    elif is_dataclass(pos):
        # Python Dataclass
        return asdict(pos)
    elif isinstance(pos, dict):
        # Already a dict
        return pos
    else:
        # Generic Object
        try:
            return vars(pos)
        except Exception:
            # Conversion failed, return valid object (best effort)
            return pos

def normalize_positions(positions: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert a list of position objects into a list of standard dictionaries.
    Filters out any items that fail conversion drastically if needed, 
    but currently attempts best-effort persistence.
    """
    if not positions:
        return []
        
    normalized = []
    for p in positions:
        norm = normalize_position(p)
        normalized.append(norm)
    return normalized
