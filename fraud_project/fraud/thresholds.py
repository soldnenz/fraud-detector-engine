"""
Thresholds Module
Handles segment-based threshold selection
"""

import json
from typing import Dict, Any, Tuple


def load_thresholds(path: str) -> Dict[str, Any]:
    """Load thresholds configuration"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_segment_thresholds(
    tx_row: Dict[str, Any], 
    cfg: Dict[str, Any]
) -> Tuple[str, float, float]:
    """
    Select thresholds based on transaction segment.
    
    Returns:
        (segment_name, threshold_fraud, threshold_review)
    """
    
    for seg in cfg["segments"]:
        cond = seg["condition"]
        
        # Create local scope with transaction data
        local_vars = dict(tx_row)
        
        try:
            # Evaluate condition
            if eval(cond, {"__builtins__": {}}, local_vars):
                return seg["name"], seg["threshold_fraud"], seg["threshold_review"]
        except Exception as e:
            # If condition fails, continue to next segment
            continue
    
    # Fallback to default segment
    default = cfg["segments"][-1]
    return default["name"], default["threshold_fraud"], default["threshold_review"]

