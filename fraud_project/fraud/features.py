"""
Features Module
Applies derived features from config
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple


def load_features_config(path: str) -> Dict[str, Any]:
    """Load features configuration"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_features_config(df: pd.DataFrame, cfg: Dict[str, Any], global_stats: Dict = None) -> pd.DataFrame:
    """
    Apply derived features from config.
    global_stats: dict with global statistics (e.g., amount percentiles) computed on train set
    """
    df = df.copy()
    
    for feat in cfg.get("derived_features", []):
        t = feat["type"]
        name = feat["name"]
        
        if t == "transform":
            inp = feat["input"]
            op = feat["operation"]
            if op == "log1p":
                df[name] = np.log1p(df[inp].clip(lower=0))
            elif op == "sqrt":
                df[name] = np.sqrt(df[inp].clip(lower=0))
            else:
                raise ValueError(f"Unknown transform operation: {op}")
        
        elif t == "diff":
            a, b = feat["inputs"]
            df[name] = np.abs(df[a] - df[b])
        
        elif t == "ratio":
            num = feat["numerator"]
            den = feat["denominator"]
            eps = feat.get("eps", 1e-6)
            df[name] = df[num] / (df[den] + eps)
        
        elif t == "flag_percentile":
            # Requires global_stats
            inp = feat["input"]
            percentile = feat["percentile"]
            if global_stats and f"{inp}_p{percentile}" in global_stats:
                threshold = global_stats[f"{inp}_p{percentile}"]
            else:
                # Fallback to computing on current data
                threshold = df[inp].quantile(percentile / 100.0)
            df[name] = (df[inp] >= threshold).astype(int)
        
        elif t == "flag_threshold_ratio":
            ratio_feat = feat["ratio_feature"]
            thr = feat["threshold"]
            df[name] = (df[ratio_feat] >= thr).astype(int)
        
        elif t == "flag_comparison":
            left = feat["left"]
            right = feat["right"]
            op = feat["operation"]
            multiplier = feat.get("multiplier", 1.0)
            
            if op == ">":
                df[name] = (df[left] > df[right] * multiplier).astype(int)
            elif op == "<":
                df[name] = (df[left] < df[right] * multiplier).astype(int)
            elif op == ">=":
                df[name] = (df[left] >= df[right] * multiplier).astype(int)
            elif op == "<=":
                df[name] = (df[left] <= df[right] * multiplier).astype(int)
            else:
                raise ValueError(f"Unknown comparison operation: {op}")
        
        elif t == "sum":
            inputs = feat["inputs"]
            df[name] = df[inputs].sum(axis=1)
        
        elif t == "product":
            a, b = feat["inputs"]
            df[name] = df[a] * df[b]
        
        else:
            raise ValueError(f"Unknown feature type: {t}")
    
    return df


def compute_global_stats(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Compute global statistics needed for feature derivation"""
    stats = {}
    
    # Find all flag_percentile features
    for feat in cfg.get("derived_features", []):
        if feat["type"] == "flag_percentile":
            inp = feat["input"]
            percentile = feat["percentile"]
            stats[f"{inp}_p{percentile}"] = df[inp].quantile(percentile / 100.0)
    
    return stats


def prepare_X_y(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    global_stats: Dict = None,
    label_col: str = "label"
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features matrix X and target y
    """
    # Apply derived features
    X = apply_features_config(df, cfg, global_stats)
    
    # Drop unnecessary columns
    drop_cols = cfg.get("drop_features", [])
    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=c)
    
    # Extract target
    if label_col in df.columns:
        y = df[label_col].astype(int)
    else:
        y = None
    
    # Convert categorical features
    cat_cols = cfg.get("categorical_features", [])
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
    
    # Fill missing values
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(0)
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], 0)
    
    return X, y, cat_cols

