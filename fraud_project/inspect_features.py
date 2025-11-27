#!/usr/bin/env python3
"""
Feature Inspection Script
Shows what features are expected by the model
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fraud.data_loading import load_raw_transactions, load_behavior, join_data
from fraud.feature_store import add_history_features
from fraud.features import load_features_config, prepare_X_y, compute_global_stats


def main():
    print("=" * 80)
    print("FEATURE INSPECTION")
    print("=" * 80)
    
    # Load small sample
    print("\n1. Loading data...")
    df_tr = load_raw_transactions("../transactions.csv")
    df_bh = load_behavior("../customer_behavior.csv")
    data = join_data(df_tr, df_bh)
    data = data.head(1000)  # Just sample
    
    # Add history features
    print("2. Adding history features...")
    data = add_history_features(data)
    
    # Load config
    features_cfg = load_features_config("config/features_config.json")
    global_stats = compute_global_stats(data, features_cfg)
    
    # Prepare features
    print("3. Preparing features...")
    X, y, cat_cols = prepare_X_y(data, features_cfg, global_stats)
    
    print("\n" + "=" * 80)
    print(f"TOTAL FEATURES: {X.shape[1]}")
    print("=" * 80)
    
    print("\n📋 Feature List:")
    for i, col in enumerate(X.columns, 1):
        dtype = "cat" if col in cat_cols else "num"
        print(f"  {i:2d}. {col:40s} [{dtype}]")
    
    # Check which features from data are in X
    print("\n" + "=" * 80)
    print("FEATURES NOT IN X (dropped or derived):")
    print("=" * 80)
    
    data_cols = set(data.columns)
    X_cols = set(X.columns)
    drop_cols = set(features_cfg.get("drop_features", []))
    
    not_in_X = data_cols - X_cols - drop_cols
    if not_in_X:
        for col in sorted(not_in_X):
            print(f"  - {col}")
    else:
        print("  (none)")
    
    # Save feature list
    feature_list = {
        "total_features": X.shape[1],
        "feature_names": list(X.columns),
        "categorical_features": cat_cols,
        "global_stats": global_stats
    }
    
    with open("feature_list.json", "w") as f:
        json.dump(feature_list, f, indent=2)
    
    print("\n✅ Saved feature list to feature_list.json")


if __name__ == "__main__":
    main()

