#!/usr/bin/env python3
"""
Training Script
Trains fraud detection model using config
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fraud.training import train_cv_ensemble_with_config


if __name__ == "__main__":
    # Use training config
    config_path = "config/training_config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    print("🚀 Starting training...")
    print(f"   Config: {config_path}\n")
    
    try:
        train_cv_ensemble_with_config(config_path)
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

