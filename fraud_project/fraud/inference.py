"""
Inference Module
Handles model loading and transaction scoring
"""

import json
import pickle
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from .features import load_features_config, apply_features_config
from .thresholds import load_thresholds, select_segment_thresholds
from .explain import explain_decision


class FraudModel:
    """Fraud detection model for scoring transactions"""
    
    def __init__(self, model_meta_path: str):
        """
        Load model from metadata file
        
        Args:
            model_meta_path: Path to model_meta.json
        """
        with open(model_meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        
        # Load configs
        self.features_cfg = load_features_config(self.meta["features_config_path"])
        self.thresholds_cfg = load_thresholds(self.meta["thresholds_config_path"])
        
        # Load global stats for feature derivation
        self.global_stats = self.meta.get("global_stats", {})
        
        # Load feature names (for correct ordering)
        self.feature_names = self.meta.get("feature_names", [])
        
        # Load models
        self.models = []
        for m in self.meta["models"]:
            lgb_model = lgb.Booster(model_file=m["lgbm_path"])
            with open(m["xgb_path"], "rb") as f:
                xgb_model = pickle.load(f)
            self.models.append({"lgbm": lgb_model, "xgb": xgb_model})
        
        self.cat_cols = self.features_cfg.get("categorical_features", [])
        
        print(f"✅ Loaded {len(self.models)} models")
        print(f"   Model: {self.meta['model_name']} v{self.meta['version']}")
        print(f"   CV AUC: {self.meta['cv_auc']:.4f}")
    
    def _prepare_features_single(self, tx_row: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features for a single transaction.
        
        tx_row should contain:
        - Basic fields: amount, last_os, last_phone_model, etc.
        - Pre-computed offline features: cst_amount_mean_past, target_fraud_rate_past_smooth, etc.
        
        Args:
            tx_row: Dict with transaction data and offline features
        
        Returns:
            Feature DataFrame ready for prediction
        """
        df = pd.DataFrame([tx_row])
        
        # Apply derived features from config
        df = apply_features_config(df, self.features_cfg, self.global_stats)
        
        # Drop unnecessary columns
        drop_cols = self.features_cfg.get("drop_features", [])
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=c)
        
        # Convert categorical features
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        
        # Fill missing values
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(0)
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0)
        
        # IMPORTANT: Reorder columns to match training order
        if self.feature_names:
            # Make sure all expected features are present
            for feat in self.feature_names:
                if feat not in df.columns:
                    df[feat] = 0  # Add missing with default
            # Reorder
            df = df[self.feature_names]
        
        return df
    
    def predict_proba(self, tx_row: Dict[str, Any]) -> float:
        """
        Predict fraud probability for a transaction.
        
        Args:
            tx_row: Dict with transaction data
        
        Returns:
            Fraud probability (0-1)
        """
        X = self._prepare_features_single(tx_row)
        
        # Debug: print feature count
        print(f"DEBUG: Features in X: {X.shape[1]}")
        print(f"DEBUG: X columns: {list(X.columns)}")
        
        # Ensemble prediction across all folds
        probs = []
        for m in self.models:
            lgbm = m["lgbm"]
            xgbm = m["xgb"]
            
            # LightGBM prediction
            p_lgb = lgbm.predict(X, num_iteration=lgbm.best_iteration)[0]
            
            # XGBoost prediction
            X_xgb = X.copy()
            for col in X_xgb.select_dtypes(include=["category"]).columns:
                X_xgb[col] = X_xgb[col].cat.codes
            d = xgb.DMatrix(X_xgb)
            p_xgb = xgbm.predict(d, iteration_range=(0, xgbm.best_iteration))[0]
            
            # Average of two models
            probs.append((p_lgb + p_xgb) / 2)
        
        # Average across folds
        proba = float(np.mean(probs))
        return proba
    
    def score_transaction(
        self, 
        tx_row: Dict[str, Any],
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Score a transaction and return decision with explanation.
        
        Args:
            tx_row: Dict with transaction data and offline features
            explain: Whether to include explanation
        
        Returns:
            Dict with decision, probability, segment, thresholds, and optional explanation
        """
        # Get fraud probability
        proba = self.predict_proba(tx_row)
        
        # Select segment and thresholds
        seg_name, thr_fraud, thr_review = select_segment_thresholds(
            tx_row, self.thresholds_cfg
        )
        
        # Make decision
        if proba >= thr_fraud:
            decision = "BLOCK"
        elif proba >= thr_review:
            decision = "REVIEW"
        else:
            decision = "ACCEPT"
        
        result = {
            "decision": decision,
            "probability": proba,
            "segment": seg_name,
            "threshold_fraud": thr_fraud,
            "threshold_review": thr_review,
        }
        
        # Add explanation if requested
        if explain:
            explanation = explain_decision(
                tx_row, proba, decision, seg_name, thr_fraud, thr_review
            )
            result.update(explanation)
        
        return result

