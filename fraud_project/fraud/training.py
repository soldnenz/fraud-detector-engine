"""
Training Module
Handles model training with cross-validation
"""

import os
import json
import pickle
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold

from .metrics import (
    compute_class_weight, 
    pick_best_threshold_by_f1,
    pick_threshold_by_cost,
    evaluate_predictions,
    print_metrics,
)
from .features import load_features_config, prepare_X_y, compute_global_stats
from .data_loading import load_raw_transactions, load_behavior, join_data, filter_by_window
from .feature_store import add_history_features


# Model hyperparameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "min_gain_to_split": 0.01,
    "seed": 42,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "seed": 42,
    "tree_method": "hist",
}


def train_cv_ensemble_with_config(
    training_cfg_path: str,
) -> None:
    """
    Main training function that loads config and trains models
    """
    
    # Load config
    with open(training_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = json.load(f)
    
    print("=" * 80)
    print(f"TRAINING: {train_cfg['model_name']} v{train_cfg['version']}")
    print("=" * 80)
    
    # Load features config
    features_cfg = load_features_config(train_cfg["features_config_path"])
    
    # 1. Load data
    print("\n📁 Loading data...")
    df_tr = load_raw_transactions(train_cfg["transactions_path"])
    df_bh = load_behavior(train_cfg["behavior_path"])
    data = join_data(df_tr, df_bh)
    data = filter_by_window(data, train_cfg)
    
    print(f"   Loaded {len(data):,} transactions")
    print(f"   Fraud rate: {data['label'].mean():.4f}")
    
    # 2. Add heavy history features
    print("\n🔧 Creating history features...")
    data = add_history_features(data)
    
    # 3. Compute global stats (for percentiles, etc.)
    print("📊 Computing global statistics...")
    global_stats = compute_global_stats(data, features_cfg)
    
    # 4. Prepare X, y
    print("🎯 Preparing feature matrix...")
    X, y, cat_cols = prepare_X_y(data, features_cfg, global_stats)
    amounts = data["amount"]
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {len(X):,}")
    
    # 5. Train with CV
    print("\n" + "=" * 80)
    print(f"TRAINING WITH {train_cfg['n_folds']}-FOLD STRATIFIED CV")
    print("=" * 80)
    
    n_folds = train_cfg.get("n_folds", 5)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    spw = compute_class_weight(y)
    print(f"Scale pos weight: {spw:.3f}\n")
    
    all_oof = np.zeros(len(y))
    fold_models: List[Dict[str, Any]] = []
    fold_metrics = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"{'=' * 60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'=' * 60}")
        
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        amounts_val = amounts.iloc[val_idx]
        
        print(f"Train: {len(y_tr):,} | Val: {len(y_val):,}")
        print(f"Train fraud rate: {y_tr.mean():.4f} | Val fraud rate: {y_val.mean():.4f}")
        
        # Train LightGBM
        print("\n--- LightGBM ---")
        params_lgbm = LGBM_PARAMS.copy()
        params_lgbm["scale_pos_weight"] = spw
        
        dtrain_lgb = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dval_lgb = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)
        
        lgbm_model = lgb.train(
            params_lgbm,
            dtrain_lgb,
            num_boost_round=1000,
            valid_sets=[dval_lgb],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)],
        )
        
        # Train XGBoost
        print("--- XGBoost ---")
        params_xgb = XGB_PARAMS.copy()
        params_xgb["scale_pos_weight"] = spw
        
        X_tr_xgb = X_tr.copy()
        X_val_xgb = X_val.copy()
        for col in X_tr_xgb.select_dtypes(include=["category"]).columns:
            X_tr_xgb[col] = X_tr_xgb[col].cat.codes
            X_val_xgb[col] = X_val_xgb[col].cat.codes
        
        dtrain_xgb = xgb.DMatrix(X_tr_xgb, label=y_tr)
        dval_xgb = xgb.DMatrix(X_val_xgb, label=y_val)
        
        xgb_model = xgb.train(
            params_xgb,
            dtrain_xgb,
            num_boost_round=1000,
            evals=[(dval_xgb, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        
        # Ensemble predictions
        lgb_proba = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)
        xgb_proba = xgb_model.predict(dval_xgb, iteration_range=(0, xgb_model.best_iteration))
        
        fold_proba = (lgb_proba + xgb_proba) / 2
        all_oof[val_idx] = fold_proba
        
        # Evaluate fold
        thr = pick_best_threshold_by_f1(y_val, fold_proba)
        met = evaluate_predictions(
            y_val, fold_proba, thr, amounts_val,
            fp_cost_ratio=train_cfg["fp_cost_ratio"],
            fn_cost_ratio=train_cfg["fn_cost_ratio"]
        )
        
        print(f"\nFold {fold+1} Results:")
        print_metrics(f"Fold {fold+1}", met)
        
        fold_metrics.append(met)
        fold_models.append({"lgbm": lgbm_model, "xgb": xgb_model})
    
    # Overall metrics
    print("\n" + "=" * 80)
    print("OVERALL CV RESULTS")
    print("=" * 80)
    
    cv_thr = pick_best_threshold_by_f1(y, all_oof)
    cv_metrics = evaluate_predictions(
        y, all_oof, cv_thr, amounts,
        fp_cost_ratio=train_cfg["fp_cost_ratio"],
        fn_cost_ratio=train_cfg["fn_cost_ratio"]
    )
    print_metrics("CV Overall", cv_metrics)
    
    # Cost-based threshold
    cost_thr = pick_threshold_by_cost(
        y, all_oof, amounts,
        fp_cost_ratio=train_cfg["fp_cost_ratio"],
        fn_cost_ratio=train_cfg["fn_cost_ratio"]
    )
    cost_metrics = evaluate_predictions(
        y, all_oof, cost_thr, amounts,
        fp_cost_ratio=train_cfg["fp_cost_ratio"],
        fn_cost_ratio=train_cfg["fn_cost_ratio"]
    )
    print(f"\n💰 Cost-Optimized Threshold: {cost_thr:.4f}")
    print_metrics("CV Cost-Optimized", cost_metrics)
    
    # 6. Save models and metadata
    print("\n" + "=" * 80)
    print("SAVING MODELS")
    print("=" * 80)
    
    models_dir = train_cfg["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    
    meta_models = []
    for i, fm in enumerate(fold_models):
        lgb_path = os.path.join(models_dir, f"lgbm_fold{i}.txt")
        xgb_path = os.path.join(models_dir, f"xgb_fold{i}.pkl")
        
        fm["lgbm"].save_model(lgb_path)
        with open(xgb_path, "wb") as f:
            pickle.dump(fm["xgb"], f)
        
        meta_models.append({"fold": i, "lgbm_path": lgb_path, "xgb_path": xgb_path})
        print(f"✅ Saved fold {i} models")
    
    # Save metadata
    model_meta = {
        "model_name": train_cfg["model_name"],
        "version": train_cfg["version"],
        "train_end_date": train_cfg["train_end_date"],
        "train_window_days": train_cfg["train_window_days"],
        "cv_auc": cv_metrics.roc_auc,
        "cv_f1": cv_metrics.f1,
        "cv_threshold": cv_thr,
        "cost_threshold": cost_thr,
        "features_config_path": train_cfg["features_config_path"],
        "thresholds_config_path": train_cfg["thresholds_config_path"],
        "n_folds": n_folds,
        "n_features": X.shape[1],
        "feature_names": list(X.columns),  # IMPORTANT: feature order!
        "models": meta_models,
        "global_stats": global_stats,  # Save for inference
    }
    
    meta_path = os.path.join(models_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved metadata to {meta_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"🎯 CV AUC: {cv_metrics.roc_auc:.4f}")
    print(f"📊 CV F1: {cv_metrics.f1:.3f}")
    print(f"💰 Cost-optimized F1: {cost_metrics.f1:.3f}")
    if cost_metrics.business_metrics:
        print(f"💵 Total Cost: ₸{cost_metrics.business_metrics.total_cost:,.0f}")

