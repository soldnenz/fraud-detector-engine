"""
ULTIMATE Fraud Detection на PaySim датасете (6+ млн транзакций)

Обучение на большом синтетическом датасете мобильных платежей.
Гетерогенный ансамбль: LightGBM + XGBoost с продвинутым feature engineering.

Dataset: PS_20174392719_1491204439457_log.csv
- 6+ million transactions
- Highly imbalanced (fraud rate ~0.13%)
- Multiple transaction types
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    confusion_matrix,
)
import lightgbm as lgb
import xgboost as xgb

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

PAYSIM_FILE = "PS_20174392719_1491204439457_log.csv"

# Sampling for faster training (set to 1.0 for full dataset)
SAMPLE_FRACTION = 1  # Use 30% of data for faster training

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 3.0,
    "min_gain_to_split": 0.01,
    "seed": 42,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 3.0,
    "seed": 42,
    "tree_method": "hist",
}


@dataclass
class BusinessMetrics:
    """Бизнес-метрики по суммам транзакций"""
    total_fraud_amount: float
    blocked_fraud_amount: float
    missed_fraud_amount: float
    blocked_legit_amount: float
    fraud_prevention_rate: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Metrics:
    roc_auc: float
    precision: float
    recall: float
    f1: float
    fbeta_05: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    business_metrics: Optional[BusinessMetrics] = None

    def to_dict(self):
        d = asdict(self)
        if self.business_metrics:
            d['business_metrics'] = self.business_metrics.to_dict()
        return d


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def compute_class_weight(y: pd.Series) -> float:
    """scale_pos_weight = #neg / #pos"""
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return neg / (pos + 1e-6)


def pick_best_threshold_by_f1(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Подбор порога по максимальному F1"""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1_arr = 2 * prec * rec / (prec + rec + 1e-9)
    if len(thr) == 0:
        return 0.5
    best_idx = int(np.argmax(f1_arr[:-1]))
    return float(thr[best_idx])


def compute_business_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    amounts: pd.Series,
) -> BusinessMetrics:
    """Вычисление бизнес-метрик по суммам транзакций"""
    fraud_mask = y_true == 1
    pred_fraud_mask = y_pred == 1
    
    total_fraud_amount = amounts[fraud_mask].sum()
    
    tp_mask = fraud_mask & pred_fraud_mask
    blocked_fraud_amount = amounts[tp_mask].sum()
    
    fn_mask = fraud_mask & (~pred_fraud_mask)
    missed_fraud_amount = amounts[fn_mask].sum()
    
    fp_mask = (~fraud_mask) & pred_fraud_mask
    blocked_legit_amount = amounts[fp_mask].sum()
    
    fraud_prevention_rate = (
        blocked_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0.0
    )
    
    return BusinessMetrics(
        total_fraud_amount=float(total_fraud_amount),
        blocked_fraud_amount=float(blocked_fraud_amount),
        missed_fraud_amount=float(missed_fraud_amount),
        blocked_legit_amount=float(blocked_legit_amount),
        fraud_prevention_rate=float(fraud_prevention_rate),
    )


def evaluate_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    amounts: Optional[pd.Series] = None,
    beta: float = 0.5,
) -> Metrics:
    """Оценка предсказаний с бизнес-метриками"""
    y_pred = (y_proba >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    business_metrics = None
    if amounts is not None:
        business_metrics = compute_business_metrics(y_true, y_pred, amounts)
    
    return Metrics(
        roc_auc=roc,
        precision=prec,
        recall=rec,
        f1=f1,
        fbeta_05=fbeta,
        threshold=float(threshold),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        business_metrics=business_metrics,
    )


def print_metrics(label: str, m: Metrics):
    print(
        f"[{label}] ROC-AUC: {m.roc_auc:.4f} | "
        f"P: {m.precision:.3f} | R: {m.recall:.3f} | "
        f"F1: {m.f1:.3f} | F0.5: {m.fbeta_05:.3f} | thr: {m.threshold:.4f}"
    )
    print(
        f"    Confusion: TP={m.tp}, FP={m.fp}, TN={m.tn}, FN={m.fn}"
    )
    if m.business_metrics:
        bm = m.business_metrics
        print(f"    💰 Business:")
        print(f"       Total fraud: ${bm.total_fraud_amount:,.0f}")
        print(f"       ✅ Blocked: ${bm.blocked_fraud_amount:,.0f} ({bm.fraud_prevention_rate*100:.1f}%)")
        print(f"       ❌ Missed: ${bm.missed_fraud_amount:,.0f}")
        print(f"       ⚠️  FP cost: ${bm.blocked_legit_amount:,.0f}")


# ------------------------------------------------------------------------------
# Data Loading and Feature Engineering for PaySim
# ------------------------------------------------------------------------------

def load_and_prepare_paysim(sample_frac: float = 1.0) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Загружает PaySim датасет и создает продвинутые фичи.
    
    Columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
             nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
    """
    print(f"Loading PaySim dataset from {PAYSIM_FILE}...")
    print(f"Sample fraction: {sample_frac:.1%}")
    
    # Read with sampling for speed
    if sample_frac < 1.0:
        # Count lines first
        n_lines = sum(1 for _ in open(PAYSIM_FILE)) - 1  # -1 for header
        skip_idx = np.random.RandomState(42).choice(
            range(1, n_lines + 1), 
            size=int(n_lines * (1 - sample_frac)), 
            replace=False
        )
        df = pd.read_csv(PAYSIM_FILE, skiprows=skip_idx)
    else:
        df = pd.read_csv(PAYSIM_FILE)
    
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud rate: {df['isFraud'].mean():.4f}")
    print(f"Fraud count: {df['isFraud'].sum():,}")
    
    # --- Basic cleaning ---
    df = df.dropna(subset=['isFraud', 'amount'])
    df['isFraud'] = df['isFraud'].astype(int)
    
    # --- Feature Engineering ---
    print("\nCreating features...")
    
    # 1. Transaction type encoding
    df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
    df['type_PAYMENT'] = (df['type'] == 'PAYMENT').astype(int)
    df['type_CASH_IN'] = (df['type'] == 'CASH_IN').astype(int)
    df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
    df['type_DEBIT'] = (df['type'] == 'DEBIT').astype(int)
    
    # 2. Amount features
    df['log_amount'] = np.log1p(df['amount'])
    df['amount_squared'] = df['amount'] ** 2
    
    # 3. Balance features
    df['balance_orig_ratio'] = df['newbalanceOrig'] / (df['oldbalanceOrg'] + 1)
    df['balance_dest_ratio'] = df['newbalanceDest'] / (df['oldbalanceDest'] + 1)
    
    df['balance_orig_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_dest_change'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Ошибка в балансе (должно быть: old - amount = new)
    df['error_balance_orig'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['error_balance_dest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # 4. Zero balance flags (suspicious)
    df['orig_zero_balance'] = (df['oldbalanceOrg'] == 0).astype(int)
    df['dest_zero_balance'] = (df['oldbalanceDest'] == 0).astype(int)
    df['orig_zero_after'] = (df['newbalanceOrig'] == 0).astype(int)
    df['dest_zero_after'] = (df['newbalanceDest'] == 0).astype(int)
    
    # 5. Amount equals balance (suspicious)
    df['amount_equals_old_balance'] = (df['amount'] == df['oldbalanceOrg']).astype(int)
    df['amount_greater_balance'] = (df['amount'] > df['oldbalanceOrg']).astype(int)
    
    # 6. Customer-based features (aggregated)
    print("Computing customer features...")
    
    # Sort by customer and step
    df = df.sort_values(['nameOrig', 'step']).reset_index(drop=True)
    
    customer_groups = df.groupby('nameOrig', group_keys=False)
    
    # Customer transaction count (cumulative, excluding current)
    df['cust_txn_count'] = customer_groups.cumcount()
    
    # Customer amount statistics (cumulative)
    df['cust_amount_cumsum'] = customer_groups['amount'].cumsum() - df['amount']
    df['cust_amount_mean'] = df['cust_amount_cumsum'] / (df['cust_txn_count'] + 1)
    df['cust_amount_mean'] = df['cust_amount_mean'].fillna(df['amount'].mean())
    
    df['amount_vs_cust_mean'] = df['amount'] / (df['cust_amount_mean'] + 1)
    df['amount_diff_cust_mean'] = df['amount'] - df['cust_amount_mean']
    
    # Time since last transaction
    df['prev_step'] = customer_groups['step'].shift(1)
    df['hours_since_prev'] = df['step'] - df['prev_step']
    df['hours_since_prev'] = df['hours_since_prev'].fillna(999)
    
    df = df.drop(columns=['prev_step', 'cust_amount_cumsum'])
    
    # 7. Destination-based features
    print("Computing destination features...")
    
    df = df.sort_values(['nameDest', 'step']).reset_index(drop=True)
    dest_groups = df.groupby('nameDest', group_keys=False)
    
    df['dest_txn_count'] = dest_groups.cumcount()
    df['dest_fraud_cumsum'] = dest_groups['isFraud'].cumsum() - df['isFraud']
    
    df['dest_fraud_rate'] = df['dest_fraud_cumsum'] / (df['dest_txn_count'] + 1)
    df['dest_fraud_rate'] = df['dest_fraud_rate'].fillna(0)
    
    # Smoothed fraud rate
    alpha = 5.0
    global_fraud_rate = df['isFraud'].mean()
    df['dest_fraud_rate_smooth'] = (
        df['dest_fraud_cumsum'] + alpha * global_fraud_rate
    ) / (df['dest_txn_count'] + alpha)
    
    df = df.drop(columns=['dest_fraud_cumsum'])
    
    # 8. Interaction features
    df['amount_x_dest_fraud'] = df['amount'] * df['dest_fraud_rate_smooth']
    df['amount_x_zero_dest'] = df['amount'] * df['dest_zero_balance']
    df['amount_x_error_orig'] = df['amount'] * df['error_balance_orig'].abs()
    df['cashout_x_amount'] = df['type_CASH_OUT'] * df['log_amount']
    df['transfer_x_amount'] = df['type_TRANSFER'] * df['log_amount']
    
    # 9. High-risk flags
    df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
    
    # 10. Merchant flag (destination starts with M)
    df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
    
    # --- Final preparation ---
    df = df.sort_values('step').reset_index(drop=True)
    
    feature_drop = [
        'step', 'type', 'nameOrig', 'nameDest', 
        'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest',
        'isFraud', 'isFlaggedFraud'
    ]
    
    X = df.drop(columns=feature_drop)
    y = df['isFraud']
    
    # Fill NaN
    X = X.fillna(0)
    
    # Get categorical features (none in this case, all numeric)
    categorical_features = []
    
    print(f"\nFinal dataset:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Fraud rate: {y.mean():.4f}")
    
    return df, X, y, categorical_features


# ------------------------------------------------------------------------------
# Splits
# ------------------------------------------------------------------------------

def make_splits(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series
) -> Tuple[Tuple, Tuple]:
    """Time-based split (80/10/10)"""
    
    N = len(df)
    train_end = int(N * 0.8)
    val_end = int(N * 0.9)
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    amounts_train = df['amount'].iloc[:train_end]
    amounts_val = df['amount'].iloc[train_end:val_end]
    amounts_test = df['amount'].iloc[val_end:]
    
    print("\nTime-based split (80/10/10):")
    print(f"  Train: {len(y_train):,} samples, fraud rate: {y_train.mean():.4f}")
    print(f"  Val:   {len(y_val):,} samples, fraud rate: {y_val.mean():.4f}")
    print(f"  Test:  {len(y_test):,} samples, fraud rate: {y_test.mean():.4f}")
    
    return (
        (X_train, y_train, X_val, y_val, X_test, y_test),
        (amounts_train, amounts_val, amounts_test)
    )


# ------------------------------------------------------------------------------
# Model Training
# ------------------------------------------------------------------------------

def train_lightgbm(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    amounts_val: pd.Series,
    amounts_test: pd.Series,
    params: Dict,
) -> Tuple[lgb.Booster, Metrics, Metrics, np.ndarray, np.ndarray]:
    print("\n" + "=" * 80)
    print(f"=== {name} ===")
    print("=" * 80)
    
    params = params.copy()
    spw = compute_class_weight(y_train)
    params["scale_pos_weight"] = spw
    print(f"Using scale_pos_weight = {spw:.3f}")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    print("Training LightGBM...")
    start_time = time.time()
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )
    
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s")
    print(f"Best iteration: {model.best_iteration}")
    
    # Predictions
    y_val_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_test_proba = model.predict(X_test, num_iteration=model.best_iteration)
    
    best_threshold = pick_best_threshold_by_f1(y_val, y_val_proba)
    print(f"Best threshold (F1-optimal): {best_threshold:.4f}")
    
    val_metrics = evaluate_predictions(y_val, y_val_proba, best_threshold, amounts_val)
    print_metrics("VAL", val_metrics)
    
    test_metrics = evaluate_predictions(y_test, y_test_proba, best_threshold, amounts_test)
    print_metrics("TEST", test_metrics)
    
    # Feature importance
    print("\nTop 20 feature importances:")
    importance = model.feature_importance(importance_type="gain")
    feats_imps = sorted(
        zip(X_train.columns, importance), key=lambda x: x[1], reverse=True
    )
    for feat, imp in feats_imps[:20]:
        print(f"  {feat}: {imp:.1f}")
    
    return model, val_metrics, test_metrics, y_val_proba, y_test_proba


def train_xgboost(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    amounts_val: pd.Series,
    amounts_test: pd.Series,
    params: Dict,
) -> Tuple[Any, Metrics, Metrics, np.ndarray, np.ndarray]:
    print("\n" + "=" * 80)
    print(f"=== {name} ===")
    print("=" * 80)
    
    params = params.copy()
    spw = compute_class_weight(y_train)
    params["scale_pos_weight"] = spw
    print(f"Using scale_pos_weight = {spw:.3f}")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    print("Training XGBoost...")
    start_time = time.time()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "valid")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s")
    print(f"Best iteration: {model.best_iteration}")
    
    # Predictions
    y_val_proba = model.predict(dval, iteration_range=(0, model.best_iteration))
    y_test_proba = model.predict(dtest, iteration_range=(0, model.best_iteration))
    
    best_threshold = pick_best_threshold_by_f1(y_val, y_val_proba)
    print(f"Best threshold (F1-optimal): {best_threshold:.4f}")
    
    val_metrics = evaluate_predictions(y_val, y_val_proba, best_threshold, amounts_val)
    print_metrics("VAL", val_metrics)
    
    test_metrics = evaluate_predictions(y_test, y_test_proba, best_threshold, amounts_test)
    print_metrics("TEST", test_metrics)
    
    return model, val_metrics, test_metrics, y_val_proba, y_test_proba


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    amounts_val: pd.Series,
    amounts_test: pd.Series,
) -> Tuple[Dict, Metrics, Metrics]:
    """Train heterogeneous ensemble with weighted voting"""
    
    print("\n" + "=" * 80)
    print("=== HETEROGENEOUS ENSEMBLE (LGBM + XGB) ===")
    print("=" * 80)
    
    all_models = {"lgbm": [], "xgb": []}
    all_val_probas = []
    all_test_probas = []
    all_val_aucs = []
    
    spw = compute_class_weight(y_train)
    
    # Train 2 LGBM models
    for seed in [42, 123]:
        print(f"\n--- LightGBM seed={seed} ---")
        params = LGBM_PARAMS.copy()
        params.update({"seed": seed, "scale_pos_weight": spw})
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)],
        )
        
        val_proba = model.predict(X_val, num_iteration=model.best_iteration)
        test_proba = model.predict(X_test, num_iteration=model.best_iteration)
        val_auc = roc_auc_score(y_val, val_proba)
        
        all_models["lgbm"].append(model)
        all_val_probas.append(val_proba)
        all_test_probas.append(test_proba)
        all_val_aucs.append(val_auc)
        print(f"  Val AUC: {val_auc:.4f}")
    
    # Train 2 XGB models
    for seed in [42, 123]:
        print(f"\n--- XGBoost seed={seed} ---")
        params = XGB_PARAMS.copy()
        params.update({"seed": seed, "scale_pos_weight": spw})
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        
        val_proba = model.predict(dval, iteration_range=(0, model.best_iteration))
        test_proba = model.predict(dtest, iteration_range=(0, model.best_iteration))
        val_auc = roc_auc_score(y_val, val_proba)
        
        all_models["xgb"].append(model)
        all_val_probas.append(val_proba)
        all_test_probas.append(test_proba)
        all_val_aucs.append(val_auc)
        print(f"  Val AUC: {val_auc:.4f}")
    
    # Weighted averaging
    print("\n--- Weighted Ensemble ---")
    weights = np.array(all_val_aucs)
    weights = weights / weights.sum()
    
    model_names = ["LGBM-42", "LGBM-123", "XGB-42", "XGB-123"]
    for name, w, auc in zip(model_names, weights, all_val_aucs):
        print(f"  {name}: weight={w:.3f}, Val AUC={auc:.4f}")
    
    val_pred_weighted = sum(w * p for w, p in zip(weights, all_val_probas))
    test_pred_weighted = sum(w * p for w, p in zip(weights, all_test_probas))
    
    best_threshold = pick_best_threshold_by_f1(y_val, val_pred_weighted)
    print(f"\nEnsemble threshold: {best_threshold:.4f}")
    
    val_metrics = evaluate_predictions(y_val, val_pred_weighted, best_threshold, amounts_val)
    print_metrics("VAL ENSEMBLE", val_metrics)
    
    test_metrics = evaluate_predictions(y_test, test_pred_weighted, best_threshold, amounts_test)
    print_metrics("TEST ENSEMBLE", test_metrics)
    
    return all_models, val_metrics, test_metrics


# ------------------------------------------------------------------------------
# Save models
# ------------------------------------------------------------------------------

def save_models(
    models: Dict,
    feature_names: List[str],
    threshold: float,
    prefix: str = "paysim_ensemble"
):
    """Save ensemble models"""
    model_paths = []
    
    for i, m in enumerate(models["lgbm"]):
        path = f"{prefix}_lgbm_{i}.txt"
        m.save_model(path)
        model_paths.append(path)
    
    # XGB models saved with pickle (for simplicity)
    import pickle
    for i, m in enumerate(models["xgb"]):
        path = f"{prefix}_xgb_{i}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(m, f)
        model_paths.append(path)
    
    meta = {
        "model_paths": model_paths,
        "features": feature_names,
        "threshold": float(threshold),
        "num_lgbm": len(models["lgbm"]),
        "num_xgb": len(models["xgb"]),
    }
    
    meta_path = f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Saved ensemble to {meta_path}")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PAYSIM FRAUD DETECTION - ULTIMATE ENSEMBLE")
    print("=" * 80)
    
    # 1. Load and prepare data
    df, X, y, cat_features = load_and_prepare_paysim(sample_frac=SAMPLE_FRACTION)
    
    # 2. Make splits
    splits, amounts = make_splits(df, X, y)
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    amounts_train, amounts_val, amounts_test = amounts
    
    # 3. Train individual models
    print("\n" + "=" * 80)
    print("PHASE 1: Individual Models")
    print("=" * 80)
    
    lgbm_model, lgbm_val, lgbm_test, _, _ = train_lightgbm(
        name="LightGBM",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        amounts_val=amounts_val,
        amounts_test=amounts_test,
        params=LGBM_PARAMS,
    )
    
    xgb_model, xgb_val, xgb_test, _, _ = train_xgboost(
        name="XGBoost",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        amounts_val=amounts_val,
        amounts_test=amounts_test,
        params=XGB_PARAMS,
    )
    
    # 4. Train ensemble
    print("\n" + "=" * 80)
    print("PHASE 2: Heterogeneous Ensemble")
    print("=" * 80)
    
    ensemble_models, ens_val, ens_test = train_ensemble(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        amounts_val=amounts_val,
        amounts_test=amounts_test,
    )
    
    # 5. Save models
    save_models(
        models=ensemble_models,
        feature_names=list(X.columns),
        threshold=ens_val.threshold,
        prefix="paysim_ensemble",
    )
    
    # 6. Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n🔷 Individual Models (TEST):")
    print_metrics("LightGBM", lgbm_test)
    print()
    print_metrics("XGBoost", xgb_test)
    
    print("\n🏆 ENSEMBLE (TEST):")
    print_metrics("ENSEMBLE", ens_test)
    
    print("\n💡 Key Insights:")
    print(f"  Dataset: {len(df):,} transactions")
    print(f"  Fraud rate: {y.mean():.4f}")
    print(f"  Best single AUC: {max(lgbm_test.roc_auc, xgb_test.roc_auc):.4f}")
    print(f"  Ensemble AUC: {ens_test.roc_auc:.4f}")
    print(f"  Ensemble F1: {ens_test.f1:.4f}")
    if ens_test.business_metrics:
        bm = ens_test.business_metrics
        print(f"  Fraud prevention: {bm.fraud_prevention_rate*100:.1f}%")
        print(f"  Money saved: ${bm.blocked_fraud_amount:,.0f}")


if __name__ == "__main__":
    main()

