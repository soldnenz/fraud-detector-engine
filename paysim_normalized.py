"""
NORMALIZED PaySim Fraud Detection

Ключевая идея: масштабировать суммы в PaySim под распределение transactions.csv
Это позволит моделям из разных датасетов работать на одном масштабе.

Метод: Percentile matching
- Для каждой суммы в PaySim находим её перцентиль
- Заменяем на соответствующий перцентиль из transactions.csv
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
TRANSACTIONS_FILE = "transactions.csv"
BEHAVIOR_FILE = "customer_behavior.csv"

# Sampling
SAMPLE_FRACTION = 0.5  # 50% of PaySim

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
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
    "max_depth": 7,
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
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return neg / (pos + 1e-6)


def pick_best_threshold_by_f1(y_true: pd.Series, y_proba: np.ndarray) -> float:
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
    print(f"    TP={m.tp}, FP={m.fp}, TN={m.tn}, FN={m.fn}")
    if m.business_metrics:
        bm = m.business_metrics
        print(f"    💰 Total fraud: ₸{bm.total_fraud_amount:,.0f}")
        print(f"    ✅ Blocked: ₸{bm.blocked_fraud_amount:,.0f} ({bm.fraud_prevention_rate*100:.1f}%)")
        print(f"    ❌ Missed: ₸{bm.missed_fraud_amount:,.0f}")
        print(f"    ⚠️  FP cost: ₸{bm.blocked_legit_amount:,.0f}")


# ------------------------------------------------------------------------------
# Amount Normalization
# ------------------------------------------------------------------------------

def get_target_distribution() -> Dict:
    """Получить статистику распределения сумм из transactions.csv"""
    
    print("\n" + "=" * 80)
    print("ANALYZING TARGET DISTRIBUTION (transactions.csv)")
    print("=" * 80)
    
    CSV_PARAMS = dict(encoding="cp1251", sep=";")
    df = pd.read_csv(TRANSACTIONS_FILE, **CSV_PARAMS)
    df.columns = ["cst_id", "trans_date", "trans_datetime", "amount", 
                  "trans_id", "target_id", "label"]
    
    df = df[df["cst_id"] != "cst_dim_id"].copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    
    amounts = df["amount"]
    
    # Compute percentiles (0, 1, 2, ..., 99, 100)
    percentiles = np.percentile(amounts, np.arange(0, 101))
    
    stats = {
        "mean": float(amounts.mean()),
        "std": float(amounts.std()),
        "min": float(amounts.min()),
        "max": float(amounts.max()),
        "median": float(amounts.median()),
        "percentiles": percentiles,
        "q25": float(amounts.quantile(0.25)),
        "q75": float(amounts.quantile(0.75)),
    }
    
    print(f"Target distribution statistics:")
    print(f"  Count: {len(amounts):,}")
    print(f"  Mean: ₸{stats['mean']:,.0f}")
    print(f"  Median: ₸{stats['median']:,.0f}")
    print(f"  Std: ₸{stats['std']:,.0f}")
    print(f"  Min: ₸{stats['min']:,.0f}")
    print(f"  Max: ₸{stats['max']:,.0f}")
    print(f"  Q25-Q75: ₸{stats['q25']:,.0f} - ₸{stats['q75']:,.0f}")
    
    return stats


def normalize_amounts_percentile(
    amounts: pd.Series,
    target_stats: Dict
) -> pd.Series:
    """
    Нормализует суммы через percentile matching.
    Каждая сумма заменяется на соответствующий перцентиль из target.
    """
    
    # Находим перцентили для каждой суммы в исходных данных
    percentile_ranks = amounts.rank(pct=True) * 100
    
    # Заменяем на соответствующие значения из target
    target_percentiles = target_stats['percentiles']
    
    # Интерполяция для точного соответствия
    normalized = np.interp(
        percentile_ranks,
        np.arange(0, 101),
        target_percentiles
    )
    
    return pd.Series(normalized, index=amounts.index)


def normalize_amounts_linear(
    amounts: pd.Series,
    target_stats: Dict
) -> pd.Series:
    """
    Простая линейная нормализация.
    """
    source_min = amounts.min()
    source_max = amounts.max()
    
    target_min = target_stats['min']
    target_max = target_stats['max']
    
    # Scale to [0, 1]
    normalized = (amounts - source_min) / (source_max - source_min)
    
    # Scale to target range
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized


# ------------------------------------------------------------------------------
# Unified Feature Engineering
# ------------------------------------------------------------------------------

def create_unified_features(
    df: pd.DataFrame,
    customer_col: str,
    recipient_col: str,
    amount_col: str,
    time_col: str,
    label_col: str
) -> pd.DataFrame:
    """Универсальные фичи для обоих датасетов"""
    
    print("Creating unified features...")
    
    X = pd.DataFrame(index=df.index)
    
    # === AMOUNT FEATURES ===
    X['amount'] = df[amount_col].fillna(0).clip(lower=0)
    X['log_amount'] = np.log1p(X['amount'])
    X['sqrt_amount'] = np.sqrt(X['amount'])
    X['amount_squared'] = X['amount'] ** 2
    
    # Percentiles
    X['amount_percentile'] = df[amount_col].rank(pct=True)
    
    # Round amounts
    X['is_round_100'] = (X['amount'] % 100 == 0).astype(int)
    X['is_round_1000'] = (X['amount'] % 1000 == 0).astype(int)
    X['is_round_10000'] = (X['amount'] % 10000 == 0).astype(int)
    
    # High/low flags
    X['is_very_high_amount'] = (X['amount'] > df[amount_col].quantile(0.99)).astype(int)
    X['is_high_amount'] = (X['amount'] > df[amount_col].quantile(0.95)).astype(int)
    X['is_medium_amount'] = (
        (X['amount'] > df[amount_col].quantile(0.25)) & 
        (X['amount'] < df[amount_col].quantile(0.75))
    ).astype(int)
    X['is_low_amount'] = (X['amount'] < df[amount_col].quantile(0.05)).astype(int)
    
    # === CUSTOMER HISTORY ===
    print("  Customer history...")
    df_sorted = df.sort_values([customer_col, time_col]).reset_index(drop=True)
    customer_groups = df_sorted.groupby(customer_col, group_keys=False)
    
    cust_txn_count = customer_groups.cumcount()
    X['cust_txn_count'] = cust_txn_count
    X['log_cust_txn_count'] = np.log1p(cust_txn_count)
    X['is_first_cust_txn'] = (cust_txn_count == 0).astype(int)
    X['is_early_cust_txn'] = (cust_txn_count < 3).astype(int)
    
    # Customer amount stats
    cust_amount_cumsum = customer_groups[amount_col].cumsum() - df_sorted[amount_col]
    cust_amount_mean = cust_amount_cumsum / (cust_txn_count + 1)
    global_mean = df[amount_col].mean()
    cust_amount_mean = cust_amount_mean.fillna(global_mean)
    
    X['cust_amount_mean'] = cust_amount_mean
    X['amount_vs_cust_mean_ratio'] = X['amount'] / (cust_amount_mean + 1)
    X['amount_vs_cust_mean_diff'] = X['amount'] - cust_amount_mean
    X['is_unusual_high_for_cust'] = (X['amount_vs_cust_mean_ratio'] > 3).astype(int)
    X['is_unusual_low_for_cust'] = (X['amount_vs_cust_mean_ratio'] < 0.3).astype(int)
    
    # === RECIPIENT HISTORY ===
    print("  Recipient history...")
    df_sorted_rcpt = df.sort_values([recipient_col, time_col]).reset_index(drop=True)
    recipient_groups = df_sorted_rcpt.groupby(recipient_col, group_keys=False)
    
    rcpt_txn_count = recipient_groups.cumcount()
    X['rcpt_txn_count'] = rcpt_txn_count
    X['log_rcpt_txn_count'] = np.log1p(rcpt_txn_count)
    X['is_first_rcpt_txn'] = (rcpt_txn_count == 0).astype(int)
    
    # Recipient fraud rate (cumulative, no leakage)
    rcpt_fraud_cumsum = recipient_groups[label_col].cumsum() - df_sorted_rcpt[label_col]
    rcpt_fraud_rate = rcpt_fraud_cumsum / (rcpt_txn_count + 1)
    
    global_fraud_rate = df[label_col].mean()
    rcpt_fraud_rate = rcpt_fraud_rate.fillna(global_fraud_rate)
    
    # Smoothed
    alpha = 5.0
    rcpt_fraud_rate_smooth = (
        rcpt_fraud_cumsum + alpha * global_fraud_rate
    ) / (rcpt_txn_count + alpha)
    
    X['rcpt_fraud_rate'] = rcpt_fraud_rate
    X['rcpt_fraud_rate_smooth'] = rcpt_fraud_rate_smooth
    X['log_rcpt_fraud_rate_smooth'] = np.log1p(rcpt_fraud_rate_smooth * 100)
    X['is_high_risk_rcpt'] = (rcpt_fraud_rate_smooth > 0.1).astype(int)
    X['is_medium_risk_rcpt'] = (
        (rcpt_fraud_rate_smooth > 0.01) & 
        (rcpt_fraud_rate_smooth <= 0.1)
    ).astype(int)
    
    # === CUSTOMER-RECIPIENT PAIR ===
    print("  Customer-recipient interactions...")
    df_cust_rcpt = df.sort_values([customer_col, recipient_col, time_col])
    cust_rcpt_count = df_cust_rcpt.groupby([customer_col, recipient_col]).cumcount()
    X['is_new_rcpt_for_cust'] = (cust_rcpt_count == 0).astype(int)
    X['cust_rcpt_pair_count'] = cust_rcpt_count
    X['log_cust_rcpt_pair_count'] = np.log1p(cust_rcpt_count)
    
    # === INTERACTIONS ===
    X['amount_x_rcpt_fraud'] = X['amount'] * X['rcpt_fraud_rate_smooth']
    X['amount_x_new_rcpt'] = X['amount'] * X['is_new_rcpt_for_cust']
    X['amount_x_first_cust'] = X['amount'] * X['is_first_cust_txn']
    X['amount_x_high_risk_rcpt'] = X['amount'] * X['is_high_risk_rcpt']
    X['log_amount_x_rcpt_fraud'] = X['log_amount'] * X['rcpt_fraud_rate_smooth']
    X['high_amount_x_new_rcpt'] = X['is_high_amount'] * X['is_new_rcpt_for_cust']
    X['high_amount_x_high_risk'] = X['is_high_amount'] * X['is_high_risk_rcpt']
    X['cust_exp_vs_rcpt_risk'] = X['log_cust_txn_count'] * X['rcpt_fraud_rate_smooth']
    
    # === RATIOS ===
    X['cust_vs_rcpt_count_ratio'] = X['cust_txn_count'] / (X['rcpt_txn_count'] + 1)
    X['amount_vs_global_mean_ratio'] = X['amount'] / global_mean
    
    # Fill NaN
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"  Created {X.shape[1]} features")
    
    return X


# ------------------------------------------------------------------------------
# Load PaySim with normalization
# ------------------------------------------------------------------------------

def load_paysim_normalized(
    target_stats: Dict,
    sample_frac: float = 1.0
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Загружает PaySim и нормализует суммы под transactions.csv"""
    
    print("\n" + "=" * 80)
    print(f"LOADING PAYSIM (sample={sample_frac:.1%})")
    print("=" * 80)
    
    # Read with sampling
    if sample_frac < 1.0:
        n_lines = sum(1 for _ in open(PAYSIM_FILE)) - 1
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
    
    # Original amounts stats
    print(f"\nOriginal PaySim amounts:")
    print(f"  Mean: ${df['amount'].mean():,.0f}")
    print(f"  Median: ${df['amount'].median():,.0f}")
    print(f"  Min: ${df['amount'].min():,.0f}")
    print(f"  Max: ${df['amount'].max():,.0f}")
    
    # Normalize amounts
    print(f"\n🔄 Normalizing amounts to match transactions.csv distribution...")
    df['amount'] = normalize_amounts_percentile(df['amount'], target_stats)
    
    print(f"\nNormalized PaySim amounts:")
    print(f"  Mean: ₸{df['amount'].mean():,.0f}")
    print(f"  Median: ₸{df['amount'].median():,.0f}")
    print(f"  Min: ₸{df['amount'].min():,.0f}")
    print(f"  Max: ₸{df['amount'].max():,.0f}")
    
    # Sort by time
    df = df.sort_values('step').reset_index(drop=True)
    
    # Create features
    X = create_unified_features(
        df=df,
        customer_col='nameOrig',
        recipient_col='nameDest',
        amount_col='amount',
        time_col='step',
        label_col='isFraud'
    )
    
    y = df['isFraud'].astype(int)
    amounts = df['amount']
    
    print(f"\nFinal: {len(X):,} samples, {X.shape[1]} features")
    
    return X, y, amounts


# ------------------------------------------------------------------------------
# Load transactions.csv
# ------------------------------------------------------------------------------

def load_transactions_unified() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Загружает transactions.csv с едиными фичами"""
    
    print("\n" + "=" * 80)
    print("LOADING TRANSACTIONS.CSV")
    print("=" * 80)
    
    CSV_PARAMS = dict(encoding="cp1251", sep=";")
    
    df_trans = pd.read_csv(TRANSACTIONS_FILE, **CSV_PARAMS)
    df_trans.columns = [
        "cst_id", "trans_date", "trans_datetime", "amount",
        "trans_id", "target_id", "label"
    ]
    
    df_trans = df_trans[df_trans["cst_id"] != "cst_dim_id"].copy()
    
    def _parse_date(series):
        cleaned = series.astype(str).str.strip().str.strip("'\"").replace("", pd.NA)
        return pd.to_datetime(cleaned, errors="coerce")
    
    df_trans["trans_datetime"] = _parse_date(df_trans["trans_datetime"])
    df_trans["amount"] = pd.to_numeric(df_trans["amount"], errors="coerce")
    df_trans["label"] = pd.to_numeric(df_trans["label"], errors="coerce")
    
    df_trans = df_trans.dropna(subset=["cst_id", "label", "amount"])
    df_trans["label"] = df_trans["label"].astype(int)
    df_trans = df_trans[df_trans["label"].isin([0, 1])]
    df_trans = df_trans.sort_values("trans_datetime").reset_index(drop=True)
    
    print(f"Loaded {len(df_trans):,} transactions")
    print(f"Fraud rate: {df_trans['label'].mean():.4f}")
    print(f"Amount: ₸{df_trans['amount'].min():,.0f} - ₸{df_trans['amount'].max():,.0f}")
    
    X = create_unified_features(
        df=df_trans,
        customer_col='cst_id',
        recipient_col='target_id',
        amount_col='amount',
        time_col='trans_datetime',
        label_col='label'
    )
    
    y = df_trans['label']
    amounts = df_trans['amount']
    
    print(f"Final: {len(X):,} samples, {X.shape[1]} features")
    
    return X, y, amounts


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    amounts_val: pd.Series,
) -> Tuple[Dict, float, np.ndarray]:
    
    print("\n" + "=" * 80)
    print("TRAINING ENSEMBLE")
    print("=" * 80)
    
    spw = compute_class_weight(y_train)
    print(f"Scale pos weight: {spw:.3f}")
    
    all_models = {"lgbm": [], "xgb": []}
    all_val_probas = []
    all_val_aucs = []
    
    # Train 2 LGBM
    for seed in [42, 123]:
        print(f"\n--- LGBM seed={seed} ---")
        params = LGBM_PARAMS.copy()
        params.update({"seed": seed, "scale_pos_weight": spw})
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )
        
        val_proba = model.predict(X_val, num_iteration=model.best_iteration)
        val_auc = roc_auc_score(y_val, val_proba)
        
        all_models["lgbm"].append(model)
        all_val_probas.append(val_proba)
        all_val_aucs.append(val_auc)
        print(f"  Val AUC: {val_auc:.4f}")
    
    # Train 2 XGB
    for seed in [42, 123]:
        print(f"\n--- XGBoost seed={seed} ---")
        params = XGB_PARAMS.copy()
        params.update({"seed": seed, "scale_pos_weight": spw})
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "valid")],
            early_stopping_rounds=50,
            verbose_eval=50,
        )
        
        val_proba = model.predict(dval, iteration_range=(0, model.best_iteration))
        val_auc = roc_auc_score(y_val, val_proba)
        
        all_models["xgb"].append(model)
        all_val_probas.append(val_proba)
        all_val_aucs.append(val_auc)
        print(f"  Val AUC: {val_auc:.4f}")
    
    # Ensemble
    print("\n--- Ensemble ---")
    weights = np.array(all_val_aucs)
    weights = weights / weights.sum()
    
    names = ["LGBM-42", "LGBM-123", "XGB-42", "XGB-123"]
    for name, w, auc in zip(names, weights, all_val_aucs):
        print(f"  {name}: {w:.3f} (AUC={auc:.4f})")
    
    val_proba_ensemble = sum(w * p for w, p in zip(weights, all_val_probas))
    threshold = pick_best_threshold_by_f1(y_val, val_proba_ensemble)
    
    print(f"\nThreshold: {threshold:.4f}")
    
    val_metrics = evaluate_predictions(y_val, val_proba_ensemble, threshold, amounts_val)
    print_metrics("VALIDATION", val_metrics)
    
    return all_models, threshold, weights


def predict_ensemble(models: Dict, weights: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    all_probas = []
    
    for model in models["lgbm"]:
        proba = model.predict(X, num_iteration=model.best_iteration)
        all_probas.append(proba)
    
    for model in models["xgb"]:
        dtest = xgb.DMatrix(X)
        proba = model.predict(dtest, iteration_range=(0, model.best_iteration))
        all_probas.append(proba)
    
    return sum(w * p for w, p in zip(weights, all_probas))


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    print("\n" + "=" * 80)
    print("NORMALIZED PAYSIM FRAUD DETECTION")
    print("=" * 80)
    
    # 1. Get target distribution
    target_stats = get_target_distribution()
    
    # 2. Load normalized PaySim
    X_paysim, y_paysim, amounts_paysim = load_paysim_normalized(
        target_stats, 
        SAMPLE_FRACTION
    )
    
    # Split PaySim 80/20
    N = len(X_paysim)
    split_idx = int(N * 0.8)
    
    X_train = X_paysim.iloc[:split_idx]
    y_train = y_paysim.iloc[:split_idx]
    amounts_train = amounts_paysim.iloc[:split_idx]
    
    X_val = X_paysim.iloc[split_idx:]
    y_val = y_paysim.iloc[split_idx:]
    amounts_val = amounts_paysim.iloc[split_idx:]
    
    print(f"\nPaySim split:")
    print(f"  Train: {len(y_train):,}, fraud: {y_train.mean():.4f}")
    print(f"  Val:   {len(y_val):,}, fraud: {y_val.mean():.4f}")
    
    # 3. Train
    models, threshold, weights = train_ensemble(
        X_train, y_train, X_val, y_val, amounts_val
    )
    
    # 4. Test on transactions.csv
    print("\n" + "=" * 80)
    print("TESTING ON TRANSACTIONS.CSV")
    print("=" * 80)
    
    X_trans, y_trans, amounts_trans = load_transactions_unified()
    
    print("\nPredicting...")
    y_trans_proba = predict_ensemble(models, weights, X_trans)
    
    print(f"\nUsing threshold: {threshold:.4f}")
    trans_metrics = evaluate_predictions(
        y_trans, y_trans_proba, threshold, amounts_trans
    )
    print_metrics("TRANSACTIONS.CSV", trans_metrics)
    
    # Optimal threshold
    threshold_opt = pick_best_threshold_by_f1(y_trans, y_trans_proba)
    if abs(threshold_opt - threshold) > 0.05:
        print(f"\nOptimal threshold: {threshold_opt:.4f}")
        trans_metrics_opt = evaluate_predictions(
            y_trans, y_trans_proba, threshold_opt, amounts_trans
        )
        print_metrics("TRANSACTIONS.CSV (opt)", trans_metrics_opt)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training: PaySim normalized ({len(X_train):,} samples)")
    print(f"  Val AUC: {roc_auc_score(y_val, predict_ensemble(models, weights, X_val)):.4f}")
    print(f"\nTesting: transactions.csv ({len(X_trans):,} samples)")
    print(f"  ROC-AUC: {trans_metrics.roc_auc:.4f}")
    print(f"  F1: {trans_metrics.f1:.3f}")
    print(f"  Precision: {trans_metrics.precision:.3f}")
    print(f"  Recall: {trans_metrics.recall:.3f}")
    if trans_metrics.business_metrics:
        bm = trans_metrics.business_metrics
        print(f"  Fraud prevention: {bm.fraud_prevention_rate*100:.1f}%")
        print(f"  Money saved: ₸{bm.blocked_fraud_amount:,.0f}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

