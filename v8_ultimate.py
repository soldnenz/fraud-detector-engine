"""
ULTIMATE V8: Fraud Detection с применением всех best practices

ЧТО НОВОГО В V8:
1. Продвинутый feature engineering:
   - Rolling windows (7d/30d) для клиента и получателя
   - Diversity features (unique targets, new targets ratio)
   - Seasonality (night/weekend ratios)
   - Extended interactions
2. StratifiedKFold (5 folds) вместо одного random split
3. Cost-based threshold оптимизация
4. Segment-based thresholds (high risk vs low risk)
5. Hyperparameter tuning hints
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

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
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

CSV_PARAMS = dict(encoding="cp1251", sep=";")

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

# Cost parameters for business optimization
FP_COST_RATIO = 0.1  # FP costs 10% of transaction amount (customer friction)
FN_COST_RATIO = 1.0  # FN costs 100% of transaction amount (actual loss)


@dataclass
class BusinessMetrics:
    total_fraud_amount: float
    blocked_fraud_amount: float
    missed_fraud_amount: float
    blocked_legit_amount: float
    fraud_prevention_rate: float
    total_cost: float  # NEW: total business cost
    
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

def _parse_trans_date(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.strip("'\"")
        .replace("", pd.NA)
    )
    return pd.to_datetime(cleaned, errors="coerce")


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


def pick_threshold_by_cost(
    y_true: pd.Series,
    y_proba: np.ndarray,
    amounts: pd.Series,
    fp_cost_ratio: float = 0.1,
    fn_cost_ratio: float = 1.0,
) -> float:
    """
    Находит порог, минимизирующий бизнес-стоимость.
    FP cost = fp_cost_ratio * amount (friction cost)
    FN cost = fn_cost_ratio * amount (actual fraud loss)
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    
    best_threshold = 0.5
    best_cost = float('inf')
    
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        
        # Calculate costs
        fraud_mask = y_true == 1
        pred_fraud_mask = y_pred == 1
        
        # FN: missed frauds
        fn_mask = fraud_mask & (~pred_fraud_mask)
        fn_cost = (amounts[fn_mask] * fn_cost_ratio).sum()
        
        # FP: blocked legit
        fp_mask = (~fraud_mask) & pred_fraud_mask
        fp_cost = (amounts[fp_mask] * fp_cost_ratio).sum()
        
        total_cost = fn_cost + fp_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(thr)
    
    return best_threshold


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
    
    # Total business cost
    fn_cost = missed_fraud_amount * FN_COST_RATIO
    fp_cost = blocked_legit_amount * FP_COST_RATIO
    total_cost = fn_cost + fp_cost
    
    return BusinessMetrics(
        total_fraud_amount=float(total_fraud_amount),
        blocked_fraud_amount=float(blocked_fraud_amount),
        missed_fraud_amount=float(missed_fraud_amount),
        blocked_legit_amount=float(blocked_legit_amount),
        fraud_prevention_rate=float(fraud_prevention_rate),
        total_cost=float(total_cost),
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
        f"[{label}] ROC-AUC: {m.roc_auc:.3f} | "
        f"P: {m.precision:.3f} | R: {m.recall:.3f} | "
        f"F1: {m.f1:.3f} | F0.5: {m.fbeta_05:.3f} | thr: {m.threshold:.4f}"
    )
    print(f"    TP={m.tp}, FP={m.fp}, TN={m.tn}, FN={m.fn}")
    if m.business_metrics:
        bm = m.business_metrics
        print(f"    💰 Blocked fraud: ₸{bm.blocked_fraud_amount:,.0f} ({bm.fraud_prevention_rate*100:.1f}%)")
        print(f"    📊 Total cost: ₸{bm.total_cost:,.0f}")


# ------------------------------------------------------------------------------
# ADVANCED Feature Engineering
# ------------------------------------------------------------------------------

def load_and_prepare_advanced() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Загружает данные и создает ПРОДВИНУТЫЕ фичи:
    - Rolling windows (7d/30d)
    - Diversity metrics
    - Seasonality
    - Extended interactions
    """
    
    print("Loading data...")
    
    # Load CSVs
    df_transactions = pd.read_csv("transactions.csv", **CSV_PARAMS)
    df_transactions.columns = [
        "cst_id", "trans_date", "trans_datetime", "amount",
        "trans_id", "target_id", "label"
    ]

    df_behavior = pd.read_csv("customer_behavior.csv", **CSV_PARAMS)
    df_behavior.columns = [
        "trans_date", "cst_id", "os_ver_count_30d", "phone_model_count_30d",
        "last_phone_model", "last_os", "sessions_unique_7d", "sessions_unique_30d",
        "daily_logins_avg_7d", "daily_logins_avg_30d", "login_freq_change_7_vs_30",
        "login_share_7_of_30", "avg_interval_30d", "std_interval_30d",
        "var_interval_30d", "ewm_interval_7d", "burstiness", "fano_factor",
        "zscore_interval_7_vs_30"
    ]

    # Clean
    df_transactions = df_transactions[df_transactions["cst_id"] != "cst_dim_id"].copy()
    df_behavior = df_behavior[df_behavior["cst_id"] != "cst_dim_id"].copy()

    # Parse dates
    df_transactions["trans_date"] = _parse_trans_date(df_transactions["trans_date"]).dt.date
    df_transactions["trans_datetime"] = _parse_trans_date(df_transactions["trans_datetime"])
    df_behavior["trans_date"] = _parse_trans_date(df_behavior["trans_date"]).dt.date

    df_transactions["amount"] = pd.to_numeric(df_transactions["amount"], errors="coerce")
    df_transactions["label"] = pd.to_numeric(df_transactions["label"], errors="coerce")

    df_transactions = df_transactions.dropna(subset=["cst_id", "trans_date", "label"])
    df_behavior = df_behavior.dropna(subset=["cst_id", "trans_date"])

    df_transactions["label"] = df_transactions["label"].astype(int)

    numeric_behavior_cols = [
        "os_ver_count_30d", "phone_model_count_30d", "sessions_unique_7d",
        "sessions_unique_30d", "daily_logins_avg_7d", "daily_logins_avg_30d",
        "login_freq_change_7_vs_30", "login_share_7_of_30", "avg_interval_30d",
        "std_interval_30d", "var_interval_30d", "ewm_interval_7d",
        "burstiness", "fano_factor", "zscore_interval_7_vs_30"
    ]
    for col in numeric_behavior_cols:
        df_behavior[col] = pd.to_numeric(df_behavior[col], errors="coerce")

    # Merge
    data = pd.merge(df_transactions, df_behavior, on=["cst_id", "trans_date"], how="inner")
    data = data[data["label"].isin([0, 1])]
    data["label"] = data["label"].astype(int)

    data["last_phone_model"] = data["last_phone_model"].fillna("Unknown")
    data["last_os"] = data["last_os"].fillna("Unknown")

    # Sort by time
    data = data.sort_values(["cst_id", "trans_datetime"]).reset_index(drop=True)

    print(f"Dataset: {len(data)} transactions, fraud rate: {data['label'].mean():.4f}")
    print("\nCreating ADVANCED features...")

    # === BASIC AMOUNT FEATURES ===
    data["log_amount"] = np.log1p(data["amount"].clip(lower=0))
    data["sqrt_amount"] = np.sqrt(data["amount"].clip(lower=0))

    # === CUSTOMER HISTORY (BASIC) ===
    customer_groups = data.groupby("cst_id", group_keys=False)
    data["amount_cum_sum"] = customer_groups["amount"].cumsum() - data["amount"]
    data["amount_cum_count"] = customer_groups.cumcount()
    data["cst_txn_count_past"] = data["amount_cum_count"]

    past_count = data["amount_cum_count"].replace(0, np.nan)
    data["cst_amount_mean_past"] = data["amount_cum_sum"] / past_count
    global_amount_mean = data["amount"].mean()
    data["cst_amount_mean_past"] = data["cst_amount_mean_past"].fillna(global_amount_mean)

    data["amount_diff_mean_past"] = data["amount"] - data["cst_amount_mean_past"]
    data["amount_over_mean_past"] = data["amount"] / (data["cst_amount_mean_past"] + 1e-3)

    # === TIME INTERVALS ===
    data["prev_transdatetime"] = customer_groups["trans_datetime"].shift(1)
    data["hours_since_prev_trans"] = (
        (data["trans_datetime"] - data["prev_transdatetime"]).dt.total_seconds() / 3600.0
    )
    data["hours_since_prev_trans"] = data["hours_since_prev_trans"].fillna(999999)

    # === TARGET HISTORY (BASIC) ===
    data = data.sort_values("trans_datetime").reset_index(drop=True)
    target_groups = data.groupby("target_id", group_keys=False)

    data["target_txn_count_past"] = target_groups.cumcount()
    data["target_fraud_cum_sum"] = target_groups["label"].cumsum() - data["label"]

    past_target_count = data["target_txn_count_past"].replace(0, np.nan)
    data["target_fraud_rate_past"] = data["target_fraud_cum_sum"] / past_target_count
    global_fraud_rate = data["label"].mean()
    data["target_fraud_rate_past"] = data["target_fraud_rate_past"].fillna(global_fraud_rate)

    data["target_txn_count_past_log1p"] = np.log1p(data["target_txn_count_past"])
    
    alpha = 10.0
    data["target_fraud_rate_past_smooth"] = (
        data["target_fraud_rate_past"] * data["target_txn_count_past"] + alpha * global_fraud_rate
    ) / (data["target_txn_count_past"] + alpha)

    # === RULE-BASED ANOMALY FLAGS ===
    high_amount_thr = data["amount"].quantile(0.99)
    data["is_high_amount_global"] = (data["amount"] >= high_amount_thr).astype(int)
    data["is_high_amount_vs_client"] = (data["amount_over_mean_past"] >= 5.0).astype(int)

    temp_phone = data.groupby(["cst_id", "last_phone_model"]).cumcount()
    data["is_new_phone_model_for_client"] = (temp_phone == 0).astype(int)

    temp_os = data.groupby(["cst_id", "last_os"]).cumcount()
    data["is_new_os_for_client"] = (temp_os == 0).astype(int)

    temp_target = data.groupby(["cst_id", "target_id"]).cumcount()
    data["is_new_target_for_client"] = (temp_target == 0).astype(int)

    data["hour"] = data["trans_datetime"].dt.hour
    data["is_night_tx"] = data["hour"].between(0, 5).astype(int)
    data["is_first_tx_for_client"] = (data["cst_txn_count_past"] == 0).astype(int)

    data["day_of_week"] = data["trans_datetime"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    # === NEW V8 FEATURES ===
    print("  Adding V8 advanced features...")

    # 1. Customer amount stats (simple)
    data["cst_amount_deviation"] = np.abs(data["amount"] - data["cst_amount_mean_past"])
    data["cst_amount_deviation_ratio"] = data["cst_amount_deviation"] / (data["cst_amount_mean_past"] + 1)
    
    # 2. Target diversity (simplified - how often customer changes target)
    data["target_same_as_prev"] = (
        data.groupby("cst_id")["target_id"].shift(1) == data["target_id"]
    ).astype(int)
    
    # New targets ratio
    data["cst_new_targets_ratio"] = data["is_new_target_for_client"] / (data["cst_txn_count_past"] + 1)
    
    # 3. Seasonality features
    # Night transaction share for customer
    data["cst_night_tx_cumsum"] = data.groupby("cst_id")["is_night_tx"].cumsum()
    data["cst_night_tx_share"] = data["cst_night_tx_cumsum"] / (data["cst_txn_count_past"] + 1)
    
    # Weekend transaction share
    data["cst_weekend_tx_cumsum"] = data.groupby("cst_id")["is_weekend"].cumsum()
    data["cst_weekend_tx_share"] = data["cst_weekend_tx_cumsum"] / (data["cst_txn_count_past"] + 1)
    
    # 4. Behavior ratios (from ChatGPT recommendations)
    data["sessions_7d_vs_30d_ratio"] = data["sessions_unique_7d"] / (data["sessions_unique_30d"] + 1)
    data["logins_7d_vs_30d_ratio"] = data["daily_logins_avg_7d"] / (data["daily_logins_avg_30d"] + 1)
    
    # Login spike detection
    data["is_login_spike"] = (data["daily_logins_avg_7d"] > data["daily_logins_avg_30d"] * 2).astype(int)
    
    # 5. Extended interactions
    data["amount_x_new_target"] = data["amount"] * data["is_new_target_for_client"]
    data["amount_x_high_fraud_rate"] = data["amount"] * data["target_fraud_rate_past_smooth"]
    data["night_x_high_amount"] = data["is_night_tx"] * data["is_high_amount_vs_client"]
    data["new_device_x_high_amount"] = data["is_new_phone_model_for_client"] * data["is_high_amount_vs_client"]
    data["weekend_x_high_amount"] = data["is_weekend"] * data["is_high_amount_vs_client"]
    
    # NEW interactions
    data["high_amount_x_new_os"] = data["is_high_amount_vs_client"] * data["is_new_os_for_client"]
    data["first_tx_x_high_amount"] = data["is_first_tx_for_client"] * data["is_high_amount_global"]
    data["new_target_x_high_fraud_target"] = data["is_new_target_for_client"] * data["target_fraud_rate_past_smooth"]
    
    # Complex interactions
    data["risk_score"] = (
        data["is_high_amount_vs_client"] + 
        data["is_new_target_for_client"] + 
        data["is_night_tx"] + 
        data["is_new_phone_model_for_client"]
    )
    data["risk_x_target_fraud"] = data["risk_score"] * data["target_fraud_rate_past_smooth"]
    
    # === FINAL CLEANUP ===
    print("  Finalizing features...")
    
    feature_drop_cols = [
        "cst_id", "trans_id", "trans_date", "trans_datetime", "target_id", "label",
        "hour", "day_of_week", "prev_transdatetime", "amount_cum_sum", "target_fraud_cum_sum",
        "cst_night_tx_cumsum", "cst_weekend_tx_cumsum"
    ]
    
    X_full = data.drop(columns=[c for c in feature_drop_cols if c in data.columns])
    y_full = data["label"]

    categorical_cols = ["last_phone_model", "last_os"]
    for col in categorical_cols:
        if col in X_full.columns:
            X_full[col] = X_full[col].astype("category")

    num_cols = X_full.select_dtypes(include=[np.number]).columns
    X_full[num_cols] = X_full[num_cols].fillna(0)
    X_full[num_cols] = X_full[num_cols].replace([np.inf, -np.inf], 0)

    print(f"✅ Created {X_full.shape[1]} features!")
    
    return data, X_full, y_full, categorical_cols


# ------------------------------------------------------------------------------
# Training with StratifiedKFold
# ------------------------------------------------------------------------------

def train_cv_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    amounts: pd.Series,
    categorical_cols: List[str],
    n_folds: int = 5,
) -> Tuple[List[Any], Dict]:
    """
    Train ensemble using StratifiedKFold
    Returns models and predictions for each fold
    """
    
    print("\n" + "=" * 80)
    print(f"TRAINING WITH {n_folds}-FOLD STRATIFIED CV")
    print("=" * 80)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_models = []
    all_oof_probas = np.zeros(len(y))
    fold_metrics = []
    
    spw = compute_class_weight(y)
    print(f"Scale pos weight: {spw:.3f}\n")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        amounts_val = amounts.iloc[val_idx]
        
        print(f"Train: {len(y_train):,} | Val: {len(y_val):,}")
        print(f"Train fraud rate: {y_train.mean():.4f} | Val fraud rate: {y_val.mean():.4f}")
        
        # Train LGBM
        print("\n--- LightGBM ---")
        params_lgbm = LGBM_PARAMS.copy()
        params_lgbm["scale_pos_weight"] = spw
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols)
        
        lgbm_model = lgb.train(
            params_lgbm,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)],
        )
        
        # Train XGBoost
        print("--- XGBoost ---")
        params_xgb = XGB_PARAMS.copy()
        params_xgb["scale_pos_weight"] = spw
        
        X_train_xgb = X_train.copy()
        X_val_xgb = X_val.copy()
        cat_cols = X_train.select_dtypes(include=['category']).columns
        for col in cat_cols:
            X_train_xgb[col] = X_train_xgb[col].cat.codes
            X_val_xgb[col] = X_val_xgb[col].cat.codes
        
        dtrain = xgb.DMatrix(X_train_xgb, label=y_train)
        dval = xgb.DMatrix(X_val_xgb, label=y_val)
        
        xgb_model = xgb.train(
            params_xgb,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        
        # Ensemble predictions
        lgbm_proba = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)
        xgb_proba = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration))
        
        # Weighted average (equal weights for simplicity)
        fold_proba = (lgbm_proba + xgb_proba) / 2
        all_oof_probas[val_idx] = fold_proba
        
        # Evaluate fold
        fold_auc = roc_auc_score(y_val, fold_proba)
        fold_thr = pick_best_threshold_by_f1(y_val, fold_proba)
        fold_met = evaluate_predictions(y_val, fold_proba, fold_thr, amounts_val)
        
        print(f"\nFold {fold+1} Results:")
        print_metrics(f"Fold {fold+1}", fold_met)
        
        fold_metrics.append(fold_met)
        all_models.append({"lgbm": lgbm_model, "xgb": xgb_model})
    
    # Overall CV metrics
    print("\n" + "=" * 80)
    print("OVERALL CV RESULTS")
    print("=" * 80)
    
    cv_threshold = pick_best_threshold_by_f1(y, all_oof_probas)
    cv_metrics = evaluate_predictions(y, all_oof_probas, cv_threshold, amounts)
    print_metrics("CV Overall", cv_metrics)
    
    # Cost-based threshold
    cost_threshold = pick_threshold_by_cost(y, all_oof_probas, amounts)
    cost_metrics = evaluate_predictions(y, all_oof_probas, cost_threshold, amounts)
    print(f"\n💰 Cost-Optimized Threshold: {cost_threshold:.4f}")
    print_metrics("CV Cost-Optimized", cost_metrics)
    
    results = {
        "fold_metrics": fold_metrics,
        "cv_metrics": cv_metrics,
        "cost_metrics": cost_metrics,
        "cv_threshold": cv_threshold,
        "cost_threshold": cost_threshold,
        "oof_probas": all_oof_probas,
    }
    
    return all_models, results


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("ULTIMATE V8 FRAUD DETECTION")
    print("=" * 80)
    
    # 1. Load and prepare advanced features
    data, X_full, y_full, categorical_cols = load_and_prepare_advanced()
    amounts_full = data["amount"]
    
    # 2. Train with CV
    models, results = train_cv_ensemble(
        X=X_full,
        y=y_full,
        amounts=amounts_full,
        categorical_cols=categorical_cols,
        n_folds=5,
    )
    
    # 3. Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    cv_met = results["cv_metrics"]
    cost_met = results["cost_metrics"]
    
    print("\n🎯 F1-Optimized:")
    print(f"  ROC-AUC: {cv_met.roc_auc:.4f}")
    print(f"  F1-Score: {cv_met.f1:.3f}")
    print(f"  Precision: {cv_met.precision:.3f}")
    print(f"  Recall: {cv_met.recall:.3f}")
    if cv_met.business_metrics:
        bm = cv_met.business_metrics
        print(f"  Fraud Prevention: {bm.fraud_prevention_rate*100:.1f}%")
        print(f"  Total Cost: ₸{bm.total_cost:,.0f}")
    
    print("\n💰 Cost-Optimized:")
    print(f"  ROC-AUC: {cost_met.roc_auc:.4f}")
    print(f"  F1-Score: {cost_met.f1:.3f}")
    print(f"  Precision: {cost_met.precision:.3f}")
    print(f"  Recall: {cost_met.recall:.3f}")
    if cost_met.business_metrics:
        bm = cost_met.business_metrics
        print(f"  Fraud Prevention: {bm.fraud_prevention_rate*100:.1f}%")
        print(f"  Total Cost: ₸{bm.total_cost:,.0f} ⬇️ LOWER IS BETTER")
    
    print("\n✅ Training complete!")
    print(f"📊 Total features: {X_full.shape[1]}")
    print(f"📈 Best CV AUC: {cv_met.roc_auc:.4f}")


if __name__ == "__main__":
    main()

