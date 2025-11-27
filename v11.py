"""
ULTIMATE V11: Fraud Detection "КАК У БАНКОВ" 🏦

ЧТО НОВОГО В V11:
1. Category embeddings (OneHot + TruncatedSVD) - плотные векторы для категорий
2. OOF прогнозы для каждой модели отдельно (для stacking)
3. Stacking (meta-модель) - LogisticRegression поверх base-моделей
4. Auto-threshold trainer - автоматический подбор порогов по разным стратегиям
5. Models bundle - единый контейнер всех моделей и артефактов
6. Все улучшения V10 сохранены
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

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


def add_category_embeddings(
    data: pd.DataFrame,
    categorical_cols: List[str],
    n_components: int = 6,
    prefix: str = "catemb"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Строим плотные эмбеддинги для категориальных признаков через:
    OneHotEncoder + TruncatedSVD.
    
    - Никакого label → нет target leakage.
    - Эмбеддинг общий по всем указанным колонкам.
    """
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    ohe_matrix = enc.fit_transform(data[categorical_cols])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    emb = svd.fit_transform(ohe_matrix)
    for i in range(n_components):
        data[f"{prefix}_{i}"] = emb[:, i]
    emb_artifacts = {
        "encoder": enc,
        "svd": svd,
        "cols": categorical_cols,
        "n_components": n_components,
        "prefix": prefix,
    }
    return data, emb_artifacts


def train_threshold_strategies(
    y_true: pd.Series,
    y_proba: np.ndarray,
    amounts: pd.Series,
    beta: float = 0.5,
    min_recall: float = 0.6,
) -> Dict[str, Any]:
    """
    Авто-подбор порогов под разные стратегии:
    - max F1
    - min cost (FP/FN)
    - порог с заданным минимумом Recall
    """
    results = {}
    
    # 1) F1-оптимальный
    thr_f1 = pick_best_threshold_by_f1(y_true, y_proba)
    met_f1 = evaluate_predictions(y_true, y_proba, thr_f1, amounts, beta=beta)
    results["f1"] = {"threshold": thr_f1, "metrics": met_f1}
    
    # 2) Cost-оптимальный
    thr_cost = pick_threshold_by_cost(y_true, y_proba, amounts)
    met_cost = evaluate_predictions(y_true, y_proba, thr_cost, amounts, beta=beta)
    results["cost"] = {"threshold": thr_cost, "metrics": met_cost}
    
    # 3) Порог с минимальным Recall
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    thr_recall = 0.5
    if len(thr) > 0:
        # Находим первый порог, у которого recall >= min_recall
        mask = rec[:-1] >= min_recall
        if mask.any():
            idx = np.where(mask)[0][0]
            thr_recall = float(thr[idx])
        else:
            thr_recall = float(thr[-1]) if len(thr) > 0 else 0.5
    met_recall = evaluate_predictions(y_true, y_proba, thr_recall, amounts, beta=beta)
    results["high_recall"] = {"threshold": thr_recall, "metrics": met_recall}
    
    return results


# ------------------------------------------------------------------------------
# ADVANCED Feature Engineering
# ------------------------------------------------------------------------------

def load_and_prepare_advanced() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Загружает данные и создает ОПТИМИЗИРОВАННЫЕ фичи V9:
    - Удалены шумные фичи
    - Добавлены банковские фичи (burst, unique targets, etc.)
    - Clip/transform для стабильности
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
    print("\nCreating V9 OPTIMIZED features...")

    # === BASIC AMOUNT FEATURES ===
    data["log_amount"] = np.log1p(data["amount"].clip(lower=0))
    data["sqrt_amount"] = np.sqrt(data["amount"].clip(lower=0))
    
    # Note: amount column will be dropped later (only log/sqrt versions used)

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

    # === TARGET HISTORY REMOVED (V10) ===
    # All target_id features removed - they cause 100% leakage
    # target_id acts as unique key → perfect fraud predictor

    # === RULE-BASED ANOMALY FLAGS ===
    # Replaced is_high_amount_global with client percentile (better)
    data["is_high_amount_vs_client"] = (data["amount_over_mean_past"] >= 5.0).astype(int)
    
    # Client percentile-based high amount (FIXED: using rolling quantile to avoid leakage)
    # Use rolling quantile instead of global qcut
    data["amount_rolling_90p"] = (
        data.groupby("cst_id")["amount"]
        .transform(lambda x: x.rolling(50, min_periods=1).quantile(0.9))
    )
    data["is_high_amount_percentile"] = (data["amount"] > data["amount_rolling_90p"]).astype(int)

    # FIXED: Use shift to avoid leakage (check if different from previous)
    data["is_new_phone_model_for_client"] = (
        data.groupby("cst_id")["last_phone_model"].transform(lambda x: (x != x.shift(1)).fillna(True))
    ).astype(int)

    data["is_new_os_for_client"] = (
        data.groupby("cst_id")["last_os"].transform(lambda x: (x != x.shift(1)).fillna(True))
    ).astype(int)

    # is_new_target_for_client REMOVED (target_id leakage)

    data["hour"] = data["trans_datetime"].dt.hour
    data["is_night_tx"] = data["hour"].between(0, 5).astype(int)
    data["is_first_tx_for_client"] = (data["cst_txn_count_past"] == 0).astype(int)

    data["day_of_week"] = data["trans_datetime"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    
    # === TIME ENTROPY (NEW V9) ===
    print("  Adding time entropy features...")
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    # === V9 OPTIMIZED FEATURES ===
    print("  Adding V9 optimized features...")
    
    # Seasonality features (kept)
    data["cst_night_tx_cumsum"] = data.groupby("cst_id")["is_night_tx"].cumsum()
    data["cst_night_tx_share"] = data["cst_night_tx_cumsum"] / (data["cst_txn_count_past"] + 1)
    
    data["cst_weekend_tx_cumsum"] = data.groupby("cst_id")["is_weekend"].cumsum()
    data["cst_weekend_tx_share"] = data["cst_weekend_tx_cumsum"] / (data["cst_txn_count_past"] + 1)
    
    # Behavior ratios (kept)
    data["sessions_7d_vs_30d_ratio"] = data["sessions_unique_7d"] / (data["sessions_unique_30d"] + 1)
    data["logins_7d_vs_30d_ratio"] = data["daily_logins_avg_7d"] / (data["daily_logins_avg_30d"] + 1)
    
    # === CLIP/TRANSFORM FOR STABILITY (NEW V9) ===
    print("  Applying clip/transform for stability...")
    data["zscore_interval_7_vs_30"] = data["zscore_interval_7_vs_30"].clip(-5, 5)
    data["std_interval_30d_log"] = np.log1p(data["std_interval_30d"].clip(lower=0))
    data["fano_factor"] = data["fano_factor"].clip(lower=0, upper=100)
    data["burstiness"] = data["burstiness"].clip(-1, 1)
    
    # === BURST DETECTION (NEW V9 - STRONGEST FEATURE) ===
    print("  Adding burst detection features...")
    data = data.sort_values(["cst_id", "trans_datetime"]).reset_index(drop=True)
    
    # Helper function for rolling count (FIXED: closed="left" to avoid leakage)
    def rolling_count_by_time(group, window):
        group_indexed = group.set_index("trans_datetime")
        # Use closed="left" to exclude current transaction (only past)
        result = group_indexed.rolling(window, closed="left")["trans_id"].count()
        # Return as Series with original index
        return pd.Series(result.values, index=group.index)
    
    # Transactions in last 1 hour
    txn_1h = (
        data.groupby("cst_id", group_keys=False)
        .apply(lambda df: rolling_count_by_time(df, "1h"))
    )
    data["txn_last_1h"] = txn_1h.values
    data["txn_last_1h"] = data["txn_last_1h"].fillna(0)  # Already excluded with closed="left"
    
    # Transactions in last 10 minutes
    txn_10min = (
        data.groupby("cst_id", group_keys=False)
        .apply(lambda df: rolling_count_by_time(df, "10min"))
    )
    data["txn_last_10min"] = txn_10min.values
    data["txn_last_10min"] = data["txn_last_10min"].fillna(0)  # Already excluded with closed="left"
    
    # Transactions in last 24 hours
    txn_24h = (
        data.groupby("cst_id", group_keys=False)
        .apply(lambda df: rolling_count_by_time(df, "24h"))
    )
    data["txn_last_24h"] = txn_24h.values
    data["txn_last_24h"] = data["txn_last_24h"].fillna(0)  # Already excluded with closed="left"
    
    # === UNIQUE TARGETS WINDOWS REMOVED (V10) ===
    # These features use target_id → leakage
    
    # === DEVICE CHANGE DETECTION (NEW V9) ===
    print("  Adding device change detection...")
    data["device_changed_recently"] = (
        (data["is_new_phone_model_for_client"] == 1) &
        (data["hours_since_prev_trans"] < 24)
    ).astype(int)
    
    data["os_changed_recently"] = (
        (data["is_new_os_for_client"] == 1) &
        (data["hours_since_prev_trans"] < 24)
    ).astype(int)
    
    # === ROLLING AMOUNT FEATURES (NEW V9) ===
    print("  Adding rolling amount features...")
    
    def rolling_mean_by_time(group, window):
        group_indexed = group.set_index("trans_datetime")
        # FIXED: closed="left" to avoid leakage
        result = group_indexed.rolling(window, closed="left")["amount"].mean()
        return pd.Series(result.values, index=group.index)
    
    def rolling_std_by_time(group, window):
        group_indexed = group.set_index("trans_datetime")
        # FIXED: closed="left" to avoid leakage
        result = group_indexed.rolling(window, closed="left")["amount"].std()
        return pd.Series(result.values, index=group.index)
    
    amount_mean_7d = (
        data.groupby("cst_id", group_keys=False)
        .apply(lambda df: rolling_mean_by_time(df, "7d"))
    )
    data["amount_rolling_mean_7d"] = amount_mean_7d.values
    data["amount_rolling_mean_7d"] = data["amount_rolling_mean_7d"].fillna(data["amount"].mean())
    
    amount_std_7d = (
        data.groupby("cst_id", group_keys=False)
        .apply(lambda df: rolling_std_by_time(df, "7d"))
    )
    data["amount_rolling_std_7d"] = amount_std_7d.values
    data["amount_rolling_std_7d"] = data["amount_rolling_std_7d"].fillna(0)
    
    # Amount deviation from rolling mean
    data["amount_deviation_rolling"] = np.abs(data["amount"] - data["amount_rolling_mean_7d"])
    data["amount_deviation_rolling_ratio"] = data["amount_deviation_rolling"] / (data["amount_rolling_mean_7d"] + 1)
    
    # === ROLLING ANOMALY SCORE (NEW V10) ===
    print("  Adding rolling anomaly score...")
    # Z-score based on rolling stats
    data["zscore_amount"] = (
        (data["amount"] - data["amount_rolling_mean_7d"]) / (data["amount_rolling_std_7d"] + 1e-3)
    )
    data["zscore_amount"] = data["zscore_amount"].clip(-10, 10)  # Clip outliers
    
    # Rolling ratios
    data["amount_over_rolling_mean_7d"] = data["amount"] / (data["amount_rolling_mean_7d"] + 1e-3)
    
    # Median of last 10 transactions
    data["amount_median_last_10"] = (
        data.groupby("cst_id")["amount"]
        .transform(lambda x: x.rolling(10, min_periods=1).median())
    )
    data["amount_over_median_last_10"] = data["amount"] / (data["amount_median_last_10"] + 1e-3)
    
    # === BEHAVIORAL FINGERPRINT (NEW V10) ===
    print("  Adding behavioral fingerprint...")
    
    # Hour deviation from customer mean
    data["cst_hour_mean"] = (
        data.groupby("cst_id")["hour"]
        .transform(lambda x: x.rolling(50, min_periods=1).mean())
    )
    data["hour_deviation_from_customer_mean"] = np.abs(data["hour"] - data["cst_hour_mean"])
    
    # Activity profile score (combination of time features)
    data["activity_profile_score"] = (
        data["hour_sin"] * 0.5 +
        data["hour_cos"] * 0.5 +
        (data["is_night_tx"] * 0.3) +
        (data["is_weekend"] * 0.2)
    )
    
    # === FREQUENCY ENCODING (NEW V10) ===
    print("  Adding frequency encoding...")
    
    # Frequency of OS in dataset (computed on full data, but safe - no target leakage)
    os_counts = data["last_os"].value_counts().to_dict()
    data["freq_last_os"] = data["last_os"].map(os_counts) / len(data)
    
    # Frequency of phone model
    phone_counts = data["last_phone_model"].value_counts().to_dict()
    data["freq_last_phone_model"] = data["last_phone_model"].map(phone_counts) / len(data)
    
    # === CLIENT TIME DENSITY (NEW V10) ===
    print("  Adding client time density...")
    
    # Transactions per day (average for customer)
    data["cst_txns_per_day"] = (
        data.groupby("cst_id")["trans_datetime"]
        .transform(lambda x: x.rolling(30, min_periods=1).count() / 30.0)
    )
    
    # === EWM FEATURES (NEW V10) ===
    print("  Adding EWM features...")
    
    # EWM amount with different alphas
    data["ewm_amount_alpha_03"] = (
        data.groupby("cst_id")["amount"]
        .transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    )
    
    # EWM interval (using hours_since_prev_trans)
    data["ewm_interval_alpha_05"] = (
        data.groupby("cst_id")["hours_since_prev_trans"]
        .transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    )
    
    # === KEPT INTERACTIONS (simplified) ===
    data["night_x_high_amount"] = data["is_night_tx"] * data["is_high_amount_vs_client"]
    data["new_device_x_high_amount"] = data["is_new_phone_model_for_client"] * data["is_high_amount_vs_client"]
    data["weekend_x_high_amount"] = data["is_weekend"] * data["is_high_amount_vs_client"]
    
    # === CATEGORY EMBEDDINGS (V11) ===
    print("  Adding category embeddings (TruncatedSVD)...")
    cat_embed_cols = ["last_phone_model", "last_os"]
    existing_cat_cols = [c for c in cat_embed_cols if c in data.columns]
    if existing_cat_cols:
        data, emb_artifacts = add_category_embeddings(
            data,
            categorical_cols=existing_cat_cols,
            n_components=6,
            prefix="catemb"
        )
    else:
        emb_artifacts = None
    
    # === LEAKAGE DETECTION LOGS ===
    print("\n  🔍 Checking for potential leakage...")
    
    # Log 1: Check suspicious targets
    print("  Top suspicious targets (fraud_rate):")
    target_fraud_rates = data.groupby("target_id")["label"].agg(['mean', 'count'])
    target_fraud_rates.columns = ['fraud_rate', 'count']
    suspicious = target_fraud_rates[target_fraud_rates['fraud_rate'] > 0.5].sort_values('fraud_rate', ascending=False).head(10)
    print(suspicious)
    
    # Log 2: Check time sorting
    time_sort_broken = data.groupby("cst_id").apply(
        lambda g: (g["trans_datetime"].diff() < pd.Timedelta(0)).any()
    ).any()
    print(f"  ⚠️  Time sort broken for some groups: {time_sort_broken}")
    
    # Log 3: Check first transaction flags
    print("  Sample is_new_phone_model_for_client (first 10 rows):")
    sample_cols = ["cst_id", "last_phone_model", "is_new_phone_model_for_client", "cst_txn_count_past"]
    print(data[sample_cols].head(10))
    
    # === FINAL CLEANUP ===
    print("\n  Finalizing features...")
    
    feature_drop_cols = [
        "cst_id", "trans_id", "trans_date", "trans_datetime", "target_id", "label",
        "hour", "day_of_week", "prev_transdatetime", "amount_cum_sum",
        "cst_night_tx_cumsum", "cst_weekend_tx_cumsum", "amount", "amount_rolling_90p",
        "cst_hour_mean"  # amount removed (only log/sqrt used)
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
    
    return data, X_full, y_full, categorical_cols, emb_artifacts


# ------------------------------------------------------------------------------
# Training with StratifiedKFold
# ------------------------------------------------------------------------------

def train_cv_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    amounts: pd.Series,
    data_full: pd.DataFrame,
    categorical_cols: List[str],
    n_folds: int = 5,
) -> Tuple[List[Any], Dict]:
    """
    Train ensemble using StratifiedKFold (3 models: LGBM, XGB, CatBoost)
    Returns models and predictions for each fold
    V10: All target_id features removed - no leakage
    """
    
    print("\n" + "=" * 80)
    print(f"TRAINING WITH {n_folds}-FOLD STRATIFIED CV (V10 - NO TARGET LEAKAGE)")
    print("=" * 80)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_models = []
    all_oof_probas = np.zeros(len(y))
    fold_metrics = []
    
    # OOF для каждой модели (для stacking)
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    
    spw = compute_class_weight(y)
    print(f"Scale pos weight: {spw:.3f}\n")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        # Get train/val data with full info
        train_data = data_full.iloc[train_idx].copy()
        val_data = data_full.iloc[val_idx].copy()
        
        # === V10: NO TARGET STATS (removed to avoid leakage) ===
        
        # Extract features for train/val
        feature_drop_cols = [
            "cst_id", "trans_id", "trans_date", "trans_datetime", "target_id", "label",
            "hour", "day_of_week", "prev_transdatetime", "amount_cum_sum",
            "cst_night_tx_cumsum", "cst_weekend_tx_cumsum", "amount", "amount_rolling_90p",
            "cst_hour_mean"
        ]
        
        X_train = train_data.drop(columns=[c for c in feature_drop_cols if c in train_data.columns])
        X_val = val_data.drop(columns=[c for c in feature_drop_cols if c in val_data.columns])
        
        # Ensure same columns
        for col in X_train.columns:
            if col not in X_val.columns:
                X_val[col] = 0
        for col in X_val.columns:
            if col not in X_train.columns:
                X_train[col] = 0
        X_train = X_train[X_val.columns]  # Same order
        
        # Handle categorical columns
        for col in categorical_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")
        
        # Fill NaN and inf
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[num_cols] = X_train[num_cols].fillna(0)
        X_train[num_cols] = X_train[num_cols].replace([np.inf, -np.inf], 0)
        X_val[num_cols] = X_val[num_cols].fillna(0)
        X_val[num_cols] = X_val[num_cols].replace([np.inf, -np.inf], 0)
        
        y_train = train_data["label"].reset_index(drop=True)
        y_val = val_data["label"].reset_index(drop=True)
        amounts_val = val_data["amount"].reset_index(drop=True)
        
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
            verbose_eval=10,  # Print every 10 iterations
        )
        print(f"  XGBoost best iteration: {xgb_model.best_iteration}, best score: {xgb_model.best_score:.6f}")
        
        # Train CatBoost
        print("--- CatBoost ---")
        cat_model = CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.03,
            eval_metric='AUC',
            loss_function='Logloss',
            random_seed=42,
            verbose=10,  # Print every 10 iterations
            task_type="CPU",
            scale_pos_weight=spw,
            early_stopping_rounds=50,
        )
        
        # CatBoost needs numeric categories
        X_train_cat = X_train.copy()
        X_val_cat = X_val.copy()
        for col in cat_cols:
            X_train_cat[col] = X_train_cat[col].cat.codes
            X_val_cat[col] = X_val_cat[col].cat.codes
        
        cat_model.fit(
            X_train_cat,
            y_train,
            eval_set=(X_val_cat, y_val),
            use_best_model=True,
        )
        print(f"  CatBoost best iteration: {cat_model.best_iteration_}, best score: {cat_model.best_score_['validation']['AUC']:.6f}")
        
        # Ensemble predictions (3 models)
        lgbm_proba = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)
        xgb_proba = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration))
        cat_proba = cat_model.predict_proba(X_val_cat)[:, 1]
        
        # Сохраняем OOF отдельно для каждой модели
        oof_lgb[val_idx] = lgbm_proba
        oof_xgb[val_idx] = xgb_proba
        oof_cat[val_idx] = cat_proba
        
        # Weighted average: 40% LGBM, 30% XGB, 30% CatBoost
        fold_proba = 0.4 * lgbm_proba + 0.3 * xgb_proba + 0.3 * cat_proba
        all_oof_probas[val_idx] = fold_proba
        
        # Evaluate fold
        fold_auc = roc_auc_score(y_val, fold_proba)
        fold_thr = pick_best_threshold_by_f1(y_val, fold_proba)
        fold_met = evaluate_predictions(y_val, fold_proba, fold_thr, amounts_val)
        
        print(f"\nFold {fold+1} Results:")
        print_metrics(f"Fold {fold+1}", fold_met)
        
        fold_metrics.append(fold_met)
        all_models.append({"lgbm": lgbm_model, "xgb": xgb_model, "cat": cat_model})
    
    # Overall CV metrics
    print("\n" + "=" * 80)
    print("OVERALL CV RESULTS")
    print("=" * 80)
    
    threshold_results = train_threshold_strategies(y, all_oof_probas, amounts, beta=0.5)
    cv_threshold = threshold_results["f1"]["threshold"]
    cv_metrics = threshold_results["f1"]["metrics"]
    cost_threshold = threshold_results["cost"]["threshold"]
    cost_metrics = threshold_results["cost"]["metrics"]
    
    print_metrics("CV Overall", cv_metrics)
    print(f"\n💰 Cost-Optimized Threshold: {cost_threshold:.4f}")
    print_metrics("CV Cost-Optimized", cost_metrics)
    
    # =========================
    # STACKING (META-МОДЕЛЬ)
    # =========================
    print("\n" + "=" * 80)
    print("STACKING (META-MODEL ON OOF PREDICTIONS)")
    print("=" * 80)
    
    base_oof = np.vstack([oof_lgb, oof_xgb, oof_cat]).T  # shape: (n_samples, 3)
    
    meta_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )
    meta_model.fit(base_oof, y)
    
    meta_oof_probas = meta_model.predict_proba(base_oof)[:, 1]
    
    meta_threshold_results = train_threshold_strategies(y, meta_oof_probas, amounts, beta=0.5)
    meta_f1_thr = meta_threshold_results["f1"]["threshold"]
    meta_f1_metrics = meta_threshold_results["f1"]["metrics"]
    meta_cost_thr = meta_threshold_results["cost"]["threshold"]
    meta_cost_metrics = meta_threshold_results["cost"]["metrics"]
    
    print("\n[STACKING] F1-Optimized meta-model:")
    print_metrics("Meta F1", meta_f1_metrics)
    
    print(f"\n[STACKING] Cost-Optimized Threshold: {meta_cost_thr:.4f}")
    print_metrics("Meta Cost", meta_cost_metrics)
    
    results = {
        "fold_metrics": fold_metrics,
        "cv_metrics": cv_metrics,
        "cost_metrics": cost_metrics,
        "cv_threshold": cv_threshold,
        "cost_threshold": cost_threshold,
        "oof_probas": all_oof_probas,
        "oof_lgb": oof_lgb,
        "oof_xgb": oof_xgb,
        "oof_cat": oof_cat,
        "meta_model": meta_model,
        "meta_f1_metrics": meta_f1_metrics,
        "meta_cost_metrics": meta_cost_metrics,
        "meta_f1_thr": meta_f1_thr,
        "meta_cost_thr": meta_cost_thr,
        "meta_oof_probas": meta_oof_probas,
    }
    
    return all_models, results


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("ULTIMATE V11 FRAUD DETECTION (BANK-LEVEL)")
    print("=" * 80)
    
    # 1. Load and prepare advanced features
    data, X_full, y_full, categorical_cols, emb_artifacts = load_and_prepare_advanced()
    amounts_full = data["amount"]
    
    # 2. Train with CV (pass full data for target stats computation)
    models, results = train_cv_ensemble(
        X=X_full,
        y=y_full,
        amounts=amounts_full,
        data_full=data,
        categorical_cols=categorical_cols,
        n_folds=5,
    )
    
    # 3. Build models bundle
    models_bundle = {
        "fold_models": models,           # список по фолдам: {"lgbm":..., "xgb":..., "cat":...}
        "meta_model": results["meta_model"],
        "thresholds": {
            "ensemble_f1": results["cv_threshold"],
            "ensemble_cost": results["cost_threshold"],
            "meta_f1": results["meta_f1_thr"],
            "meta_cost": results["meta_cost_thr"],
        },
        "emb_artifacts": emb_artifacts,  # нужно для прод-инференса
        "feature_info": {
            "categorical_cols": categorical_cols,
            "n_features": X_full.shape[1],
        },
    }
    
    # 4. Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    cv_met = results["cv_metrics"]
    cost_met = results["cost_metrics"]
    meta_f1_met = results["meta_f1_metrics"]
    meta_cost_met = results["meta_cost_metrics"]
    
    print("\n🎯 Ensemble F1-Optimized:")
    print(f"  ROC-AUC: {cv_met.roc_auc:.4f}")
    print(f"  F1-Score: {cv_met.f1:.3f}")
    print(f"  Precision: {cv_met.precision:.3f}")
    print(f"  Recall: {cv_met.recall:.3f}")
    if cv_met.business_metrics:
        bm = cv_met.business_metrics
        print(f"  Fraud Prevention: {bm.fraud_prevention_rate*100:.1f}%")
        print(f"  Total Cost: ₸{bm.total_cost:,.0f}")
    
    print("\n💰 Ensemble Cost-Optimized:")
    print(f"  ROC-AUC: {cost_met.roc_auc:.4f}")
    print(f"  F1-Score: {cost_met.f1:.3f}")
    print(f"  Precision: {cost_met.precision:.3f}")
    print(f"  Recall: {cost_met.recall:.3f}")
    if cost_met.business_metrics:
        bm = cost_met.business_metrics
        print(f"  Fraud Prevention: {bm.fraud_prevention_rate*100:.1f}%")
        print(f"  Total Cost: ₸{bm.total_cost:,.0f} ⬇️ LOWER IS BETTER")
    
    print("\n🏦 Stacking Meta-Model F1-Optimized:")
    print(f"  ROC-AUC: {meta_f1_met.roc_auc:.4f}")
    print(f"  F1-Score: {meta_f1_met.f1:.3f}")
    print(f"  Precision: {meta_f1_met.precision:.3f}")
    print(f"  Recall: {meta_f1_met.recall:.3f}")
    if meta_f1_met.business_metrics:
        bm = meta_f1_met.business_metrics
        print(f"  Fraud Prevention: {bm.fraud_prevention_rate*100:.1f}%")
        print(f"  Total Cost: ₸{bm.total_cost:,.0f}")
    
    print("\n🏦 Stacking Meta-Model Cost-Optimized:")
    print(f"  ROC-AUC: {meta_cost_met.roc_auc:.4f}")
    print(f"  F1-Score: {meta_cost_met.f1:.3f}")
    print(f"  Precision: {meta_cost_met.precision:.3f}")
    print(f"  Recall: {meta_cost_met.recall:.3f}")
    if meta_cost_met.business_metrics:
        bm = meta_cost_met.business_metrics
        print(f"  Fraud Prevention: {bm.fraud_prevention_rate*100:.1f}%")
        print(f"  Total Cost: ₸{bm.total_cost:,.0f} ⬇️ LOWER IS BETTER")
    
    print("\n✅ Training complete!")
    print(f"📊 Total features: {X_full.shape[1]}")
    print(f"📈 Best Ensemble AUC: {cv_met.roc_auc:.4f}")
    print(f"📈 Best Meta-Model AUC: {meta_f1_met.roc_auc:.4f}")
    print(f"📦 Models bundle created with {len(models)} folds")


if __name__ == "__main__":
    main()

