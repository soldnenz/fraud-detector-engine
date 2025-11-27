"""
ULTIMATE fraud detection pipeline with heterogeneous ensemble.

НОВЫЕ ВОЗМОЖНОСТИ V6:
1. Гетерогенный ансамбль: LightGBM + XGBoost + CatBoost
2. Бизнес-метрики: сумма предотвращенных мошеннических транзакций
3. Weighted voting based on validation performance
4. Более продвинутый feature engineering (взаимодействия фичей)
5. Stacking meta-model (опционально)
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any

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
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb

# CatBoost is optional
try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available. Install with: pip install catboost")
    print("   Ensemble will use only LightGBM + XGBoost")

# ------------------------------------------------------------------------------
# 1. Конфиг
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

CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 100,
    "eval_metric": "AUC",
}


@dataclass
class BusinessMetrics:
    """Бизнес-метрики по суммам транзакций"""
    total_fraud_amount: float  # Общая сумма мошеннических транзакций
    blocked_fraud_amount: float  # Сумма предотвращенных мошеннических транзакций (TP)
    missed_fraud_amount: float  # Сумма пропущенных мошеннических транзакций (FN)
    blocked_legit_amount: float  # Сумма заблокированных легитимных транзакций (FP)
    fraud_prevention_rate: float  # Процент предотвращенного мошенничества
    
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
# 2. Утилиты
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
    """scale_pos_weight = #neg / #pos"""
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return neg / (pos + 1e-6)


def pick_best_threshold_by_f1(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Подбор порога по максимальному F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1_arr = 2 * prec * rec / (prec + rec + 1e-9)
    if len(thr) == 0:
        return 0.5
    best_idx = int(np.argmax(f1_arr[:-1]))
    return float(thr[best_idx])


def pick_threshold_for_precision(
    y_true: pd.Series,
    y_proba: np.ndarray,
    target_precision: float = 0.6,
) -> Tuple[float, float]:
    """Подбор порога для precision >= target_precision"""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    best_thr = 0.5
    best_rec = 0.0
    for p, r, t in zip(prec, rec, thr):
        if p >= target_precision and r > best_rec:
            best_rec = r
            best_thr = float(t)
    return best_thr, best_rec


def compute_business_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    amounts: pd.Series,
) -> BusinessMetrics:
    """Вычисление бизнес-метрик по суммам транзакций"""
    fraud_mask = y_true == 1
    pred_fraud_mask = y_pred == 1
    
    # Суммы мошеннических транзакций
    total_fraud_amount = amounts[fraud_mask].sum()
    
    # TP: правильно заблокированные мошеннические
    tp_mask = fraud_mask & pred_fraud_mask
    blocked_fraud_amount = amounts[tp_mask].sum()
    
    # FN: пропущенные мошеннические
    fn_mask = fraud_mask & (~pred_fraud_mask)
    missed_fraud_amount = amounts[fn_mask].sum()
    
    # FP: заблокированные легитимные
    fp_mask = (~fraud_mask) & pred_fraud_mask
    blocked_legit_amount = amounts[fp_mask].sum()
    
    # Процент предотвращенного мошенничества
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
    
    # Бизнес-метрики
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
    print(
        f"    Confusion: TP={m.tp}, FP={m.fp}, TN={m.tn}, FN={m.fn} "
        f"(blocked frauds = TP)"
    )
    if m.business_metrics:
        bm = m.business_metrics
        print(f"    💰 Business Metrics:")
        print(f"       Total fraud amount: ${bm.total_fraud_amount:,.2f}")
        print(f"       ✅ Blocked fraud: ${bm.blocked_fraud_amount:,.2f} ({bm.fraud_prevention_rate*100:.1f}%)")
        print(f"       ❌ Missed fraud: ${bm.missed_fraud_amount:,.2f}")
        print(f"       ⚠️  Blocked legit: ${bm.blocked_legit_amount:,.2f}")


# ------------------------------------------------------------------------------
# 3. Загрузка данных + feature engineering
# ------------------------------------------------------------------------------

def load_and_prepare() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """Загружает данные и строит фичи"""
    
    # --- 1. Reading CSVs ---
    df_transactions = pd.read_csv("transactions.csv", **CSV_PARAMS)
    df_transactions.columns = [
        "cst_id",
        "trans_date",
        "trans_datetime",
        "amount",
        "trans_id",
        "target_id",
        "label",
    ]

    df_behavior = pd.read_csv("customer_behavior.csv", **CSV_PARAMS)
    df_behavior.columns = [
        "trans_date",
        "cst_id",
        "os_ver_count_30d",
        "phone_model_count_30d",
        "last_phone_model",
        "last_os",
        "sessions_unique_7d",
        "sessions_unique_30d",
        "daily_logins_avg_7d",
        "daily_logins_avg_30d",
        "login_freq_change_7_vs_30",
        "login_share_7_of_30",
        "avg_interval_30d",
        "std_interval_30d",
        "var_interval_30d",
        "ewm_interval_7d",
        "burstiness",
        "fano_factor",
        "zscore_interval_7_vs_30",
    ]

    # --- 2. Drop duplicated header rows ---
    df_transactions = df_transactions[df_transactions["cst_id"] != "cst_dim_id"].copy()
    df_behavior = df_behavior[df_behavior["cst_id"] != "cst_dim_id"].copy()

    # --- 3. Normalize dates, numeric columns, and labels ---
    df_transactions["trans_date"] = _parse_trans_date(
        df_transactions["trans_date"]
    ).dt.date
    df_transactions["trans_datetime"] = _parse_trans_date(
        df_transactions["trans_datetime"]
    )
    df_behavior["trans_date"] = _parse_trans_date(df_behavior["trans_date"]).dt.date

    df_transactions["amount"] = pd.to_numeric(
        df_transactions["amount"], errors="coerce"
    )
    df_transactions["label"] = pd.to_numeric(
        df_transactions["label"], errors="coerce"
    )

    # Drop rows with missing merge keys or labels
    df_transactions = df_transactions.dropna(subset=["cst_id", "trans_date", "label"])
    df_behavior = df_behavior.dropna(subset=["cst_id", "trans_date"])

    df_transactions["label"] = df_transactions["label"].astype(int)
    df_transactions["amount"] = pd.to_numeric(
        df_transactions["amount"], errors="coerce"
    )

    numeric_behavior_cols = [
        "os_ver_count_30d",
        "phone_model_count_30d",
        "sessions_unique_7d",
        "sessions_unique_30d",
        "daily_logins_avg_7d",
        "daily_logins_avg_30d",
        "login_freq_change_7_vs_30",
        "login_share_7_of_30",
        "avg_interval_30d",
        "std_interval_30d",
        "var_interval_30d",
        "ewm_interval_7d",
        "burstiness",
        "fano_factor",
        "zscore_interval_7_vs_30",
    ]
    for col in numeric_behavior_cols:
        df_behavior[col] = pd.to_numeric(df_behavior[col], errors="coerce")

    # --- 4. Merge datasets ---
    data = pd.merge(
        df_transactions, df_behavior, on=["cst_id", "trans_date"], how="inner"
    )

    # Basic cleaning
    data = data[data["label"].isin([0, 1])]
    data["label"] = data["label"].astype(int)
    for col in numeric_behavior_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Convert categorical fields
    data["last_phone_model"] = data["last_phone_model"].fillna("Unknown")
    data["last_os"] = data["last_os"].fillna("Unknown")

    # --- 5. Sort by client+time ---
    data = data.sort_values(["cst_id", "trans_datetime"]).reset_index(drop=True)

    # --- 6. Client-based amount features ---
    data["log_amount"] = np.log1p(data["amount"].clip(lower=0))

    customer_groups = data.groupby("cst_id", group_keys=False)
    data["amount_cum_sum"] = customer_groups["amount"].cumsum() - data["amount"]
    data["amount_cum_count"] = customer_groups.cumcount()
    data["cst_txn_count_past"] = data["amount_cum_count"]

    past_count = data["amount_cum_count"].replace(0, np.nan)
    data["cst_amount_mean_past"] = data["amount_cum_sum"] / past_count
    global_amount_mean = data["amount"].mean()
    data["cst_amount_mean_past"] = data["cst_amount_mean_past"].fillna(
        global_amount_mean
    )

    data["amount_diff_mean_past"] = data["amount"] - data["cst_amount_mean_past"]
    data["amount_over_mean_past"] = data["amount"] / (
        data["cst_amount_mean_past"] + 1e-3
    )

    # --- 7. Time interval features ---
    data["prev_transdatetime"] = customer_groups["trans_datetime"].shift(1)
    data["hours_since_prev_trans"] = (
        (data["trans_datetime"] - data["prev_transdatetime"])
        .dt.total_seconds()
        .div(3600.0)
    )
    data["hours_since_prev_trans"] = data["hours_since_prev_trans"].fillna(999999)

    data = data.drop(columns=["prev_transdatetime", "amount_cum_sum"])

    # --- 8. Target-based features ---
    data = data.sort_values("trans_datetime").reset_index(drop=True)
    target_groups = data.groupby("target_id", group_keys=False)

    data["target_txn_count_past"] = target_groups.cumcount()
    data["target_fraud_cum_sum"] = target_groups["label"].cumsum() - data["label"]

    past_target_count = data["target_txn_count_past"].replace(0, np.nan)
    data["target_fraud_rate_past"] = data["target_fraud_cum_sum"] / past_target_count

    global_fraud_rate = data["label"].mean()
    data["target_fraud_rate_past"] = data["target_fraud_rate_past"].fillna(
        global_fraud_rate
    )

    data = data.drop(columns=["target_fraud_cum_sum"])

    # --- 9. Smoothed target features ---
    data["target_txn_count_past_log1p"] = np.log1p(data["target_txn_count_past"])

    alpha = 10.0
    data["target_fraud_rate_past_smooth"] = (
        data["target_fraud_rate_past"] * data["target_txn_count_past"] +
        alpha * global_fraud_rate
    ) / (data["target_txn_count_past"] + alpha)

    # --- 10. Rule-based anomaly features ---
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

    # --- 11. NEW: Feature interactions ---
    # Взаимодействия фичей для усиления паттернов
    data["amount_x_new_target"] = data["amount"] * data["is_new_target_for_client"]
    data["amount_x_high_fraud_rate"] = data["amount"] * data["target_fraud_rate_past"]
    data["night_x_high_amount"] = data["is_night_tx"] * data["is_high_amount_vs_client"]
    data["new_device_x_high_amount"] = data["is_new_phone_model_for_client"] * data["is_high_amount_vs_client"]
    
    # Временные паттерны
    data["day_of_week"] = data["trans_datetime"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["weekend_x_high_amount"] = data["is_weekend"] * data["is_high_amount_vs_client"]

    # --- 12. Final preprocessing ---
    data = data.sort_values("trans_datetime").reset_index(drop=True)

    feature_drop_cols = [
        "cst_id",
        "trans_id",
        "trans_date",
        "trans_datetime",
        "target_id",
        "label",
        "hour",
        "day_of_week",
    ]
    X_full = data.drop(columns=feature_drop_cols)
    y_full = data["label"]

    categorical_cols = ["last_phone_model", "last_os"]
    for col in categorical_cols:
        X_full[col] = X_full[col].astype("category")

    # Fill NaN
    num_cols = X_full.select_dtypes(include=[np.number]).columns
    X_full[num_cols] = X_full[num_cols].fillna(0)

    print(f"Full dataset size: {len(data)}, fraud rate: {y_full.mean():.4f}")
    return data, X_full, y_full, categorical_cols


# ------------------------------------------------------------------------------
# 4. Сплиты
# ------------------------------------------------------------------------------

def make_splits(
    data: pd.DataFrame, X_full: pd.DataFrame, y_full: pd.Series
):
    N = len(data)
    train_end = int(N * 0.6)
    val_end = int(N * 0.8)

    # Time-based split
    X_train_t = X_full.iloc[:train_end]
    y_train_t = y_full.iloc[:train_end]
    X_val_t = X_full.iloc[train_end:val_end]
    y_val_t = y_full.iloc[train_end:val_end]
    X_test_t = X_full.iloc[val_end:]
    y_test_t = y_full.iloc[val_end:]
    
    # Amount splits for business metrics
    amounts_train_t = data["amount"].iloc[:train_end]
    amounts_val_t = data["amount"].iloc[train_end:val_end]
    amounts_test_t = data["amount"].iloc[val_end:]

    print("Time-based split positive rates:")
    print(
        f"  Train: {y_train_t.mean():.4f}, "
        f"Val: {y_val_t.mean():.4f}, "
        f"Test: {y_test_t.mean():.4f}"
    )

    # Random split
    rng = np.random.RandomState(42)
    indices = np.arange(N)
    rng.shuffle(indices)

    train_end_r = int(N * 0.6)
    val_end_r = int(N * 0.8)
    train_idx_r = indices[:train_end_r]
    val_idx_r = indices[train_end_r:val_end_r]
    test_idx_r = indices[val_end_r:]

    X_train_r = X_full.iloc[train_idx_r]
    y_train_r = y_full.iloc[train_idx_r]
    X_val_r = X_full.iloc[val_idx_r]
    y_val_r = y_full.iloc[val_idx_r]
    X_test_r = X_full.iloc[test_idx_r]
    y_test_r = y_full.iloc[test_idx_r]
    
    amounts_train_r = data["amount"].iloc[train_idx_r]
    amounts_val_r = data["amount"].iloc[val_idx_r]
    amounts_test_r = data["amount"].iloc[test_idx_r]

    print("\nRandom split positive rates:")
    print(
        f"  Train: {y_train_r.mean():.4f}, "
        f"Val: {y_val_r.mean():.4f}, "
        f"Test: {y_test_r.mean():.4f}"
    )

    return (
        (X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t,
         amounts_train_t, amounts_val_t, amounts_test_t),
        (X_train_r, y_train_r, X_val_r, y_val_r, X_test_r, y_test_r,
         amounts_train_r, amounts_val_r, amounts_test_r),
    )


# ------------------------------------------------------------------------------
# 5. XGBoost model training
# ------------------------------------------------------------------------------

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
    use_class_weight: bool = False,
) -> Tuple[Any, Metrics, Metrics]:
    print("\n" + "=" * 80)
    print(f"=== {name} (XGBoost) ===")
    print("=" * 80)
    
    params = params.copy()
    if use_class_weight:
        spw = compute_class_weight(y_train)
        params["scale_pos_weight"] = spw
        print(f"Using scale_pos_weight = {spw:.3f}")
    
    # Prepare categorical features
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    X_test_xgb = X_test.copy()
    
    cat_cols = X_train.select_dtypes(include=['category']).columns
    for col in cat_cols:
        X_train_xgb[col] = X_train_xgb[col].cat.codes
        X_val_xgb[col] = X_val_xgb[col].cat.codes
        X_test_xgb[col] = X_test_xgb[col].cat.codes
    
    dtrain = xgb.DMatrix(X_train_xgb, label=y_train)
    dval = xgb.DMatrix(X_val_xgb, label=y_val)
    dtest = xgb.DMatrix(X_test_xgb, label=y_test)
    
    evals = [(dtrain, "train"), (dval, "valid")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    
    print(f"Best iteration: {model.best_iteration}")
    
    # Predictions
    y_val_proba = model.predict(dval, iteration_range=(0, model.best_iteration))
    y_test_proba = model.predict(dtest, iteration_range=(0, model.best_iteration))
    
    best_threshold = pick_best_threshold_by_f1(y_val, y_val_proba)
    print(f"Best threshold (validated by F1): {best_threshold:.4f}")
    
    val_metrics = evaluate_predictions(y_val, y_val_proba, best_threshold, amounts_val)
    print_metrics("VAL", val_metrics)
    
    test_metrics = evaluate_predictions(y_test, y_test_proba, best_threshold, amounts_test)
    print_metrics("TEST", test_metrics)
    
    return model, val_metrics, test_metrics


# ------------------------------------------------------------------------------
# 6. CatBoost model training
# ------------------------------------------------------------------------------

def train_catboost(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    amounts_val: pd.Series,
    amounts_test: pd.Series,
    categorical_cols: List[str],
    params: Dict,
    use_class_weight: bool = False,
) -> Tuple[Any, Metrics, Metrics]:
    print("\n" + "=" * 80)
    print(f"=== {name} (CatBoost) ===")
    print("=" * 80)
    
    params = params.copy()
    if use_class_weight:
        spw = compute_class_weight(y_train)
        params["scale_pos_weight"] = spw
        print(f"Using scale_pos_weight = {spw:.3f}")
    
    # Get categorical indices
    cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols if col in X_train.columns]
    
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_indices)
    
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    print(f"Best iteration: {model.get_best_iteration()}")
    
    # Predictions
    y_val_proba = model.predict_proba(val_pool)[:, 1]
    y_test_proba = model.predict_proba(test_pool)[:, 1]
    
    best_threshold = pick_best_threshold_by_f1(y_val, y_val_proba)
    print(f"Best threshold (validated by F1): {best_threshold:.4f}")
    
    val_metrics = evaluate_predictions(y_val, y_val_proba, best_threshold, amounts_val)
    print_metrics("VAL", val_metrics)
    
    test_metrics = evaluate_predictions(y_test, y_test_proba, best_threshold, amounts_test)
    print_metrics("TEST", test_metrics)
    
    return model, val_metrics, test_metrics


# ------------------------------------------------------------------------------
# 7. LightGBM model training (with business metrics)
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
    categorical_cols: List[str],
    params: Dict,
    use_class_weight: bool = False,
) -> Tuple[lgb.Booster, Metrics, Metrics]:
    print("\n" + "=" * 80)
    print(f"=== {name} (LightGBM) ===")
    print("=" * 80)
    
    params = params.copy()
    if use_class_weight:
        spw = compute_class_weight(y_train)
        params["scale_pos_weight"] = spw
        print(f"Using scale_pos_weight = {spw:.3f}")
    else:
        params.pop("scale_pos_weight", None)
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    
    best_iter = getattr(model, "best_iteration", 2000)
    print(f"Best iteration: {best_iter}")
    
    # Predictions
    y_val_proba = model.predict(X_val, num_iteration=best_iter)
    y_test_proba = model.predict(X_test, num_iteration=best_iter)
    
    best_threshold = pick_best_threshold_by_f1(y_val, y_val_proba)
    print(f"Best threshold (validated by F1): {best_threshold:.4f}")
    
    val_metrics = evaluate_predictions(y_val, y_val_proba, best_threshold, amounts_val)
    print_metrics("VAL", val_metrics)
    
    test_metrics = evaluate_predictions(y_test, y_test_proba, best_threshold, amounts_test)
    print_metrics("TEST", test_metrics)
    
    return model, val_metrics, test_metrics


# ------------------------------------------------------------------------------
# 8. Heterogeneous Ensemble (LGBM + XGB + CatBoost)
# ------------------------------------------------------------------------------

def train_heterogeneous_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    amounts_val: pd.Series,
    amounts_test: pd.Series,
    categorical_cols: List[str],
) -> Tuple[Dict[str, List[Any]], Metrics, Metrics]:
    """
    Обучает гетерогенный ансамбль из трех типов моделей.
    Использует weighted averaging на основе validation AUC.
    """
    print("\n" + "=" * 80)
    print("=== HETEROGENEOUS ENSEMBLE (LGBM + XGB + CatBoost) ===")
    print("=" * 80)
    
    spw = compute_class_weight(y_train)
    print(f"Using scale_pos_weight = {spw:.3f} for all models")
    
    all_models = {"lgbm": [], "xgb": [], "catboost": []}
    all_val_probas = []
    all_test_probas = []
    all_val_aucs = []
    
    # Train 2 LightGBM models with different seeds
    for seed in [42, 52]:
        print(f"\n--- Training LightGBM with seed={seed} ---")
        params_lgbm = LGBM_PARAMS.copy()
        params_lgbm.update({"seed": seed, "scale_pos_weight": spw})
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols)
        
        model = lgb.train(
            params_lgbm,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(False)],
        )
        
        val_proba = model.predict(X_val, num_iteration=model.best_iteration)
        test_proba = model.predict(X_test, num_iteration=model.best_iteration)
        val_auc = roc_auc_score(y_val, val_proba)
        
        all_models["lgbm"].append(model)
        all_val_probas.append(val_proba)
        all_test_probas.append(test_proba)
        all_val_aucs.append(val_auc)
        print(f"  LGBM seed={seed}: Val AUC = {val_auc:.4f}")
    
    # Train 2 XGBoost models with different seeds
    for seed in [42, 52]:
        print(f"\n--- Training XGBoost with seed={seed} ---")
        params_xgb = XGB_PARAMS.copy()
        params_xgb.update({"seed": seed, "scale_pos_weight": spw})
        
        X_train_xgb = X_train.copy()
        X_val_xgb = X_val.copy()
        X_test_xgb = X_test.copy()
        
        cat_cols = X_train.select_dtypes(include=['category']).columns
        for col in cat_cols:
            X_train_xgb[col] = X_train_xgb[col].cat.codes
            X_val_xgb[col] = X_val_xgb[col].cat.codes
            X_test_xgb[col] = X_test_xgb[col].cat.codes
        
        dtrain = xgb.DMatrix(X_train_xgb, label=y_train)
        dval = xgb.DMatrix(X_val_xgb, label=y_val)
        dtest = xgb.DMatrix(X_test_xgb, label=y_test)
        
        model = xgb.train(
            params_xgb,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        
        val_proba = model.predict(dval, iteration_range=(0, model.best_iteration))
        test_proba = model.predict(dtest, iteration_range=(0, model.best_iteration))
        val_auc = roc_auc_score(y_val, val_proba)
        
        all_models["xgb"].append(model)
        all_val_probas.append(val_proba)
        all_test_probas.append(test_proba)
        all_val_aucs.append(val_auc)
        print(f"  XGB seed={seed}: Val AUC = {val_auc:.4f}")
    
    # Train 1 CatBoost model (if available)
    if CATBOOST_AVAILABLE:
        print(f"\n--- Training CatBoost ---")
        params_cb = CATBOOST_PARAMS.copy()
        params_cb.update({"scale_pos_weight": spw})
        
        cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols if col in X_train.columns]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)
        test_pool = Pool(X_test, y_test, cat_features=cat_indices)
        
        model = CatBoostClassifier(**params_cb)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        val_proba = model.predict_proba(val_pool)[:, 1]
        test_proba = model.predict_proba(test_pool)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        
        all_models["catboost"].append(model)
        all_val_probas.append(val_proba)
        all_test_probas.append(test_proba)
        all_val_aucs.append(val_auc)
        print(f"  CatBoost: Val AUC = {val_auc:.4f}")
    else:
        print(f"\n--- Skipping CatBoost (not installed) ---")
    
    # Weighted averaging based on validation AUC
    print("\n--- Computing weighted ensemble predictions ---")
    weights = np.array(all_val_aucs)
    weights = weights / weights.sum()
    
    print("Model weights:")
    model_names = ["LGBM-42", "LGBM-52", "XGB-42", "XGB-52"]
    if CATBOOST_AVAILABLE:
        model_names.append("CatBoost")
    for name, w, auc in zip(model_names, weights, all_val_aucs):
        print(f"  {name}: weight={w:.3f}, Val AUC={auc:.4f}")
    
    # Weighted average predictions
    val_pred_weighted = sum(w * p for w, p in zip(weights, all_val_probas))
    test_pred_weighted = sum(w * p for w, p in zip(weights, all_test_probas))
    
    best_threshold = pick_best_threshold_by_f1(y_val, val_pred_weighted)
    print(f"\nEnsemble best threshold (validated by F1): {best_threshold:.4f}")
    
    val_metrics = evaluate_predictions(y_val, val_pred_weighted, best_threshold, amounts_val)
    print_metrics("VAL ENSEMBLE", val_metrics)
    
    test_metrics = evaluate_predictions(y_test, test_pred_weighted, best_threshold, amounts_test)
    print_metrics("TEST ENSEMBLE", test_metrics)
    
    return all_models, val_metrics, test_metrics


# ------------------------------------------------------------------------------
# 9. MAIN
# ------------------------------------------------------------------------------

def main():
    print("="*80)
    print("ULTIMATE FRAUD DETECTION WITH HETEROGENEOUS ENSEMBLE")
    print("="*80)
    
    # 1) Load and prepare data
    data, X_full, y_full, categorical_cols = load_and_prepare()
    
    # 2) Make splits
    time_split, random_split = make_splits(data, X_full, y_full)
    
    (X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t,
     amounts_train_t, amounts_val_t, amounts_test_t) = time_split
    
    (X_train_r, y_train_r, X_val_r, y_val_r, X_test_r, y_test_r,
     amounts_train_r, amounts_val_r, amounts_test_r) = random_split
    
    # 3) Train individual models on Random split (for comparison)
    print("\n" + "="*80)
    print("PHASE 1: Training individual models (RandomSplit)")
    print("="*80)
    
    lgbm_model, lgbm_val, lgbm_test = train_lightgbm(
        name="RandomSplit - LightGBM",
        X_train=X_train_r,
        y_train=y_train_r,
        X_val=X_val_r,
        y_val=y_val_r,
        X_test=X_test_r,
        y_test=y_test_r,
        amounts_val=amounts_val_r,
        amounts_test=amounts_test_r,
        categorical_cols=categorical_cols,
        params=LGBM_PARAMS,
        use_class_weight=True,
    )
    
    xgb_model, xgb_val, xgb_test = train_xgboost(
        name="RandomSplit - XGBoost",
        X_train=X_train_r,
        y_train=y_train_r,
        X_val=X_val_r,
        y_val=y_val_r,
        X_test=X_test_r,
        y_test=y_test_r,
        amounts_val=amounts_val_r,
        amounts_test=amounts_test_r,
        params=XGB_PARAMS,
        use_class_weight=True,
    )
    
    if CATBOOST_AVAILABLE:
        catboost_model, cb_val, cb_test = train_catboost(
            name="RandomSplit - CatBoost",
            X_train=X_train_r,
            y_train=y_train_r,
            X_val=X_val_r,
            y_val=y_val_r,
            X_test=X_test_r,
            y_test=y_test_r,
            amounts_val=amounts_val_r,
            amounts_test=amounts_test_r,
            categorical_cols=categorical_cols,
            params=CATBOOST_PARAMS,
            use_class_weight=True,
        )
    else:
        catboost_model, cb_val, cb_test = None, None, None
    
    # 4) Train heterogeneous ensemble
    print("\n" + "="*80)
    print("PHASE 2: Training heterogeneous ensemble")
    print("="*80)
    
    ensemble_models, ens_val, ens_test = train_heterogeneous_ensemble(
        X_train=X_train_r,
        y_train=y_train_r,
        X_val=X_val_r,
        y_val=y_val_r,
        X_test=X_test_r,
        y_test=y_test_r,
        amounts_val=amounts_val_r,
        amounts_test=amounts_test_r,
        categorical_cols=categorical_cols,
    )
    
    # 5) Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - RandomSplit (Competition Metrics)")
    print("="*80)
    
    print("\n🔷 Individual Models:")
    print_metrics("LightGBM TEST", lgbm_test)
    print()
    print_metrics("XGBoost TEST", xgb_test)
    if CATBOOST_AVAILABLE and cb_test:
        print()
        print_metrics("CatBoost TEST", cb_test)
    
    print("\n🏆 HETEROGENEOUS ENSEMBLE:")
    print_metrics("ENSEMBLE TEST", ens_test)
    
    print("\n" + "="*80)
    print("💡 Key Insights:")
    print("="*80)
    print(f"- Dataset: {len(data)} transactions, fraud rate: {y_full.mean():.4f}")
    
    best_single_auc = max(lgbm_test.roc_auc, xgb_test.roc_auc)
    if CATBOOST_AVAILABLE and cb_test:
        best_single_auc = max(best_single_auc, cb_test.roc_auc)
    
    print(f"- Best single model AUC: {best_single_auc:.4f}")
    print(f"- Ensemble AUC: {ens_test.roc_auc:.4f}")
    print(f"- Ensemble F1: {ens_test.f1:.4f}")
    if ens_test.business_metrics:
        bm = ens_test.business_metrics
        print(f"- Fraud prevention rate: {bm.fraud_prevention_rate*100:.1f}%")
        print(f"- Money saved: ${bm.blocked_fraud_amount:,.2f}")


if __name__ == "__main__":
    main()

