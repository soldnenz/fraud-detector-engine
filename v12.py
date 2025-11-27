"""
ULTIMATE V11: Fraud Detection Inference (БЕЗ ПРИНТОВ)

Инференс версия для проверки транзакций:
- JSON конфиг с транзакцией в начале
- Вывод вероятности фрода
- Объяснение предсказания (feature importance)
- Важная информация о транзакции
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import re
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

# ============================================================================
# JSON КОНФИГ С ТРАНЗАКЦИЕЙ
# ============================================================================

TRANSACTION_CONFIG = {
    # === ОСНОВНАЯ ИНФОРМАЦИЯ О ТРАНЗАКЦИИ ===
    "cst_id": "1286648072",
    "trans_date": "2025-02-10",
    "trans_datetime": "2025-02-10 18:30:00",   # 18:30 - вечер, самое безопасное время!
    "amount": 17680,                           # +180 от среднего (1% отклонение - "человечное")
    "target_id": "merchant_grocery_014",       # обычный магазин, нет риска

    # === УСТРОЙСТВО И ОС ===
    "last_phone_model": "Samsung Galaxy A52",
    "last_os": "Android 13",
    "os_ver_count_30d": 1,                     # стабильная версия ОС
    "phone_model_count_30d": 1,                # одна модель телефона

    # === АКТИВНОСТЬ И СЕССИИ ===
    "sessions_unique_7d": 12,                  # активный пользователь
    "sessions_unique_30d": 45,
    "daily_logins_avg_7d": 3.2,
    "daily_logins_avg_30d": 3.0,
    "login_freq_change_7_vs_30": 1.07,         # стабильная активность
    "login_share_7_of_30": 0.28,               # естественное распределение

    # === ВРЕМЕННЫЕ ИНТЕРВАЛЫ В СЕКУНДАХ (реалистичные для легитимных!) ===
    "avg_interval_30d": 75000,                 # ~21 час (медиана ~20.7 часов) ✅
    "std_interval_30d": 107000,                # ~29.7 часов (медиана легитимных) ✅
    "var_interval_30d": 11500000000,           # ~107000^2 = 11.45 млрд (близко к медиане 10.4 млрд) ✅
    "ewm_interval_7d": 70000,                  # ~19.4 часов
    "burstiness": 0.184,                       # медиана легитимных
    "fano_factor": 153.3,                      # variance / mean = 11.5B / 75k
    "zscore_interval_7_vs_30": 0.15,           # небольшое отклонение
    
    # === ГОТОВЫЕ ФИЧИ ИЗ FEATURE STORE (ДОЛЖНЫ ПРИХОДИТЬ ГОТОВЫМИ!) ===
    "device_tenure_days": 120,                 # Использует Samsung Galaxy A52 уже 4 месяца
    "cst_amount_mean_past": 17500,             # Средняя сумма = текущей (идеально!)
    "cst_txn_count_past": 45,                  # Количество прошлых транзакций
    "amount_rolling_mean_7d": 17500,           # Среднее = текущей (идеально!)
    "amount_rolling_std_7d": 2800,             # Стандартное отклонение
    "txn_last_1h": 0,                          # Транзакций за последний час
    "txn_last_24h": 2,                         # Транзакций за последние 24 часа
    "is_new_phone_model_for_client": 0,        # НЕ новое устройство
    "is_new_os_for_client": 0,                 # НЕ новая ОС
    "cst_night_tx_share": 0.005,               # 0.5% транзакций ночью (медиана легитимных!)
    "cst_weekend_tx_share": 0.22,              # 22% транзакций в выходные (нормально)
    "hours_since_prev_trans": 81000,           # ~22.5 часа = 81,000 секунд
}


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

FP_COST_RATIO = 0.1
FN_COST_RATIO = 1.0


@dataclass
class BusinessMetrics:
    total_fraud_amount: float
    blocked_fraud_amount: float
    missed_fraud_amount: float
    blocked_legit_amount: float
    fraud_prevention_rate: float
    total_cost: float
    
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
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    
    best_threshold = 0.5
    best_cost = float('inf')
    
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        
        fraud_mask = y_true == 1
        pred_fraud_mask = y_pred == 1
        
        fn_mask = fraud_mask & (~pred_fraud_mask)
        fn_cost = (amounts[fn_mask] * fn_cost_ratio).sum()
        
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


def add_category_embeddings(
    data: pd.DataFrame,
    categorical_cols: List[str],
    n_components: int = 6,
    prefix: str = "catemb"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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


def normalize_os(os_raw: Any) -> Tuple[str, float]:
    """
    Превращает сырой last_os в (os_family, os_major).
    
    os_family: 'Android', 'iOS', 'HuaweiOS', 'Other', 'Unknown'
    os_major: основной номер версии (13.0 -> 13), либо NaN
    """
    if not isinstance(os_raw, str) or not os_raw.strip():
        return "Unknown", np.nan
    
    s = os_raw.strip().lower()
    
    # Определяем семейство
    if "android" in s:
        family = "Android"
    elif "ios" in s or "iphone os" in s:
        family = "iOS"
    elif "harmony" in s or "emui" in s:
        family = "HuaweiOS"
    else:
        family = "Other"
    
    # Пытаемся вытащить версию типа "13", "13.1", "17.2" и т.п.
    m = re.search(r"(\d+)(?:\.(\d+))?", s)
    if m:
        major = float(m.group(1))  # нам достаточно целой части
    else:
        major = np.nan
    
    return family, major


def train_threshold_strategies(
    y_true: pd.Series,
    y_proba: np.ndarray,
    amounts: pd.Series,
    beta: float = 0.5,
    min_recall: float = 0.6,
) -> Dict[str, Any]:
    results = {}
    
    thr_f1 = pick_best_threshold_by_f1(y_true, y_proba)
    met_f1 = evaluate_predictions(y_true, y_proba, thr_f1, amounts, beta=beta)
    results["f1"] = {"threshold": thr_f1, "metrics": met_f1}
    
    thr_cost = pick_threshold_by_cost(y_true, y_proba, amounts)
    met_cost = evaluate_predictions(y_true, y_proba, thr_cost, amounts, beta=beta)
    results["cost"] = {"threshold": thr_cost, "metrics": met_cost}
    
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    thr_recall = 0.5
    if len(thr) > 0:
        mask = rec[:-1] >= min_recall
        if mask.any():
            idx = np.where(mask)[0][0]
            thr_recall = float(thr[idx])
        else:
            thr_recall = float(thr[-1]) if len(thr) > 0 else 0.5
    met_recall = evaluate_predictions(y_true, y_proba, thr_recall, amounts, beta=beta)
    results["high_recall"] = {"threshold": thr_recall, "metrics": met_recall}
    
    return results


def compute_model_predictions(
    X_row: pd.DataFrame,
    models: List[Dict],
    meta_model: LogisticRegression,
) -> Dict[str, float]:
    """
    Считает вероятности для одной строки признаков.
    Возвращает средние вероятности base-моделей и meta-модели.
    """
    X_val = X_row.copy()
    X_xgb = X_val.copy()
    X_cat = X_val.copy()
    
    cat_cols = X_val.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        X_xgb[col] = X_xgb[col].cat.codes
        X_cat[col] = X_cat[col].cat.codes
    
    lgb_probas: List[float] = []
    xgb_probas: List[float] = []
    cat_probas: List[float] = []
    
    for fold_models in models:
        lgb_probas.append(
            float(fold_models["lgbm"].predict(X_val, num_iteration=fold_models["lgbm"].best_iteration)[0])
        )
        xgb_probas.append(
            float(fold_models["xgb"].predict(xgb.DMatrix(X_xgb))[0])
        )
        cat_probas.append(
            float(fold_models["cat"].predict_proba(X_cat)[0, 1])
        )
    
    avg_lgb = float(np.mean(lgb_probas))
    avg_xgb = float(np.mean(xgb_probas))
    avg_cat = float(np.mean(cat_probas))
    
    base_pred = np.array([[avg_lgb, avg_xgb, avg_cat]])
    meta_proba = float(meta_model.predict_proba(base_pred)[0, 1])
    
    return {
        "avg_lgb": avg_lgb,
        "avg_xgb": avg_xgb,
        "avg_cat": avg_cat,
        "meta_proba": meta_proba,
    }


def compute_feature_baselines(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Сохраняем базовые значения признаков (медиана/мода), чтобы
    использовать их в локальной интерпретации.
    """
    baselines: Dict[str, Any] = {}
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            baselines[col] = float(series.median(skipna=True))
        elif pd.api.types.is_categorical_dtype(series):
            mode = series.mode(dropna=True)
            baselines[col] = mode.iloc[0] if not mode.empty else None
        else:
            baselines[col] = None
    return baselines


def get_local_feature_importance_delta(
    X_row: pd.DataFrame,
    models: List[Dict],
    meta_model: LogisticRegression,
    feature_baselines: Dict[str, Any],
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Локальная интерпретация: для каждой фичи смотрим,
    насколько изменится вероятность, если заменить фичу на базовое значение.
    """
    base_score = compute_model_predictions(X_row, models, meta_model)["meta_proba"]
    importances: List[Tuple[str, float]] = []
    
    for feat in X_row.columns:
        if feat not in feature_baselines:
            continue
        baseline = feature_baselines[feat]
        if baseline is None:
            continue
        
        X_tmp = X_row.copy()
        col_dtype = X_tmp[feat].dtype
        
        if pd.api.types.is_numeric_dtype(col_dtype):
            X_tmp.iloc[0, X_tmp.columns.get_loc(feat)] = baseline
        elif pd.api.types.is_categorical_dtype(col_dtype):
            if baseline not in X_tmp[feat].cat.categories:
                X_tmp[feat] = X_tmp[feat].cat.add_categories([baseline])
            X_tmp.at[X_tmp.index[0], feat] = baseline
        else:
            # Преобразуем в категорию, если возможно
            X_tmp[feat] = X_tmp[feat].astype("category")
            if baseline not in X_tmp[feat].cat.categories:
                X_tmp[feat] = X_tmp[feat].cat.add_categories([baseline])
            X_tmp.at[X_tmp.index[0], feat] = baseline
        
        new_score = compute_model_predictions(X_tmp, models, meta_model)["meta_proba"]
        importances.append((feat, abs(base_score - new_score)))
    
    importances.sort(key=lambda x: x[1], reverse=True)
    return importances[:top_n]


def explain_transaction_prediction(
    X_sample: pd.DataFrame,
    feature_importance: List[Tuple[str, float]],
    proba: float,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Объясняет предсказание для транзакции.
    """
    explanation = {
        "fraud_probability": float(proba),
        "fraud_percentage": float(proba * 100),
        "is_fraud": bool(proba >= threshold),
        "threshold": float(threshold),
        "top_risk_factors": [],
        "transaction_features": {}
    }
    
    # Топ факторы риска
    for feat_name, importance in feature_importance[:5]:
        feat_value = X_sample[feat_name].iloc[0] if feat_name in X_sample.columns else None
        # Конвертируем значение только если это число
        if feat_value is not None and pd.notna(feat_value):
            try:
                val = float(feat_value) if pd.api.types.is_numeric_dtype(type(feat_value)) else str(feat_value)
            except (ValueError, TypeError):
                val = str(feat_value)
        else:
            val = None
        explanation["top_risk_factors"].append({
            "feature": feat_name,
            "importance": float(importance),
            "value": val
        })
    
    # Важные фичи транзакции
    important_features = [
        "amount", "log_amount", "sqrt_amount", "is_night_tx", "is_weekend",
        "is_high_amount_vs_client", "txn_last_1h", "txn_last_24h",
        "zscore_amount", "burstiness", "fano_factor"
    ]
    
    for feat in important_features:
        if feat in X_sample.columns:
            val = X_sample[feat].iloc[0]
            if pd.notna(val):
                explanation["transaction_features"][feat] = float(val)
    
    return explanation


# ------------------------------------------------------------------------------
# Feature Engineering (same as v11.py but without prints)
# ------------------------------------------------------------------------------

def load_and_prepare_advanced() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], Dict]:
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

    # === OS NORMALIZATION ===
    os_parsed = data["last_os"].apply(normalize_os)
    data["os_family"] = os_parsed.apply(lambda x: x[0])
    data["os_major"] = os_parsed.apply(lambda x: x[1])
    data["os_major"] = pd.to_numeric(data["os_major"], errors="coerce")

    # === ИЗВЛЕЧЕНИЕ БРЕНДА ИЗ МОДЕЛИ ТЕЛЕФОНА ===
    def extract_brand(model: str) -> str:
        if not isinstance(model, str):
            return "Other"
        m = model.lower()
        if "iphone" in m or "ios" in m:
            return "Apple"
        if "samsung" in m or "sm-" in m or "galaxy" in m:
            return "Samsung"
        if "xiaomi" in m or "mi " in m or "redmi" in m or "poco" in m:
            return "Xiaomi"
        if "huawei" in m or "honor" in m:
            return "Huawei"
        if "vivo" in m:
            return "Vivo"
        if "oppo" in m or "realme" in m or "oneplus" in m:
            return "BBK"
        return "Other"
    
    data["device_brand"] = data["last_phone_model"].apply(extract_brand)
    
    # === ГРУППИРОВКА РЕДКИХ МОДЕЛЕЙ В OtherPhoneModel ===
    PHONE_MIN_COUNT = 50  # Порог для редких моделей
    phone_counts_raw = data["last_phone_model"].value_counts()
    rare_models = phone_counts_raw[phone_counts_raw < PHONE_MIN_COUNT].index
    data["last_phone_model_clean"] = data["last_phone_model"].where(
        ~data["last_phone_model"].isin(rare_models),
        "OtherPhoneModel"
    )
    # Заменяем оригинальную колонку на очищенную
    data["last_phone_model"] = data["last_phone_model_clean"]
    data = data.drop(columns=["last_phone_model_clean"])

    # Sort by time
    data = data.sort_values(["cst_id", "trans_datetime"]).reset_index(drop=True)

    # === BASIC AMOUNT FEATURES ===
    data["log_amount"] = np.log1p(data["amount"].clip(lower=0))
    data["sqrt_amount"] = np.sqrt(data["amount"].clip(lower=0))

    # === CUSTOMER HISTORY ===
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

    # === RULE-BASED ANOMALY FLAGS ===
    data["is_high_amount_vs_client"] = (data["amount_over_mean_past"] >= 5.0).astype(int)
    
    data["amount_rolling_90p"] = (
        data.groupby("cst_id")["amount"]
        .transform(lambda x: x.rolling(50, min_periods=1).quantile(0.9))
    )
    data["is_high_amount_percentile"] = (data["amount"] > data["amount_rolling_90p"]).astype(int)

    data["is_new_phone_model_for_client"] = (
        data.groupby("cst_id")["last_phone_model"].transform(lambda x: (x != x.shift(1)).fillna(True))
    ).astype(int)

    data["is_new_os_for_client"] = (
        data.groupby("cst_id")["last_os"].transform(lambda x: (x != x.shift(1)).fillna(True))
    ).astype(int)

    data["hour"] = data["trans_datetime"].dt.hour
    data["is_night_tx"] = data["hour"].between(0, 5).astype(int)
    data["is_first_tx_for_client"] = (data["cst_txn_count_past"] == 0).astype(int)

    data["day_of_week"] = data["trans_datetime"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    
    # === TIME ENTROPY ===
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    # === V9 OPTIMIZED FEATURES ===
    data["cst_night_tx_cumsum"] = data.groupby("cst_id")["is_night_tx"].cumsum()
    data["cst_night_tx_share"] = data["cst_night_tx_cumsum"] / (data["cst_txn_count_past"] + 1)
    
    data["cst_weekend_tx_cumsum"] = data.groupby("cst_id")["is_weekend"].cumsum()
    data["cst_weekend_tx_share"] = data["cst_weekend_tx_cumsum"] / (data["cst_txn_count_past"] + 1)
    
    data["sessions_7d_vs_30d_ratio"] = data["sessions_unique_7d"] / (data["sessions_unique_30d"] + 1)
    data["logins_7d_vs_30d_ratio"] = data["daily_logins_avg_7d"] / (data["daily_logins_avg_30d"] + 1)
    
    # === CLIP/TRANSFORM ===
    data["zscore_interval_7_vs_30"] = data["zscore_interval_7_vs_30"].clip(-5, 5)
    data["std_interval_30d_log"] = np.log1p(data["std_interval_30d"].clip(lower=0))
    data["fano_factor"] = data["fano_factor"].clip(lower=0, upper=100)
    data["burstiness"] = data["burstiness"].clip(-1, 1)
    
    # === BURST DETECTION ===
    data = data.sort_values(["cst_id", "trans_datetime"]).reset_index(drop=True)
    
    def rolling_count_by_time(group, window):
        group_indexed = group.set_index("trans_datetime")
        result = group_indexed.rolling(window, closed="left")["trans_id"].count()
        return pd.Series(result.values, index=group.index)
    
    txn_1h = (
        data.groupby("cst_id", group_keys=False)
        .apply(lambda df: rolling_count_by_time(df, "1h"))
    )
    data["txn_last_1h"] = txn_1h.values
    data["txn_last_1h"] = data["txn_last_1h"].fillna(0)
    
    txn_10min = (
        data.groupby("cst_id", group_keys=False)
        .apply(lambda df: rolling_count_by_time(df, "10min"))
    )
    data["txn_last_10min"] = txn_10min.values
    data["txn_last_10min"] = data["txn_last_10min"].fillna(0)
    
    txn_24h = (
        data.groupby("cst_id", group_keys=False)
        .apply(lambda df: rolling_count_by_time(df, "24h"))
    )
    data["txn_last_24h"] = txn_24h.values
    data["txn_last_24h"] = data["txn_last_24h"].fillna(0)
    
    # === DEVICE CHANGE DETECTION ===
    data["device_changed_recently"] = (
        (data["is_new_phone_model_for_client"] == 1) &
        (data["hours_since_prev_trans"] < 24)
    ).astype(int)
    
    data["os_changed_recently"] = (
        (data["is_new_os_for_client"] == 1) &
        (data["hours_since_prev_trans"] < 24)
    ).astype(int)
    
    # === ROLLING AMOUNT FEATURES ===
    def rolling_mean_by_time(group, window):
        group_indexed = group.set_index("trans_datetime")
        result = group_indexed.rolling(window, closed="left")["amount"].mean()
        return pd.Series(result.values, index=group.index)
    
    def rolling_std_by_time(group, window):
        group_indexed = group.set_index("trans_datetime")
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
    
    data["amount_deviation_rolling"] = np.abs(data["amount"] - data["amount_rolling_mean_7d"])
    data["amount_deviation_rolling_ratio"] = data["amount_deviation_rolling"] / (data["amount_rolling_mean_7d"] + 1)
    
    # === ROLLING ANOMALY SCORE ===
    data["zscore_amount"] = (
        (data["amount"] - data["amount_rolling_mean_7d"]) / (data["amount_rolling_std_7d"] + 1e-3)
    )
    data["zscore_amount"] = data["zscore_amount"].clip(-10, 10)
    
    data["amount_over_rolling_mean_7d"] = data["amount"] / (data["amount_rolling_mean_7d"] + 1e-3)
    
    data["amount_median_last_10"] = (
        data.groupby("cst_id")["amount"]
        .transform(lambda x: x.rolling(10, min_periods=1).median())
    )
    data["amount_over_median_last_10"] = data["amount"] / (data["amount_median_last_10"] + 1e-3)
    
    # === BEHAVIORAL FINGERPRINT ===
    data["cst_hour_mean"] = (
        data.groupby("cst_id")["hour"]
        .transform(lambda x: x.rolling(50, min_periods=1).mean())
    )
    data["hour_deviation_from_customer_mean"] = np.abs(data["hour"] - data["cst_hour_mean"])
    
    data["activity_profile_score"] = (
        data["hour_sin"] * 0.5 +
        data["hour_cos"] * 0.5 +
        (data["is_night_tx"] * 0.3) +
        (data["is_weekend"] * 0.2)
    )
    
    # === OS STATS ПО СЕМЕЙСТВАМ ===
    os_stats = (
        data.groupby("os_family")["os_major"]
        .agg(["median", "min", "max"])
        .reset_index()
    )
    os_stats_dict = {
        row["os_family"]: {
            "median": float(row["median"]) if pd.notna(row["median"]) else np.nan,
            "min": float(row["min"]) if pd.notna(row["min"]) else np.nan,
            "max": float(row["max"]) if pd.notna(row["max"]) else np.nan,
        }
        for _, row in os_stats.iterrows()
    }
    
    # Мержим stats обратно, чтобы посчитать осмысленные фичи
    data = data.merge(os_stats, on="os_family", how="left", suffixes=("", "_osstat"))
    
    # Насколько версия старее/новее медианной по семейству
    data["os_major_centered"] = data["os_major"] - data["median"]
    
    # Флаг "старой" и "очень старой" ОС внутри семейства
    data["os_is_old"] = (data["os_major"] <= (data["median"] - 1)).astype(int)
    data["os_is_very_old"] = (data["os_major"] <= (data["min"] + 1)).astype(int)
    
    # Чистим временные колонки stats
    data = data.drop(columns=["median", "min", "max"])

    # === FREQUENCY ENCODING С СГЛАЖИВАНИЕМ (Laplace smoothing) ===
    def smooth_freq(value, counts, n, alpha=1.0):
        """Laplace smoothing: даже для редких моделей не будет 0"""
        count = counts.get(value, 0)
        return (count + alpha) / (n + alpha * len(counts))
    
    # Частота по os_family (не по сырому last_os)
    os_family_counts = data["os_family"].value_counts().to_dict()
    n_rows = len(data)
    data["freq_os_family"] = data["os_family"].apply(
        lambda v: smooth_freq(v, os_family_counts, n_rows, alpha=1.0)
    )
    
    phone_counts = data["last_phone_model"].value_counts().to_dict()
    data["freq_last_phone_model"] = data["last_phone_model"].apply(
        lambda v: smooth_freq(v, phone_counts, n_rows, alpha=1.0)
    )
    
    # Лог-фичи для лучшей стабильности
    data["log_freq_os_family"] = np.log1p(data["freq_os_family"] * 1e6)
    data["log_freq_last_phone_model"] = np.log1p(data["freq_last_phone_model"] * 1e6)
    
    # === DEVICE TENURE (сколько дней клиент использует это устройство) ===
    data["device_first_seen"] = (
        data.groupby(["cst_id", "last_phone_model"])["trans_datetime"]
        .transform("min")
    )
    data["device_tenure_days"] = (
        (data["trans_datetime"] - data["device_first_seen"]).dt.total_seconds() / 86400.0
    )
    data["device_tenure_days"] = data["device_tenure_days"].clip(lower=0, upper=365*3)
    
    # === CLIENT TIME DENSITY ===
    data["cst_txns_per_day"] = (
        data.groupby("cst_id")["trans_datetime"]
        .transform(lambda x: x.rolling(30, min_periods=1).count() / 30.0)
    )
    
    # === EWM FEATURES ===
    data["ewm_amount_alpha_03"] = (
        data.groupby("cst_id")["amount"]
        .transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    )
    
    data["ewm_interval_alpha_05"] = (
        data.groupby("cst_id")["hours_since_prev_trans"]
        .transform(lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    )
    
    # === INTERACTIONS ===
    data["night_x_high_amount"] = data["is_night_tx"] * data["is_high_amount_vs_client"]
    data["new_device_x_high_amount"] = data["is_new_phone_model_for_client"] * data["is_high_amount_vs_client"]
    data["weekend_x_high_amount"] = data["is_weekend"] * data["is_high_amount_vs_client"]
    
    # === CATEGORY EMBEDDINGS (ТОЛЬКО ТЕЛЕФОН) ===
    cat_embed_cols = ["last_phone_model"]
    existing_cat_cols = [c for c in cat_embed_cols if c in data.columns]
    if existing_cat_cols:
        data, emb_artifacts = add_category_embeddings(
            data,
            categorical_cols=existing_cat_cols,
            n_components=6,
            prefix="catemb"
        )
    else:
        emb_artifacts = {}
    
    # Сохраняем frequency encoding и OS stats для инференса
    emb_artifacts["os_family_counts"] = os_family_counts
    emb_artifacts["phone_counts"] = phone_counts
    emb_artifacts["n_rows"] = n_rows
    emb_artifacts["known_phone_models"] = list(phone_counts.keys())
    emb_artifacts["phone_min_count"] = PHONE_MIN_COUNT
    emb_artifacts["os_stats"] = os_stats_dict
    
    # Глобальные статистики для обработки новых клиентов/устройств
    emb_artifacts["global_amount_mean"] = float(data["amount"].mean())
    emb_artifacts["global_amount_std"] = float(data["amount"].std())
    emb_artifacts["global_amount_median"] = float(data["amount"].median())
    emb_artifacts["median_device_tenure"] = float(data["device_tenure_days"].median())
    
    # === FINAL CLEANUP ===
    feature_drop_cols = [
        "cst_id", "trans_id", "trans_date", "trans_datetime", "target_id", "label",
        "hour", "day_of_week", "prev_transdatetime", "amount_cum_sum",
        "cst_night_tx_cumsum", "cst_weekend_tx_cumsum", "amount", "amount_rolling_90p",
        "cst_hour_mean", "device_first_seen",  # Временная колонка для расчета tenure
        "last_os"  # сырую OS не даём моделям
    ]
    
    X_full = data.drop(columns=[c for c in feature_drop_cols if c in data.columns])
    y_full = data["label"]

    # Категориальные признаки: телефон, бренд, семейство ОС
    categorical_cols = ["last_phone_model", "device_brand", "os_family"]
    for col in categorical_cols:
        if col in X_full.columns:
            X_full[col] = X_full[col].astype("category")

    num_cols = X_full.select_dtypes(include=[np.number]).columns
    X_full[num_cols] = X_full[num_cols].fillna(0)
    X_full[num_cols] = X_full[num_cols].replace([np.inf, -np.inf], 0)
    
    return data, X_full, y_full, categorical_cols, emb_artifacts


def prepare_single_transaction(
    tx_config: Dict[str, Any],
    data_template: pd.DataFrame,
    emb_artifacts: Optional[Dict[str, Any]],
    categorical_cols: List[str],
    client_history: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Подготавливает одну транзакцию для предсказания.
    
    ВАЖНО: В продакшне все фичи из customer_behavior УЖЕ ГОТОВЫ и приходят в tx_config!
    Эта функция только:
    1. Берет готовые фичи из конфига
    2. Рассчитывает derived-фичи (log_amount, hour_sin, embeddings)
    3. НЕ ищет ничего в датасете
    
    Args:
        tx_config: Конфиг транзакции с ГОТОВЫМИ фичами из feature store
        data_template: Шаблон для структуры колонок и дефолтных значений
        emb_artifacts: Артефакты для embeddings/encoding
        categorical_cols: Список категориальных колонок
        client_history: DEPRECATED - больше не используется
    """
    # Создаем пустой DataFrame с правильной структурой
    tx_row = pd.DataFrame([{}], columns=data_template.columns)
    
    # === ШАГ 1: КОПИРУЕМ ВСЕ ФИЧИ ИЗ КОНФИГА НАПРЯМУЮ ===
    for key, value in tx_config.items():
        if key in tx_row.columns:
            tx_row[key] = value
    
    # Убеждаемся что основные поля заполнены
    tx_row["trans_datetime"] = pd.to_datetime(tx_config["trans_datetime"])
    tx_row["trans_date"] = tx_row["trans_datetime"].iloc[0].date()
    tx_row["amount"] = float(tx_config["amount"])
    tx_row["cst_id"] = str(tx_config["cst_id"])
    
    # === ШАГ 2: РАССЧИТЫВАЕМ DERIVED-ФИЧИ (КОТОРЫХ НЕТ В КОНФИГЕ) ===
    
    # 2.1 Amount transformations
    tx_row["log_amount"] = np.log1p(tx_row["amount"].clip(lower=0))
    tx_row["sqrt_amount"] = np.sqrt(tx_row["amount"].clip(lower=0))
    
    # 2.2 Time features
    tx_row["hour"] = tx_row["trans_datetime"].dt.hour
    tx_row["is_night_tx"] = tx_row["hour"].between(0, 5).astype(int)
    tx_row["day_of_week"] = tx_row["trans_datetime"].dt.dayofweek
    tx_row["is_weekend"] = tx_row["day_of_week"].isin([5, 6]).astype(int)
    tx_row["hour_sin"] = np.sin(2 * np.pi * tx_row["hour"] / 24)
    tx_row["hour_cos"] = np.cos(2 * np.pi * tx_row["hour"] / 24)
    
    # 2.3 OS features
    if "last_os" in tx_config:
        family, major = normalize_os(str(tx_config["last_os"]))
        tx_row["os_family"] = family
        tx_row["os_major"] = major
    
    # 2.4 Device brand extraction
    if "last_phone_model" in tx_config:
        def extract_brand(model: str) -> str:
            if not isinstance(model, str):
                return "Other"
            m = model.lower()
            if "iphone" in m or "ios" in m:
                return "Apple"
            if "samsung" in m or "sm-" in m or "galaxy" in m:
                return "Samsung"
            if "xiaomi" in m or "mi " in m or "redmi" in m or "poco" in m:
                return "Xiaomi"
            if "huawei" in m or "honor" in m:
                return "Huawei"
            if "vivo" in m:
                return "Vivo"
            if "oppo" in m or "realme" in m or "oneplus" in m:
                return "BBK"
            return "Other"
        tx_row["device_brand"] = extract_brand(tx_config["last_phone_model"])
    
    # 2.5 Amount-derived features (если базовые фичи уже в конфиге)
    if "cst_amount_mean_past" in tx_config:
        tx_row["amount_diff_mean_past"] = tx_row["amount"] - tx_config["cst_amount_mean_past"]
        tx_row["amount_over_mean_past"] = tx_row["amount"] / (tx_config["cst_amount_mean_past"] + 1e-3)
    else:
        # Если нет в конфиге - используем глобальные средние
        global_mean = emb_artifacts.get("global_amount_mean", 15000.0) if emb_artifacts else 15000.0
        tx_row["cst_amount_mean_past"] = global_mean
        tx_row["amount_diff_mean_past"] = tx_row["amount"] - global_mean
        tx_row["amount_over_mean_past"] = tx_row["amount"] / (global_mean + 1e-3)
    
    tx_row["is_high_amount_vs_client"] = (tx_row["amount_over_mean_past"] >= 5.0).astype(int)
    
    # 2.6 Rolling amount features (если есть в конфиге)
    if "amount_rolling_mean_7d" in tx_config and "amount_rolling_std_7d" in tx_config:
        amt_mean_7d = tx_config["amount_rolling_mean_7d"]
        amt_std_7d = tx_config["amount_rolling_std_7d"]
        
        tx_row["amount_deviation_rolling"] = abs(tx_row["amount"].iloc[0] - amt_mean_7d)
        tx_row["amount_deviation_rolling_ratio"] = tx_row["amount_deviation_rolling"] / (amt_mean_7d + 1e-3)
        tx_row["zscore_amount"] = (tx_row["amount"].iloc[0] - amt_mean_7d) / (amt_std_7d + 1e-3)
        tx_row["zscore_amount"] = tx_row["zscore_amount"].clip(-10, 10)
        tx_row["amount_over_rolling_mean_7d"] = tx_row["amount"] / (amt_mean_7d + 1e-3)
    
    # 2.7 Ratio features
    if "sessions_unique_7d" in tx_config and "sessions_unique_30d" in tx_config:
        tx_row["sessions_7d_vs_30d_ratio"] = tx_config["sessions_unique_7d"] / (tx_config["sessions_unique_30d"] + 1)
    if "daily_logins_avg_7d" in tx_config and "daily_logins_avg_30d" in tx_config:
        tx_row["logins_7d_vs_30d_ratio"] = tx_config["daily_logins_avg_7d"] / (tx_config["daily_logins_avg_30d"] + 1)
    
    # 2.8 Interaction features
    tx_row["night_x_high_amount"] = tx_row["is_night_tx"] * tx_row["is_high_amount_vs_client"]
    tx_row["weekend_x_high_amount"] = tx_row["is_weekend"] * tx_row["is_high_amount_vs_client"]
    if "is_new_phone_model_for_client" in tx_config:
        tx_row["new_device_x_high_amount"] = int(tx_config["is_new_phone_model_for_client"]) * int(tx_row["is_high_amount_vs_client"].iloc[0])
        # hours_since_prev_trans в секундах! 24 часа = 86400 секунд
        tx_row["device_changed_recently"] = int(
            (tx_config["is_new_phone_model_for_client"] == 1) and
            (tx_config.get("hours_since_prev_trans", 999999) < 86400)
        )
        if "is_new_os_for_client" in tx_config:
            tx_row["os_changed_recently"] = int(
                (tx_config["is_new_os_for_client"] == 1) and
                (tx_config.get("hours_since_prev_trans", 999999) < 86400)
            )
    else:
        tx_row["new_device_x_high_amount"] = 0
        tx_row["device_changed_recently"] = 0
        tx_row["os_changed_recently"] = 0
    
    # 2.9 Additional transformations
    if "std_interval_30d" in tx_config:
        tx_row["std_interval_30d_log"] = np.log1p(max(tx_config["std_interval_30d"], 0))
    
    # 2.10 Activity profile score
    tx_row["activity_profile_score"] = (
        tx_row["hour_sin"] * 0.5 +
        tx_row["hour_cos"] * 0.5 +
        (tx_row["is_night_tx"] * 0.3) +
        (tx_row["is_weekend"] * 0.2)
    )
    
    # === ШАГ 3: CATEGORY EMBEDDINGS ===
    if emb_artifacts is not None and "encoder" in emb_artifacts:
        enc = emb_artifacts["encoder"]
        svd = emb_artifacts["svd"]
        cols = emb_artifacts["cols"]
        prefix = emb_artifacts["prefix"]
        
        # Проверяем что модель телефона известна, иначе заменяем на OtherPhoneModel
        known_phone_models = set(emb_artifacts.get("known_phone_models", []))
        phone_val = str(tx_row["last_phone_model"].iloc[0])
        if phone_val not in known_phone_models:
            tx_row["last_phone_model"] = "OtherPhoneModel"
        
        # Применяем embeddings
        ohe = enc.transform(tx_row[cols])
        emb = svd.transform(ohe)
        for i in range(emb.shape[1]):
            tx_row[f"{prefix}_{i}"] = emb[0, i]
    
    # === ШАГ 4: FREQUENCY ENCODING ===
    def smooth_freq_inference(value, counts, n, alpha=1.0):
        """Laplace smoothing для инференса"""
        count = counts.get(value, 0)
        return (count + alpha) / (n + alpha * len(counts))
    
    if emb_artifacts is not None and "os_family_counts" in emb_artifacts:
        n = emb_artifacts["n_rows"]
        os_family_counts = emb_artifacts["os_family_counts"]
        phone_counts = emb_artifacts["phone_counts"]
        
        phone_val = str(tx_row["last_phone_model"].iloc[0])
        os_family_val = str(tx_row["os_family"].iloc[0]) if "os_family" in tx_row.columns else "Unknown"
        
        # Сглаженные частоты
        tx_row["freq_os_family"] = smooth_freq_inference(os_family_val, os_family_counts, n, alpha=1.0)
        tx_row["freq_last_phone_model"] = smooth_freq_inference(phone_val, phone_counts, n, alpha=1.0)
        
        # Лог-фичи
        tx_row["log_freq_os_family"] = np.log1p(tx_row["freq_os_family"] * 1e6)
        tx_row["log_freq_last_phone_model"] = np.log1p(tx_row["freq_last_phone_model"] * 1e6)
    
    # === ШАГ 5: OS-DERIVED FEATURES ===
    if emb_artifacts is not None and "os_stats" in emb_artifacts:
        os_stats = emb_artifacts["os_stats"]
        family = tx_row["os_family"].iloc[0] if "os_family" in tx_row.columns else "Unknown"
        major = tx_row["os_major"].iloc[0] if "os_major" in tx_row.columns else np.nan
        stats = os_stats.get(family)
        
        if stats is not None and pd.notna(major):
            median = stats["median"]
            min_v = stats["min"]
            max_v = stats["max"]
            
            # Клипнем версию, чтобы супер-новые не улетали в космос
            major_clipped = max(min(major, max_v + 1), min_v - 1)
            tx_row["os_major"] = major_clipped
            tx_row["os_major_centered"] = major_clipped - median
            tx_row["os_is_old"] = int(major_clipped <= median - 1)
            tx_row["os_is_very_old"] = int(major_clipped <= min_v + 1)
        else:
            # Если ничего не знаем — нейтральные значения
            tx_row["os_major_centered"] = 0.0
            tx_row["os_is_old"] = 0
            tx_row["os_is_very_old"] = 0
    
    # === ШАГ 6: EWM FEATURES (если не в конфиге - рассчитываем) ===
    if "ewm_amount_alpha_03" not in tx_config:
        # Если EWM не в конфиге - используем текущее значение или среднее
        if "cst_amount_mean_past" in tx_config:
            tx_row["ewm_amount_alpha_03"] = tx_config["cst_amount_mean_past"]
        else:
            global_mean = emb_artifacts.get("global_amount_mean", 15000.0) if emb_artifacts else 15000.0
            tx_row["ewm_amount_alpha_03"] = global_mean
    
    if "ewm_interval_alpha_05" not in tx_config:
        # Если EWM interval не в конфиге - используем avg_interval
        if "avg_interval_30d" in tx_config:
            tx_row["ewm_interval_alpha_05"] = tx_config["avg_interval_30d"]
        elif "ewm_interval_7d" in tx_config:
            tx_row["ewm_interval_alpha_05"] = tx_config["ewm_interval_7d"]
        else:
            tx_row["ewm_interval_alpha_05"] = 18.0
    
    # === ШАГ 7: ДЕФОЛТНЫЕ ЗНАЧЕНИЯ ДЛЯ ОТСУТСТВУЮЩИХ ФИЧЕЙ ===
    # Если какие-то фичи не пришли в конфиге, заполняем безопасными дефолтами
    defaults = {
        "txn_last_1h": 0,
        "txn_last_10min": 0,
        "txn_last_24h": 0,
        "hours_since_prev_trans": 999999,
        "amount_median_last_10": tx_row["cst_amount_mean_past"].iloc[0] if "cst_amount_mean_past" in tx_row.columns else 15000,
        "amount_over_median_last_10": 1.0,
        "amount_cum_count": tx_config.get("cst_txn_count_past", 0),
        "cst_hour_mean": 12.0,
        "hour_deviation_from_customer_mean": 0.0,
        "is_high_amount_percentile": 0,
        "cst_txns_per_day": 1.5,
        "is_first_tx_for_client": 1 if tx_config.get("cst_txn_count_past", 0) == 0 else 0,
    }
    
    for col, default_val in defaults.items():
        if col in tx_row.columns and (col not in tx_config or pd.isna(tx_row[col].iloc[0])):
            tx_row[col] = default_val
    
    # === ШАГ 8: CLIPPING ДЛЯ СТАБИЛЬНОСТИ ===
    if "zscore_interval_7_vs_30" in tx_row.columns:
        tx_row["zscore_interval_7_vs_30"] = tx_row["zscore_interval_7_vs_30"].clip(-5, 5)
    if "burstiness" in tx_row.columns:
        tx_row["burstiness"] = tx_row["burstiness"].clip(-1, 1)
    if "fano_factor" in tx_row.columns:
        tx_row["fano_factor"] = tx_row["fano_factor"].clip(lower=0, upper=100)
    
    return tx_row


# ------------------------------------------------------------------------------
# Training (без принтов)
# ------------------------------------------------------------------------------

def train_cv_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    amounts: pd.Series,
    data_full: pd.DataFrame,
    categorical_cols: List[str],
    n_folds: int = 5,
) -> Tuple[List[Any], Dict]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_models = []
    all_oof_probas = np.zeros(len(y))
    oof_lgb = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_cat = np.zeros(len(y))
    fold_metrics = []
    
    spw = compute_class_weight(y)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_data = data_full.iloc[train_idx].copy()
        val_data = data_full.iloc[val_idx].copy()
        
        feature_drop_cols = [
            "cst_id", "trans_id", "trans_date", "trans_datetime", "target_id", "label",
            "hour", "day_of_week", "prev_transdatetime", "amount_cum_sum",
            "cst_night_tx_cumsum", "cst_weekend_tx_cumsum", "amount", "amount_rolling_90p",
            "cst_hour_mean", "device_first_seen",  # Временная колонка для расчета tenure
            "last_os"  # сырую OS не даём моделям
        ]
        
        X_train = train_data.drop(columns=[c for c in feature_drop_cols if c in train_data.columns])
        X_val = val_data.drop(columns=[c for c in feature_drop_cols if c in val_data.columns])
        
        for col in X_train.columns:
            if col not in X_val.columns:
                X_val[col] = 0
        for col in X_val.columns:
            if col not in X_train.columns:
                X_train[col] = 0
        X_train = X_train[X_val.columns]
        
        for col in categorical_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")
        
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[num_cols] = X_train[num_cols].fillna(0)
        X_train[num_cols] = X_train[num_cols].replace([np.inf, -np.inf], 0)
        X_val[num_cols] = X_val[num_cols].fillna(0)
        X_val[num_cols] = X_val[num_cols].replace([np.inf, -np.inf], 0)
        
        y_train = train_data["label"].reset_index(drop=True)
        y_val = val_data["label"].reset_index(drop=True)
        amounts_val = val_data["amount"].reset_index(drop=True)
        
        # Train LGBM
        params_lgbm = LGBM_PARAMS.copy()
        params_lgbm["scale_pos_weight"] = spw
        
        train_data_lgb = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
        val_data_lgb = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols)
        
        lgbm_model = lgb.train(
            params_lgbm,
            train_data_lgb,
            num_boost_round=1000,
            valid_sets=[val_data_lgb],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(False)
            ],
        )
        
        # Train XGBoost
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
        
        # Train CatBoost
        cat_model = CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.03,
            eval_metric='AUC',
            loss_function='Logloss',
            random_seed=42,
            verbose=False,
            task_type="CPU",
            scale_pos_weight=spw,
            early_stopping_rounds=50,
        )
        
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
            verbose=False,
        )
        
        # Predictions
        lgbm_proba = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)
        xgb_proba = xgb_model.predict(dval, iteration_range=(0, xgb_model.best_iteration))
        cat_proba = cat_model.predict_proba(X_val_cat)[:, 1]
        
        oof_lgb[val_idx] = lgbm_proba
        oof_xgb[val_idx] = xgb_proba
        oof_cat[val_idx] = cat_proba
        
        fold_proba = 0.4 * lgbm_proba + 0.3 * xgb_proba + 0.3 * cat_proba
        all_oof_probas[val_idx] = fold_proba
        
        fold_metrics.append(evaluate_predictions(y_val, fold_proba, 0.5, amounts_val))
        all_models.append({"lgbm": lgbm_model, "xgb": xgb_model, "cat": cat_model})
    
    # Thresholds
    threshold_results = train_threshold_strategies(y, all_oof_probas, amounts, beta=0.5)
    cv_threshold = threshold_results["f1"]["threshold"]
    cost_threshold = threshold_results["cost"]["threshold"]
    
    # Stacking
    base_oof = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
    meta_model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    meta_model.fit(base_oof, y)
    meta_oof_probas = meta_model.predict_proba(base_oof)[:, 1]
    meta_threshold_results = train_threshold_strategies(y, meta_oof_probas, amounts, beta=0.5)
    meta_f1_thr = meta_threshold_results["f1"]["threshold"]
    meta_cost_thr = meta_threshold_results["cost"]["threshold"]
    
    results = {
        "fold_metrics": fold_metrics,
        "cv_threshold": cv_threshold,
        "cost_threshold": cost_threshold,
        "oof_probas": all_oof_probas,
        "oof_lgb": oof_lgb,
        "oof_xgb": oof_xgb,
        "oof_cat": oof_cat,
        "meta_model": meta_model,
        "meta_f1_thr": meta_f1_thr,
        "meta_cost_thr": meta_cost_thr,
        "meta_oof_probas": meta_oof_probas,
    }
    
    return all_models, results


# ------------------------------------------------------------------------------
# Inference для одной транзакции
# ------------------------------------------------------------------------------

def predict_single_transaction(
    tx_config: Dict[str, Any],
    models: List[Dict],
    meta_model: LogisticRegression,
    data_template: pd.DataFrame,
    emb_artifacts: Optional[Dict[str, Any]],
    categorical_cols: List[str],
    feature_names: List[str],
    feature_baselines: Dict[str, Any],
    threshold: float = 0.5,
    client_history: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Предсказывает фрод для одной транзакции и возвращает объяснение.
    
    Args:
        client_history: История транзакций клиента из БД (опционально).
                       Если None - считается новым клиентом или используются предрасчитанные фичи.
    """
    # Подготовка транзакции БЕЗ поиска в тренировочном датасете!
    tx_row = prepare_single_transaction(
        tx_config, data_template, emb_artifacts, categorical_cols, client_history=client_history
    )
    
    # Извлекаем фичи
    feature_drop_cols = [
        "cst_id", "trans_id", "trans_date", "trans_datetime", "target_id", "label",
        "hour", "day_of_week", "prev_transdatetime", "amount_cum_sum",
        "cst_night_tx_cumsum", "cst_weekend_tx_cumsum", "amount", "amount_rolling_90p",
        "cst_hour_mean", "device_first_seen",  # Временная колонка для расчета tenure
        "last_os"  # сырую OS не даём моделям
    ]
    
    X_tx = tx_row.drop(columns=[c for c in feature_drop_cols if c in tx_row.columns])
    
    # Убеждаемся что все фичи есть
    for col in feature_names:
        if col not in X_tx.columns:
            X_tx[col] = 0
    
    X_tx = X_tx[feature_names]
    
    # Handle categorical
    for col in categorical_cols:
        if col in X_tx.columns:
            X_tx[col] = X_tx[col].astype("category")
    
    # Fill NaN
    num_cols = X_tx.select_dtypes(include=[np.number]).columns
    X_tx[num_cols] = X_tx[num_cols].fillna(0)
    X_tx[num_cols] = X_tx[num_cols].replace([np.inf, -np.inf], 0)
    
    model_preds = compute_model_predictions(X_tx, models, meta_model)
    avg_lgb = model_preds["avg_lgb"]
    avg_xgb = model_preds["avg_xgb"]
    avg_cat = model_preds["avg_cat"]
    ensemble_proba = 0.4 * avg_lgb + 0.3 * avg_xgb + 0.3 * avg_cat
    meta_proba = model_preds["meta_proba"]
    
    # Локальные факторы риска (delta-importance)
    feature_importance = get_local_feature_importance_delta(
        X_tx, models, meta_model, feature_baselines, top_n=10
    )
    
    # Объяснение
    explanation = explain_transaction_prediction(X_tx, feature_importance, meta_proba, threshold)
    
    result = {
        "transaction": {
            "cst_id": tx_config["cst_id"],
            "amount": float(tx_config["amount"]),
            "datetime": tx_config["trans_datetime"],
        },
        "predictions": {
            "lgbm": float(avg_lgb),
            "xgb": float(avg_xgb),
            "cat": float(avg_cat),
            "ensemble": float(ensemble_proba),
            "meta_model": float(meta_proba),
        },
        "final_prediction": {
            "fraud_probability": float(meta_proba),
            "fraud_percentage": float(meta_proba * 100),
            "is_fraud": bool(meta_proba >= threshold),
            "threshold": float(threshold),
        },
        "explanation": explanation,
    }
    
    return result


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    # 1. Загружаем данные и обучаем модели (тихо)
    data, X_full, y_full, categorical_cols, emb_artifacts = load_and_prepare_advanced()
    amounts_full = data["amount"]
    
    models, results = train_cv_ensemble(
        X=X_full,
        y=y_full,
        amounts=amounts_full,
        data_full=data,
        categorical_cols=categorical_cols,
        n_folds=5,
    )
    
    feature_baselines = compute_feature_baselines(X_full)
    
    # 2. Предсказание для транзакции из конфига
    # Используем более разумный порог для демо (можно менять)
    # threshold = results["meta_f1_thr"]  # ~0.99 (очень консервативный)
    # threshold = results["meta_cost_thr"]  # ~0.9442 (cost-optimized)
    threshold = 0.5  # Для демо - более чувствительный порог
    
    # ВАЖНО: Передаем client_history=None для демонстрации работы БЕЗ датасета!
    # В продакшне вы получите историю клиента из БД/feature store.
    # Для нового клиента или устройства используются медианные значения.
    prediction_result = predict_single_transaction(
        TRANSACTION_CONFIG,
        models,
        results["meta_model"],
        data,
        emb_artifacts,
        categorical_cols,
        list(X_full.columns),
        feature_baselines,
        threshold,
        client_history=None  # НЕ ищем в датасете!
    )
    
    # 3. Вывод результата
    print("=" * 80)
    print("FRAUD DETECTION RESULT")
    print("=" * 80)
    print(f"\n📋 Transaction Info:")
    print(f"   Client ID: {prediction_result['transaction']['cst_id']}")
    print(f"   Amount: ₸{prediction_result['transaction']['amount']:,.0f}")
    print(f"   DateTime: {prediction_result['transaction']['datetime']}")
    
    print(f"\n🎯 Fraud Prediction:")
    final = prediction_result["final_prediction"]
    print(f"   Fraud Probability: {final['fraud_percentage']:.2f}%")
    print(f"   Is Fraud: {'🚫 YES' if final['is_fraud'] else '✅ NO'}")
    print(f"   Threshold: {final['threshold']:.4f}")
    
    print(f"\n📊 Model Predictions:")
    preds = prediction_result["predictions"]
    print(f"   LightGBM: {preds['lgbm']*100:.2f}%")
    print(f"   XGBoost:  {preds['xgb']*100:.2f}%")
    print(f"   CatBoost: {preds['cat']*100:.2f}%")
    print(f"   Ensemble: {preds['ensemble']*100:.2f}%")
    print(f"   Meta-Model: {preds['meta_model']*100:.2f}%")
    
    print(f"\n💡 Top Risk Factors:")
    for i, factor in enumerate(prediction_result["explanation"]["top_risk_factors"], 1):
        if factor['value'] is not None:
            if isinstance(factor['value'], (int, float)):
                val_str = f" = {factor['value']:.3f}"
            else:
                val_str = f" = {factor['value']}"
        else:
            val_str = ""
        print(f"   {i}. {factor['feature']}{val_str} (importance: {factor['importance']*100:.2f}%)")
    
    print(f"\n📈 Key Transaction Features:")
    tx_features = prediction_result["explanation"]["transaction_features"]
    for feat, val in list(tx_features.items())[:10]:
        print(f"   {feat}: {val:.3f}")


if __name__ == "__main__":
    main()
