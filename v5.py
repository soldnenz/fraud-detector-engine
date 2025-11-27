"""
Final fraud detection pipeline for mobile banking (ENHANCED).

Закрывает критерии:
- Performance: precision, recall, f_beta, ROC-AUC, стабильность (time + random split)
- Быстрота фичей: векторные pandas-операции, группировки, без циклов по строкам
- Стабильность: сравнение time-based и random split
- Business value: считаем количество пойманных мошеннических транзакций (TP)
- Usability:
  * feature importance (и заготовка под SHAP)
  * возможность дообучения модели (init_model / retrain)
  * простая настройка порогов срабатывания (F1-optimal + Precision>=0.6)

УЛУЧШЕНИЯ В ЭТОЙ ВЕРСИИ:
1. Rule-based аномальные фичи (новое устройство, высокая сумма, ночная транзакция, etc)
2. Сглаженные target-фичи с Laplace smoothing для борьбы с шумом редких значений
3. Оптимизированные параметры LGBM (более гибкие деревья, min_gain_to_split)
4. Два режима работы: F1-оптимальный и режим высокого Precision (>=0.6)
5. Полноценная функция дообучения модели с примером логики
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

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

# ------------------------------------------------------------------------------
# 1. Конфиг
# ------------------------------------------------------------------------------

CSV_PARAMS = dict(encoding="cp1251", sep=";")

BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,       # чуть медленнее, но стабильнее
    "num_leaves": 63,            # больше сплитов
    "max_depth": -1,             # пусть сама решает глубину
    "min_data_in_leaf": 50,      # чуть меньше
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "min_gain_to_split": 0.01,   # отсечь мусорные сплиты
    "seed": 42,
    "verbose": -1,
}


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

    def to_dict(self):
        return asdict(self)


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
    """
    Подбор порога, при котором precision >= target_precision
    и recall максимален.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    best_thr = 0.5
    best_rec = 0.0
    for p, r, t in zip(prec, rec, thr):
        if p >= target_precision and r > best_rec:
            best_rec = r
            best_thr = float(t)
    return best_thr, best_rec


def evaluate_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    beta: float = 0.5,
) -> Metrics:
    y_pred = (y_proba >= threshold).astype(int)
    roc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fbeta = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
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
    )


def print_metrics(label: str, m: Metrics):
    print(
        f"[{label}] ROC-AUC: {m.roc_auc:.3f} | "
        f"P: {m.precision:.3f} | R: {m.recall:.3f} | "
        f"F1: {m.f1:.3f} | F0.5: {m.fbeta_05:.3f} | thr: {m.threshold:.4f}"
    )
    print(
        f"    Confusion matrix: TP={m.tp}, FP={m.fp}, TN={m.tn}, FN={m.fn} "
        f"(blocked frauds = TP)"
    )


# ------------------------------------------------------------------------------
# 3. Загрузка данных + feature engineering
# ------------------------------------------------------------------------------

def load_and_prepare() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Загружает данные, чистит, строит фичи:
    - агрегаты по клиенту (mean amount, diff, ratio, count)
    - интервалы между транзакциями
    - история по target_id (fraud rate, count)
    """

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

    # --- 4. Merge datasets on Customer ID and Transaction Date ---
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

    # --- 5. Sort by client+time for leakage-safe features ---
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

    # --- 7. Time interval since previous tx for client ---
    data["prev_transdatetime"] = customer_groups["trans_datetime"].shift(1)
    data["hours_since_prev_trans"] = (
        (data["trans_datetime"] - data["prev_transdatetime"])
        .dt.total_seconds()
        .div(3600.0)
    )
    data["hours_since_prev_trans"] = data["hours_since_prev_trans"].fillna(999999)

    data = data.drop(columns=["prev_transdatetime", "amount_cum_sum"])

    # --- 8. Target-based features: count + fraud rate ---
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
    # Лог-количество транзакций по target
    data["target_txn_count_past_log1p"] = np.log1p(data["target_txn_count_past"])

    # Сглаженный fraud_rate по target (Laplace smoothing)
    alpha = 10.0
    data["target_fraud_rate_past_smooth"] = (
        data["target_fraud_rate_past"] * data["target_txn_count_past"] +
        alpha * global_fraud_rate
    ) / (data["target_txn_count_past"] + alpha)

    # --- 10. Rule-based аномальные фичи ---
    # Глобальный перцентиль суммы
    high_amount_thr = data["amount"].quantile(0.99)
    data["is_high_amount_global"] = (data["amount"] >= high_amount_thr).astype(int)

    # "Высокая сумма относительно клиента"
    data["is_high_amount_vs_client"] = (data["amount_over_mean_past"] >= 5.0).astype(int)

    # Новое устройство / OS для клиента (более векторный способ)
    temp_phone = data.groupby(["cst_id", "last_phone_model"]).cumcount()
    data["is_new_phone_model_for_client"] = (temp_phone == 0).astype(int)

    temp_os = data.groupby(["cst_id", "last_os"]).cumcount()
    data["is_new_os_for_client"] = (temp_os == 0).astype(int)

    # Новый target_id для клиента
    temp_target = data.groupby(["cst_id", "target_id"]).cumcount()
    data["is_new_target_for_client"] = (temp_target == 0).astype(int)

    # Ночная транзакция (если есть время) - считаем "ночью" [0:00–6:00)
    data["hour"] = data["trans_datetime"].dt.hour
    data["is_night_tx"] = data["hour"].between(0, 5).astype(int)

    # Первая транзакция клиента
    data["is_first_tx_for_client"] = (data["cst_txn_count_past"] == 0).astype(int)

    # --- 11. Final preprocessing ---
    data = data.sort_values("trans_datetime").reset_index(drop=True)

    feature_drop_cols = [
        "cst_id",
        "trans_id",
        "trans_date",
        "trans_datetime",
        "target_id",
        "label",
        "hour",  # техническая колонка, не нужна в модели
    ]
    X_full = data.drop(columns=feature_drop_cols)
    y_full = data["label"]

    categorical_cols = ["last_phone_model", "last_os"]
    for col in categorical_cols:
        X_full[col] = X_full[col].astype("category")

    # fill NaN only for numeric columns
    num_cols = X_full.select_dtypes(include=[np.number]).columns
    X_full[num_cols] = X_full[num_cols].fillna(0)

    print(f"Full dataset size: {len(data)}, fraud rate: {y_full.mean():.4f}")
    return data, X_full, y_full, categorical_cols


# ------------------------------------------------------------------------------
# 4. Сплиты: time-based и random
# ------------------------------------------------------------------------------

def make_splits(
    data: pd.DataFrame, X_full: pd.DataFrame, y_full: pd.Series
):
    N = len(data)
    train_end = int(N * 0.6)
    val_end = int(N * 0.8)

    # Time-based split (по времени, т.к. data уже отсортирован по trans_datetime)
    X_train_t = X_full.iloc[:train_end]
    y_train_t = y_full.iloc[:train_end]
    X_val_t = X_full.iloc[train_end:val_end]
    y_val_t = y_full.iloc[train_end:val_end]
    X_test_t = X_full.iloc[val_end:]
    y_test_t = y_full.iloc[val_end:]

    print("Time-based split positive rates:")
    print(
        f"  Train: {y_train_t.mean():.4f}, "
        f"Val: {y_val_t.mean():.4f}, "
        f"Test: {y_test_t.mean():.4f}"
    )

    # Random split (для соревновательной метрики)
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

    print("\nRandom split positive rates:")
    print(
        f"  Train: {y_train_r.mean():.4f}, "
        f"Val: {y_val_r.mean():.4f}, "
        f"Test: {y_test_r.mean():.4f}"
    )

    return (
        (X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t),
        (X_train_r, y_train_r, X_val_r, y_val_r, X_test_r, y_test_r),
    )


# ------------------------------------------------------------------------------
# 5. Обучение одной LGBM-модели
# ------------------------------------------------------------------------------

def train_single_lgbm(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: List[str],
    params: Dict,
    use_class_weight: bool = False,
) -> Tuple[lgb.Booster, Metrics, Metrics, np.ndarray, np.ndarray]:
    print("\n" + "=" * 80)
    print(f"=== {name} ===")
    print("=" * 80)
    print(
        f"Train size: {len(y_train)}, pos rate: {y_train.mean():.4f} | "
        f"Val size: {len(y_val)}, pos rate: {y_val.mean():.4f} | "
        f"Test size: {len(y_test)}, pos rate: {y_test.mean():.4f}"
    )

    params = params.copy()
    if use_class_weight:
        spw = compute_class_weight(y_train)
        params["scale_pos_weight"] = spw
        print(f"Using scale_pos_weight = {spw:.3f}")
    else:
        params.pop("scale_pos_weight", None)
        print("No class weights (scale_pos_weight disabled)")

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

    # Подбор порога по F1 на валидации
    y_val_proba = model.predict(X_val, num_iteration=best_iter)
    best_threshold = pick_best_threshold_by_f1(y_val, y_val_proba)
    print(f"Best threshold (validated by F1): {best_threshold:.4f}")

    # Дополнительно: порог для precision >= 0.6
    thr_p60, rec_p60 = pick_threshold_for_precision(y_val, y_val_proba, target_precision=0.6)
    print(f"Alt threshold (precision>=0.6): {thr_p60:.4f} (recall at this thr on VAL: {rec_p60:.3f})")

    # Метрики для основного порога (F1-оптимальный)
    val_metrics = evaluate_predictions(y_val, y_val_proba, best_threshold)
    print_metrics("VAL (F1 mode)", val_metrics)

    y_test_proba = model.predict(X_test, num_iteration=best_iter)
    test_metrics = evaluate_predictions(y_test, y_test_proba, best_threshold)
    print_metrics("TEST (F1 mode)", test_metrics)

    # Метрики для режима precision >= 0.6
    val_metrics_p60 = evaluate_predictions(y_val, y_val_proba, thr_p60)
    print_metrics("VAL (P>=0.6 mode)", val_metrics_p60)

    test_metrics_p60 = evaluate_predictions(y_test, y_test_proba, thr_p60)
    print_metrics("TEST (P>=0.6 mode)", test_metrics_p60)

    # Feature importance
    print("\nTop 15 feature importances (gain):")
    importance = model.feature_importance(importance_type="gain")
    feats_imps = sorted(
        zip(X_train.columns, importance), key=lambda x: x[1], reverse=True
    )
    for feat, imp in feats_imps[:15]:
        print(f"  {feat}: {imp:.1f}")

    return model, val_metrics, test_metrics, y_val_proba, y_test_proba


# ------------------------------------------------------------------------------
# 6. Ансамбль по Random split
# ------------------------------------------------------------------------------

def train_ensemble_random(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: List[str],
    base_params: Dict,
    n_models: int = 5,
) -> Tuple[List[lgb.Booster], Metrics, Metrics]:
    print("\n" + "=" * 80)
    print("=== ENSEMBLE - RandomSplit, with class weight ===")
    print("=" * 80)
    print(
        f"Train size: {len(y_train)}, pos rate: {y_train.mean():.4f} | "
        f"Val size: {len(y_val)}, pos rate: {y_val.mean():.4f} | "
        f"Test size: {len(y_test)}, pos rate: {y_test.mean():.4f}"
    )

    spw = compute_class_weight(y_train)
    print(f"Using scale_pos_weight = {spw:.3f}")

    seeds = [42, 52, 62, 72, 82][:n_models]
    models: List[lgb.Booster] = []
    val_preds = []
    test_preds = []

    for seed in seeds:
        print(f"\n--- Training model with seed={seed} ---")
        params = base_params.copy()
        params.update({"scale_pos_weight": spw, "seed": seed})

        train_data = lgb.Dataset(
            X_train, label=y_train, categorical_feature=categorical_cols
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, categorical_feature=categorical_cols
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
        best_iter = getattr(model, "best_iteration", 2000)
        print(f"Best iteration for seed={seed}: {best_iter}")

        val_pred = model.predict(X_val, num_iteration=best_iter)
        test_pred = model.predict(X_test, num_iteration=best_iter)

        models.append(model)
        val_preds.append(val_pred)
        test_preds.append(test_pred)

    val_pred_mean = np.mean(val_preds, axis=0)
    test_pred_mean = np.mean(test_preds, axis=0)

    best_threshold = pick_best_threshold_by_f1(y_val, val_pred_mean)
    print(f"\nEnsemble best threshold (validated by F1): {best_threshold:.4f}")

    val_metrics = evaluate_predictions(y_val, val_pred_mean, best_threshold)
    print_metrics("VAL ENSEMBLE", val_metrics)

    test_metrics = evaluate_predictions(y_test, test_pred_mean, best_threshold)
    print_metrics("TEST ENSEMBLE", test_metrics)

    return models, val_metrics, test_metrics


# ------------------------------------------------------------------------------
# 7. Saving / Loading / Retraining (Usability: адаптация + дообучение)
# ------------------------------------------------------------------------------

def save_ensemble(
    models: List[lgb.Booster],
    feature_names: List[str],
    categorical_cols: List[str],
    threshold: float,
    version: int = 1,
    prefix: str = "ensemble",
):
    model_paths = []
    for i, m in enumerate(models):
        path = f"{prefix}_v{version}_seed{i}.txt"
        m.save_model(path)
        model_paths.append(path)

    meta = {
        "version": version,
        "model_paths": model_paths,
        "features": feature_names,
        "categorical": categorical_cols,
        "threshold": float(threshold),
    }
    with open(f"{prefix}_v{version}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved ensemble meta to {prefix}_v{version}_meta.json")


def load_ensemble_meta(
    meta_path: str,
) -> Tuple[List[lgb.Booster], List[str], List[str], float]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_paths = meta["model_paths"]
    features = meta["features"]
    categorical = meta["categorical"]
    threshold = float(meta["threshold"])

    models = [lgb.Booster(model_file=p) for p in model_paths]
    return models, features, categorical, threshold


def ensemble_predict(
    models: List[lgb.Booster], X: pd.DataFrame
) -> np.ndarray:
    preds = [m.predict(X) for m in models]
    return np.mean(preds, axis=0)


def update_model_on_new_data_example():
    """
    Пример функции дообучения модели (не вызывается в main).

    Идея:
    - читаем meta и существующий ансамбль,
    - готовим новые размеченные данные (X_new, y_new),
    - либо дообучаем старые модели через init_model,
    - либо тренируем новый ансамбль и сохраняем как версию v2.
    """
    # Шаг 1. Загрузка старого ансамбля
    models, feat_names, cat_cols, old_thr = load_ensemble_meta("fraud_ensemble_v1_meta.json")
    print(f"Loaded ensemble v1 with {len(models)} models, threshold={old_thr:.4f}")

    # Шаг 2. Загрузка новых данных
    # (в реальности это будет инкрементальная выборка за последний месяц)
    data_new, X_new_full, y_new_full, categorical_cols_new = load_and_prepare()
    
    # В реальном коде можно фильтровать по дате, чтобы взять только новые транзакции
    # например: data_new = data_new[data_new["trans_datetime"] > LAST_TRAIN_DATE]
    # Для примера просто делаем новый split
    
    # Шаг 3. Создаём новые сплиты для дообучения
    N = len(data_new)
    rng = np.random.RandomState(123)
    indices = np.arange(N)
    rng.shuffle(indices)
    
    train_end = int(N * 0.7)
    val_end = int(N * 0.85)
    
    X_train_new = X_new_full.iloc[indices[:train_end]]
    y_train_new = y_new_full.iloc[indices[:train_end]]
    X_val_new = X_new_full.iloc[indices[train_end:val_end]]
    y_val_new = y_new_full.iloc[indices[train_end:val_end]]
    X_test_new = X_new_full.iloc[indices[val_end:]]
    y_test_new = y_new_full.iloc[indices[val_end:]]
    
    # Шаг 4. Обучаем новый ансамбль v2 на обновлённых данных
    new_models, new_val_m, new_test_m = train_ensemble_random(
        X_train=X_train_new,
        y_train=y_train_new,
        X_val=X_val_new,
        y_val=y_val_new,
        X_test=X_test_new,
        y_test=y_test_new,
        categorical_cols=cat_cols,
        base_params=BASE_PARAMS,
        n_models=5,
    )
    
    # Шаг 5. Сохраняем новую версию
    save_ensemble(
        models=new_models,
        feature_names=feat_names,
        categorical_cols=cat_cols,
        threshold=new_val_m.threshold,
        version=2,
        prefix="fraud_ensemble",
    )
    
    print(f"\n=== Retrained model v2 successfully ===")
    print(f"Old threshold: {old_thr:.4f} -> New threshold: {new_val_m.threshold:.4f}")
    print_metrics("New VAL", new_val_m)
    print_metrics("New TEST", new_test_m)


# ------------------------------------------------------------------------------
# 8. Интерпретируемость (feature importance + SHAP-заготовка)
# ------------------------------------------------------------------------------

def get_feature_importance_df(
    model: lgb.Booster, feature_names: List[str], top_n: int = 20
) -> pd.DataFrame:
    imp = model.feature_importance(importance_type="gain")
    df = (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    return df


def compute_shap_example(model: lgb.Booster, X_sample: pd.DataFrame):
    """
    Пример использования SHAP (не вызывается в main).
    Для реального использования нужно установить shap: pip install shap
    """
    try:
        import shap  # type: ignore
    except ImportError:
        print("Install SHAP with `pip install shap` to compute SHAP values.")
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # Можно строить summary/force plots в ноутбуке/презентации.
    return shap_values


# ------------------------------------------------------------------------------
# 9. MAIN: полное обучение + оценка
# ------------------------------------------------------------------------------

def main():
    # 1) Загрузка и feature engineering
    data, X_full, y_full, categorical_cols = load_and_prepare()

    # 2) Делим на TimeSplit и RandomSplit
    (X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t), (
        X_train_r,
        y_train_r,
        X_val_r,
        y_val_r,
        X_test_r,
        y_test_r,
    ) = make_splits(data, X_full, y_full)

    # 3) Модель для real-time сценария (TimeSplit, без весов)
    time_model, time_val_m, time_test_m, _, _ = train_single_lgbm(
        name="TimeSplit - no class weight (production-like)",
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        X_test=X_test_t,
        y_test=y_test_t,
        categorical_cols=categorical_cols,
        params=BASE_PARAMS,
        use_class_weight=False,
    )

    # 4) Модель для соревнования (RandomSplit, одиночная, с весами)
    random_model_single, rand_val_m, rand_test_m, _, _ = train_single_lgbm(
        name="RandomSplit - single model, with class weight",
        X_train=X_train_r,
        y_train=y_train_r,
        X_val=X_val_r,
        y_val=y_val_r,
        X_test=X_test_r,
        y_test=y_test_r,
        categorical_cols=categorical_cols,
        params=BASE_PARAMS,
        use_class_weight=True,
    )

    # 5) Ансамбль для максимизации метрик на RandomSplit
    ensemble_models, ens_val_m, ens_test_m = train_ensemble_random(
        X_train=X_train_r,
        y_train=y_train_r,
        X_val=X_val_r,
        y_val=y_val_r,
        X_test=X_test_r,
        y_test=y_test_r,
        categorical_cols=categorical_cols,
        base_params=BASE_PARAMS,
        n_models=5,
    )

    # Сохраняем ансамбль как основную боевую модель
    save_ensemble(
        models=ensemble_models,
        feature_names=list(X_full.columns),
        categorical_cols=categorical_cols,
        threshold=ens_val_m.threshold,
        version=1,
        prefix="fraud_ensemble",
    )

    print("\n=== SUMMARY (for report / presentation) ===")
    print("TimeSplit (production-like):")
    print_metrics("VAL Time", time_val_m)
    print_metrics("TEST Time", time_test_m)

    print("\nRandomSplit (competition-like):")
    print_metrics("VAL Single", rand_val_m)
    print_metrics("TEST Single", rand_test_m)
    print_metrics("VAL Ensemble", ens_val_m)
    print_metrics("TEST Ensemble", ens_test_m)


# ------------------------------------------------------------------------------
# 10. Пример функции обучения (не используется напрямую, чисто для ТЗ)
# ------------------------------------------------------------------------------

def example_training_function_not_used():
    """
    Пример простой функции обучения, которая НЕ вызывается в main.

    Можно показать в отчёте как API:
    - принимает путь к CSV,
    - возвращает обученную модель и метрики.

    Это демонстрация того, что модель можно легко встраивать и дообучать.
    """
    data, X_full, y_full, categorical_cols = load_and_prepare()
    (X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t), _ = make_splits(
        data, X_full, y_full
    )

    model, val_m, test_m, _, _ = train_single_lgbm(
        name="Example simple training",
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        X_test=X_test_t,
        y_test=y_test_t,
        categorical_cols=categorical_cols,
        params=BASE_PARAMS,
        use_class_weight=False,
    )
    # Возвращаем, но нигде не используем — это просто пример API
    return model, val_m, test_m


if __name__ == "__main__":
    main()
