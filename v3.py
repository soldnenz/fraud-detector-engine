import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
import lightgbm as lgb

# =========================
# 1. Загрузка и подготовка данных
# =========================

CSV_PARAMS = dict(encoding="cp1251", sep=";")


def _parse_trans_date(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.strip("'\"")
        .replace("", pd.NA)
    )
    return pd.to_datetime(cleaned, errors="coerce")


def load_and_prepare():
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

    # --- 2. Drop duplicated header rows that appear in the CSV bodies ---
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
    data = data[data["label"].isin([0, 1])]  # drop any malformed rows
    data["label"] = data["label"].astype(int)
    for col in numeric_behavior_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Convert categorical fields: сначала заполняем строкой, потом category
    data["last_phone_model"] = data["last_phone_model"].fillna("Unknown")
    data["last_os"] = data["last_os"].fillna("Unknown")

    # --- 5. Сортировка по времени (важно для безопасных фич) ---
    data = data.sort_values(["cst_id", "trans_datetime"]).reset_index(drop=True)

    # --- 6. Фичи по клиенту: log_amount + агрегаты по прошлым транзакциям ---
    data["log_amount"] = np.log1p(data["amount"].clip(lower=0))

    customer_groups = data.groupby("cst_id", group_keys=False)
    data["amount_cum_sum"] = customer_groups["amount"].cumsum() - data["amount"]
    data["amount_cum_count"] = customer_groups.cumcount()
    # количество прошлых транзакций клиента
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

    # --- 7. Временная фича: часы с предыдущей транзакции клиента ---
    data["prev_transdatetime"] = customer_groups["trans_datetime"].shift(1)
    data["hours_since_prev_trans"] = (
        (data["trans_datetime"] - data["prev_transdatetime"])
        .dt.total_seconds()
        .div(3600.0)
    )
    data["hours_since_prev_trans"] = data["hours_since_prev_trans"].fillna(999999)
    data = data.drop(columns=["prev_transdatetime", "amount_cum_sum"])

    # --- 8. Фичи по получателю (target_id): история по получателю и fraud-rate ---
    # Сортируем по времени глобально (на всякий случай ещё раз)
    data = data.sort_values("trans_datetime").reset_index(drop=True)

    # Группировка по target_id
    target_groups = data.groupby("target_id", group_keys=False)

    # Кол-во прошлых транзакций на этот target
    data["target_txn_count_past"] = target_groups.cumcount()

    # Кумулятивная сумма фрода по target, без текущей строки
    data["target_fraud_cum_sum"] = target_groups["label"].cumsum() - data["label"]

    # Fraud rate по target в прошлом
    past_target_count = data["target_txn_count_past"].replace(0, np.nan)
    data["target_fraud_rate_past"] = data["target_fraud_cum_sum"] / past_target_count

    # Глобальный fraud-rate (для таргетов без истории)
    global_fraud_rate = data["label"].mean()
    data["target_fraud_rate_past"] = data["target_fraud_rate_past"].fillna(
        global_fraud_rate
    )

    # Можно выбросить служебную колонку
    data = data.drop(columns=["target_fraud_cum_sum"])

    # --- 9. Final preprocessing: drop IDs, set dtypes для модели ---
    data = data.sort_values("trans_datetime").reset_index(drop=True)

    feature_drop_cols = [
        "cst_id",
        "trans_id",
        "trans_date",
        "trans_datetime",
        "target_id",
        "label",
    ]
    X_full = data.drop(columns=feature_drop_cols)
    y_full = data["label"]

    categorical_cols = ["last_phone_model", "last_os"]

    # Категориальные -> category
    for col in categorical_cols:
        X_full[col] = X_full[col].astype("category")

    # Числовые: заполняем NaN нулями
    num_cols = X_full.select_dtypes(include=[np.number]).columns
    X_full[num_cols] = X_full[num_cols].fillna(0)

    return data, X_full, y_full, categorical_cols, numeric_behavior_cols


# =========================
# 2. Обучение и оценка одного варианта
# =========================

def train_and_evaluate_variant(
    name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    categorical_cols,
    base_params,
    use_class_weight=False,
    weight_factor=1.0,
    num_boost_round=2000,
    use_early_stopping=True,
):
    print("\n" + "=" * 80)
    print(f"=== {name} ===")
    print("=" * 80)

    print(
        f"Train size: {len(y_train)}, pos rate: {y_train.mean():.4f} "
        f"| Val size: {len(y_val)}, pos rate: {y_val.mean():.4f} "
        f"| Test size: {len(y_test)}, pos rate: {y_test.mean():.4f}"
    )

    params = base_params.copy()

    if use_class_weight:
        pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)
        scale_pos_weight = pos_weight * weight_factor
        params["scale_pos_weight"] = scale_pos_weight
        print(f"Using scale_pos_weight = {scale_pos_weight:.3f}")
    else:
        params.pop("scale_pos_weight", None)
        print("No class weights (scale_pos_weight disabled)")

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols)

    callbacks = []
    if use_early_stopping:
        callbacks.append(lgb.early_stopping(100))
        callbacks.append(lgb.log_evaluation(100))

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    best_iter = getattr(model, "best_iteration", num_boost_round)
    print(f"Best iteration: {best_iter}")

    # --- Threshold selection on validation ---
    y_val_proba = model.predict(X_val, num_iteration=best_iter)
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-9)
    if len(thresholds) > 0:
        candidate_scores = f1_arr[:-1]
        best_idx = int(np.argmax(candidate_scores))
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5
    print(f"Best threshold (validated): {best_threshold:.4f}")

    # --- Validation metrics ---
    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    print(f"[VAL]  ROC-AUC: {val_auc:.3f} | P: {val_precision:.3f} | R: {val_recall:.3f} | F1: {val_f1:.3f}")

    # --- Test metrics ---
    y_test_proba = model.predict(X_test, num_iteration=best_iter)
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    print(f"[TEST] ROC-AUC: {test_auc:.3f} | P: {test_precision:.3f} | R: {test_recall:.3f} | F1: {test_f1:.3f}")

    # --- Feature importance ---
    print("\nTop 15 feature importances (gain):")
    importance = model.feature_importance(importance_type="gain")
    feats_imps = sorted(zip(X_train.columns, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feats_imps[:15]:
        print(f"  {feat}: {imp:.1f}")

    return model


# =========================
# 3. Запуск всех экспериментов
# =========================

def main():
    data, X_full, y_full, categorical_cols, numeric_behavior_cols = load_and_prepare()

    # Базовые параметры модели
    base_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 7,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "seed": 42,
        "verbose": -1,
    }

    N = len(data)
    train_end = int(N * 0.6)
    val_end = int(N * 0.8)

    # ---------- Time-based split (60/20/20) ----------
    X_train_t = X_full.iloc[:train_end]
    y_train_t = y_full.iloc[:train_end]
    X_val_t = X_full.iloc[train_end:val_end]
    y_val_t = y_full.iloc[train_end:val_end]
    X_test_t = X_full.iloc[val_end:]
    y_test_t = y_full.iloc[val_end:]

    print("Time-based split positive rates:")
    print(f"  Train: {y_train_t.mean():.4f}, Val: {y_val_t.mean():.4f}, Test: {y_test_t.mean():.4f}")

    # ---------- Random split (60/20/20) ----------
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
    print(f"  Train: {y_train_r.mean():.4f}, Val: {y_val_r.mean():.4f}, Test: {y_test_r.mean():.4f}")

    # ===== Variant 1: Time-based, NO class weight =====
    train_and_evaluate_variant(
        name="Variant 1 - TimeSplit, no class weight",
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        X_test=X_test_t,
        y_test=y_test_t,
        categorical_cols=categorical_cols,
        base_params=base_params,
        use_class_weight=False,
        weight_factor=1.0,
    )

    # ===== Variant 2: Time-based, WITH class weight (0.5 * pos_weight) =====
    train_and_evaluate_variant(
        name="Variant 2 - TimeSplit, scale_pos_weight * 0.5",
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        X_test=X_test_t,
        y_test=y_test_t,
        categorical_cols=categorical_cols,
        base_params=base_params,
        use_class_weight=True,
        weight_factor=0.5,
    )

    # ===== Variant 3: Time-based, stronger model (less regularization), no weight =====
    stronger_params = base_params.copy()
    stronger_params.update(
        {
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 50,
            "lambda_l2": 1.0,
        }
    )
    train_and_evaluate_variant(
        name="Variant 3 - TimeSplit, stronger model, no class weight",
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_val_t,
        y_val=y_val_t,
        X_test=X_test_t,
        y_test=y_test_t,
        categorical_cols=categorical_cols,
        base_params=stronger_params,
        use_class_weight=False,
        weight_factor=1.0,
    )

    # ===== Variant 4: Random split, no class weight =====
    train_and_evaluate_variant(
        name="Variant 4 - RandomSplit, no class weight",
        X_train=X_train_r,
        y_train=y_train_r,
        X_val=X_val_r,
        y_val=y_val_r,
        X_test=X_test_r,
        y_test=y_test_r,
        categorical_cols=categorical_cols,
        base_params=base_params,
        use_class_weight=False,
        weight_factor=1.0,
    )

    # ===== Variant 5: Random split, WITH class weight =====
    train_and_evaluate_variant(
        name="Variant 5 - RandomSplit, with class weight",
        X_train=X_train_r,
        y_train=y_train_r,
        X_val=X_val_r,
        y_val=y_val_r,
        X_test=X_test_r,
        y_test=y_test_r,
        categorical_cols=categorical_cols,
        base_params=base_params,
        use_class_weight=True,
        weight_factor=1.0,
    )


if __name__ == "__main__":
    main()
