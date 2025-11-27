"""
Feature Store Module
Handles heavy offline aggregations and history features
"""

import pandas as pd
import numpy as np


def add_history_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add all heavy history-based features - ТОЧНО КАК В v8_ultimate.py:
    - Basic amount features (log, sqrt)
    - Customer aggregations (cumulative stats)
    - Time intervals
    - Target aggregations
    - New entity flags
    - Seasonality flags
    - V8 advanced features
    """
    
    # Sort by customer and time
    data = data.sort_values(["cst_id", "trans_datetime"]).reset_index(drop=True)
    
    # === BASIC AMOUNT FEATURES (from v8_ultimate.py lines 342-343) ===
    data["log_amount"] = np.log1p(data["amount"].clip(lower=0))
    data["sqrt_amount"] = np.sqrt(data["amount"].clip(lower=0))
    
    # === CUSTOMER AGGREGATIONS ===
    customer_groups = data.groupby("cst_id", group_keys=False)
    
    # Cumulative amount
    data["amount_cum_sum"] = customer_groups["amount"].cumsum() - data["amount"]
    data["amount_cum_count"] = customer_groups.cumcount()
    data["cst_txn_count_past"] = data["amount_cum_count"]
    
    # Customer mean amount (past only)
    past_count = data["amount_cum_count"].replace(0, np.nan)
    data["cst_amount_mean_past"] = data["amount_cum_sum"] / past_count
    global_amount_mean = data["amount"].mean()
    data["cst_amount_mean_past"] = data["cst_amount_mean_past"].fillna(global_amount_mean)
    
    # Amount vs customer mean
    data["amount_diff_mean_past"] = data["amount"] - data["cst_amount_mean_past"]
    data["amount_over_mean_past"] = data["amount"] / (data["cst_amount_mean_past"] + 1e-3)
    
    # Customer amount deviation (from v8_ultimate.py lines 410-411)
    data["cst_amount_deviation"] = np.abs(data["amount"] - data["cst_amount_mean_past"])
    data["cst_amount_deviation_ratio"] = data["cst_amount_deviation"] / (data["cst_amount_mean_past"] + 1)
    
    # === TIME INTERVALS ===
    data["prev_transdatetime"] = customer_groups["trans_datetime"].shift(1)
    data["hours_since_prev_trans"] = (
        (data["trans_datetime"] - data["prev_transdatetime"]).dt.total_seconds() / 3600.0
    )
    data["hours_since_prev_trans"] = data["hours_since_prev_trans"].fillna(999999)
    
    # === TARGET HISTORY ===
    data = data.sort_values("trans_datetime").reset_index(drop=True)
    target_groups = data.groupby("target_id", group_keys=False)
    
    data["target_txn_count_past"] = target_groups.cumcount()
    data["target_fraud_cum_sum"] = target_groups["label"].cumsum() - data["label"]
    
    # Target fraud rate with smoothing
    past_target_count = data["target_txn_count_past"].replace(0, np.nan)
    global_fraud_rate = data["label"].mean()
    data["target_fraud_rate_past"] = data["target_fraud_cum_sum"] / past_target_count
    data["target_fraud_rate_past"] = data["target_fraud_rate_past"].fillna(global_fraud_rate)
    
    # Log transform
    data["target_txn_count_past_log1p"] = np.log1p(data["target_txn_count_past"])
    
    # Smoothed fraud rate (Laplace smoothing)
    alpha = 10.0
    data["target_fraud_rate_past_smooth"] = (
        data["target_fraud_rate_past"] * data["target_txn_count_past"] + alpha * global_fraud_rate
    ) / (data["target_txn_count_past"] + alpha)
    
    # === NEW ENTITY FLAGS ===
    # New phone model for customer
    temp_phone = data.groupby(["cst_id", "last_phone_model"]).cumcount()
    data["is_new_phone_model_for_client"] = (temp_phone == 0).astype(int)
    
    # New OS for customer
    temp_os = data.groupby(["cst_id", "last_os"]).cumcount()
    data["is_new_os_for_client"] = (temp_os == 0).astype(int)
    
    # New target for customer
    temp_target = data.groupby(["cst_id", "target_id"]).cumcount()
    data["is_new_target_for_client"] = (temp_target == 0).astype(int)
    
    # === SEASONALITY FLAGS ===
    data["hour"] = data["trans_datetime"].dt.hour
    data["is_night_tx"] = data["hour"].between(0, 5).astype(int)
    
    data["day_of_week"] = data["trans_datetime"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    
    # First transaction for customer
    data["is_first_tx_for_client"] = (data["cst_txn_count_past"] == 0).astype(int)
    
    # === RULE-BASED ANOMALY FLAGS (from v8_ultimate.py lines 386-388) ===
    high_amount_thr = data["amount"].quantile(0.99)
    data["is_high_amount_global"] = (data["amount"] >= high_amount_thr).astype(int)
    data["is_high_amount_vs_client"] = (data["amount_over_mean_past"] >= 5.0).astype(int)
    
    # === V8 ADVANCED FEATURES ===
    # Target diversity (from v8_ultimate.py lines 414-419)
    data["target_same_as_prev"] = (
        data.groupby("cst_id")["target_id"].shift(1) == data["target_id"]
    ).astype(int)
    
    data["cst_new_targets_ratio"] = data["is_new_target_for_client"] / (data["cst_txn_count_past"] + 1)
    
    # Seasonality ratios for customer (from v8_ultimate.py lines 422-428)
    data["cst_night_tx_cumsum"] = data.groupby("cst_id")["is_night_tx"].cumsum()
    data["cst_night_tx_share"] = data["cst_night_tx_cumsum"] / (data["cst_txn_count_past"] + 1)
    
    data["cst_weekend_tx_cumsum"] = data.groupby("cst_id")["is_weekend"].cumsum()
    data["cst_weekend_tx_share"] = data["cst_weekend_tx_cumsum"] / (data["cst_txn_count_past"] + 1)
    
    # Behavior ratios (from v8_ultimate.py lines 431-435)
    data["sessions_7d_vs_30d_ratio"] = data["sessions_unique_7d"] / (data["sessions_unique_30d"] + 1)
    data["logins_7d_vs_30d_ratio"] = data["daily_logins_avg_7d"] / (data["daily_logins_avg_30d"] + 1)
    data["is_login_spike"] = (data["daily_logins_avg_7d"] > data["daily_logins_avg_30d"] * 2).astype(int)
    
    # Extended interactions (from v8_ultimate.py lines 438-456)
    data["amount_x_new_target"] = data["amount"] * data["is_new_target_for_client"]
    data["amount_x_high_fraud_rate"] = data["amount"] * data["target_fraud_rate_past_smooth"]
    data["night_x_high_amount"] = data["is_night_tx"] * data["is_high_amount_vs_client"]
    data["new_device_x_high_amount"] = data["is_new_phone_model_for_client"] * data["is_high_amount_vs_client"]
    data["weekend_x_high_amount"] = data["is_weekend"] * data["is_high_amount_vs_client"]
    
    # NEW interactions
    data["high_amount_x_new_os"] = data["is_high_amount_vs_client"] * data["is_new_os_for_client"]
    data["first_tx_x_high_amount"] = data["is_first_tx_for_client"] * data["is_high_amount_global"]
    data["new_target_x_high_fraud_target"] = data["is_new_target_for_client"] * data["target_fraud_rate_past_smooth"]
    
    # Complex interactions (from v8_ultimate.py lines 450-456)
    data["risk_score"] = (
        data["is_high_amount_vs_client"] + 
        data["is_new_target_for_client"] + 
        data["is_night_tx"] + 
        data["is_new_phone_model_for_client"]
    )
    data["risk_x_target_fraud"] = data["risk_score"] * data["target_fraud_rate_past_smooth"]
    
    return data

