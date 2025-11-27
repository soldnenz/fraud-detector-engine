"""
Data Loading Module
Handles raw data loading and basic cleaning
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

CSV_PARAMS = dict(encoding="cp1251", sep=";")


def _parse_date(s: pd.Series) -> pd.Series:
    """Parse date strings with various formats"""
    cleaned = (
        s.astype(str)
         .str.strip()
         .str.strip("'\"")
         .replace("", pd.NA)
    )
    return pd.to_datetime(cleaned, errors="coerce")


def load_raw_transactions(path: str) -> pd.DataFrame:
    """Load and clean transactions data"""
    df = pd.read_csv(path, **CSV_PARAMS)
    df.columns = [
        "cst_id", "trans_date", "trans_datetime", "amount",
        "trans_id", "target_id", "label"
    ]
    
    # Remove header rows
    df = df[df["cst_id"] != "cst_dim_id"].copy()
    
    # Parse dates
    df["trans_date"] = _parse_date(df["trans_date"]).dt.date
    df["trans_datetime"] = _parse_date(df["trans_datetime"])
    
    # Parse numeric columns
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    
    # Drop invalid rows
    df = df.dropna(subset=["cst_id", "trans_date", "trans_datetime", "amount", "label"])
    df["label"] = df["label"].astype(int)
    
    return df


def load_behavior(path: str) -> pd.DataFrame:
    """Load and clean customer behavior data"""
    df = pd.read_csv(path, **CSV_PARAMS)
    df.columns = [
        "trans_date", "cst_id", "os_ver_count_30d", "phone_model_count_30d",
        "last_phone_model", "last_os", "sessions_unique_7d", "sessions_unique_30d",
        "daily_logins_avg_7d", "daily_logins_avg_30d", "login_freq_change_7_vs_30",
        "login_share_7_of_30", "avg_interval_30d", "std_interval_30d",
        "var_interval_30d", "ewm_interval_7d", "burstiness", "fano_factor",
        "zscore_interval_7_vs_30"
    ]
    
    # Remove header rows
    df = df[df["cst_id"] != "cst_dim_id"].copy()
    
    # Parse dates
    df["trans_date"] = _parse_date(df["trans_date"]).dt.date
    
    # Parse numeric columns
    numeric_cols = [
        "os_ver_count_30d", "phone_model_count_30d", "sessions_unique_7d",
        "sessions_unique_30d", "daily_logins_avg_7d", "daily_logins_avg_30d",
        "login_freq_change_7_vs_30", "login_share_7_of_30", "avg_interval_30d",
        "std_interval_30d", "var_interval_30d", "ewm_interval_7d",
        "burstiness", "fano_factor", "zscore_interval_7_vs_30"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["cst_id", "trans_date"])
    
    return df


def join_data(transactions: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    """Join transactions and behavior data"""
    data = pd.merge(
        transactions, behavior,
        on=["cst_id", "trans_date"],
        how="inner"
    )
    
    # Fill missing categorical values
    data["last_phone_model"] = data["last_phone_model"].fillna("Unknown")
    data["last_os"] = data["last_os"].fillna("Unknown")
    
    # Sort by customer and time
    data = data.sort_values(["cst_id", "trans_datetime"]).reset_index(drop=True)
    
    return data


def filter_by_window(data: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Filter data by training window from config"""
    if cfg.get("train_end_date"):
        end_date = datetime.fromisoformat(cfg["train_end_date"]).date()
        
        if cfg.get("train_window_days"):
            start_date = end_date - timedelta(days=cfg["train_window_days"])
            mask = (data["trans_date"] >= start_date) & (data["trans_date"] <= end_date)
            return data.loc[mask].copy()
        else:
            # Just filter by end date
            mask = data["trans_date"] <= end_date
            return data.loc[mask].copy()
    
    return data

