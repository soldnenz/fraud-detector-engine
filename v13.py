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
# ТЕСТОВЫЕ СЦЕНАРИИ ТРАНЗАКЦИЙ
# ============================================================================

TEST_SCENARIOS = {
    "scenario_1_typical_user": {
        "name": "Типичный пользователь (медианные значения)",
        "config": {
            "cst_id": "typical_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 18:00:00",  # Вечер, пик активности
            "amount": 10000,  # Медиана из датасета
            "target_id": "merchant_001",
            
            # Устройство
            "last_phone_model": "Xiaomi Redmi Note 11",  # Популярная модель
            "last_os": "Android 13",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            # Активность (медианные значения)
            "sessions_unique_7d": 8,
            "sessions_unique_30d": 32,
            "daily_logins_avg_7d": 2.5,
            "daily_logins_avg_30d": 2.4,
            "login_freq_change_7_vs_30": 1.04,
            "login_share_7_of_30": 0.25,
            
            # Интервалы (медианные из датасета)
            "avg_interval_30d": 74538,  # Медиана
            "std_interval_30d": 106845,  # Медиана
            "var_interval_30d": 10390817585,  # Медиана
            "ewm_interval_7d": 70000,
            "burstiness": 0.184,  # Медиана
            "fano_factor": 139.0,
            "zscore_interval_7_vs_30": 0.1,
            
            # История клиента
            "device_tenure_days": 180,  # Полгода использования
            "cst_amount_mean_past": 10500,
            "cst_txn_count_past": 30,
            "amount_rolling_mean_7d": 10200,
            "amount_rolling_std_7d": 2500,
            "txn_last_1h": 0,
            "txn_last_24h": 1,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.005,  # Медиана
            "cst_weekend_tx_share": 0.24,  # Медиана
            "hours_since_prev_trans": 75000,
        }
    },
    
    "scenario_2_new_customer": {
        "name": "Новый клиент (первая транзакция)",
        "config": {
            "cst_id": "new_customer_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 12:00:00",
            "amount": 15000,
            "target_id": "merchant_002",
            
            "last_phone_model": "Samsung Galaxy A54",
            "last_os": "Android 14",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 5,
            "sessions_unique_30d": 15,
            "daily_logins_avg_7d": 2.0,
            "daily_logins_avg_30d": 1.5,
            "login_freq_change_7_vs_30": 1.33,
            "login_share_7_of_30": 0.33,
            
            "avg_interval_30d": 100000,
            "std_interval_30d": 80000,
            "var_interval_30d": 6400000000,
            "ewm_interval_7d": 95000,
            "burstiness": 0.20,
            "fano_factor": 64.0,
            "zscore_interval_7_vs_30": 0.2,
            
            # Новый клиент
            "device_tenure_days": 5,  # Только 5 дней
            "cst_amount_mean_past": 15000,  # Глобальное среднее
            "cst_txn_count_past": 0,  # ПЕРВАЯ транзакция!
            "amount_rolling_mean_7d": 15000,
            "amount_rolling_std_7d": 0,
            "txn_last_1h": 0,
            "txn_last_24h": 0,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.0,  # Еще нет истории
            "cst_weekend_tx_share": 0.0,
            "hours_since_prev_trans": 999999,  # Нет предыдущей транзакции
        }
    },
    
    "scenario_3_night_transaction": {
        "name": "Ночная транзакция (3:00)",
        "config": {
            "cst_id": "night_user_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 03:00:00",  # НОЧЬ!
            "amount": 8000,
            "target_id": "merchant_003",
            
            "last_phone_model": "iPhone14,5",
            "last_os": "iOS 17",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 10,
            "sessions_unique_30d": 40,
            "daily_logins_avg_7d": 3.0,
            "daily_logins_avg_30d": 2.8,
            "login_freq_change_7_vs_30": 1.07,
            "login_share_7_of_30": 0.25,
            
            "avg_interval_30d": 70000,
            "std_interval_30d": 100000,
            "var_interval_30d": 10000000000,
            "ewm_interval_7d": 68000,
            "burstiness": 0.18,
            "fano_factor": 142.8,
            "zscore_interval_7_vs_30": 0.1,
            
            "device_tenure_days": 200,
            "cst_amount_mean_past": 8500,
            "cst_txn_count_past": 50,
            "amount_rolling_mean_7d": 8200,
            "amount_rolling_std_7d": 1500,
            "txn_last_1h": 0,
            "txn_last_24h": 1,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.005,  # Обычно не делает ночные транзакции!
            "cst_weekend_tx_share": 0.22,
            "hours_since_prev_trans": 72000,
        }
    },
    
    "scenario_4_high_amount": {
        "name": "Большая сумма (100k)",
        "config": {
            "cst_id": "high_spender_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 15:00:00",
            "amount": 100000,  # БОЛЬШАЯ СУММА!
            "target_id": "merchant_004",
            
            "last_phone_model": "Samsung Galaxy S23",
            "last_os": "Android 14",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 15,
            "sessions_unique_30d": 55,
            "daily_logins_avg_7d": 4.0,
            "daily_logins_avg_30d": 3.8,
            "login_freq_change_7_vs_30": 1.05,
            "login_share_7_of_30": 0.27,
            
            "avg_interval_30d": 60000,
            "std_interval_30d": 95000,
            "var_interval_30d": 9025000000,
            "ewm_interval_7d": 58000,
            "burstiness": 0.17,
            "fano_factor": 150.4,
            "zscore_interval_7_vs_30": 0.05,
            
            "device_tenure_days": 300,
            "cst_amount_mean_past": 95000,  # Привык к большим суммам
            "cst_txn_count_past": 80,
            "amount_rolling_mean_7d": 98000,
            "amount_rolling_std_7d": 15000,
            "txn_last_1h": 0,
            "txn_last_24h": 3,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.002,
            "cst_weekend_tx_share": 0.20,
            "hours_since_prev_trans": 65000,
        }
    },
    
    "scenario_5_burst_activity": {
        "name": "Burst активность (много транзакций)",
        "config": {
            "cst_id": "burst_user_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 14:00:00",
            "amount": 12000,
            "target_id": "merchant_005",
            
            "last_phone_model": "Xiaomi Poco X5",
            "last_os": "Android 12",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 20,  # Высокая активность!
            "sessions_unique_30d": 60,
            "daily_logins_avg_7d": 5.5,
            "daily_logins_avg_30d": 4.0,
            "login_freq_change_7_vs_30": 1.38,  # Скачок активности
            "login_share_7_of_30": 0.33,
            
            "avg_interval_30d": 30000,  # Частые транзакции
            "std_interval_30d": 45000,
            "var_interval_30d": 2025000000,
            "ewm_interval_7d": 25000,
            "burstiness": 0.30,  # ВЫСОКАЯ burst-ность!
            "fano_factor": 67.5,
            "zscore_interval_7_vs_30": 0.5,
            
            "device_tenure_days": 90,
            "cst_amount_mean_past": 11000,
            "cst_txn_count_past": 120,  # Много транзакций
            "amount_rolling_mean_7d": 11500,
            "amount_rolling_std_7d": 3000,
            "txn_last_1h": 2,  # МНОГО транзакций за час!
            "txn_last_24h": 15,  # И за день!
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.008,
            "cst_weekend_tx_share": 0.25,
            "hours_since_prev_trans": 1800,  # 30 минут назад!
        }
    },
    
    "scenario_6_device_change": {
        "name": "Смена устройства",
        "config": {
            "cst_id": "device_changer_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 16:00:00",
            "amount": 20000,
            "target_id": "merchant_006",
            
            "last_phone_model": "iPhone 15 Pro",  # Новое дорогое устройство
            "last_os": "iOS 17",
            "os_ver_count_30d": 2,  # Недавно обновился
            "phone_model_count_30d": 2,  # Сменил модель!
            
            "sessions_unique_7d": 8,
            "sessions_unique_30d": 35,
            "daily_logins_avg_7d": 2.8,
            "daily_logins_avg_30d": 2.5,
            "login_freq_change_7_vs_30": 1.12,
            "login_share_7_of_30": 0.23,
            
            "avg_interval_30d": 80000,
            "std_interval_30d": 110000,
            "var_interval_30d": 12100000000,
            "ewm_interval_7d": 75000,
            "burstiness": 0.19,
            "fano_factor": 151.2,
            "zscore_interval_7_vs_30": 0.12,
            
            "device_tenure_days": 3,  # НОВОЕ УСТРОЙСТВО!
            "cst_amount_mean_past": 18000,
            "cst_txn_count_past": 40,
            "amount_rolling_mean_7d": 19000,
            "amount_rolling_std_7d": 4000,
            "txn_last_1h": 0,
            "txn_last_24h": 1,
            "is_new_phone_model_for_client": 1,  # СМЕНИЛ УСТРОЙСТВО!
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.004,
            "cst_weekend_tx_share": 0.23,
            "hours_since_prev_trans": 50000,
        }
    },
    
    "scenario_7_high_amount_regular": {
        "name": "Большая сумма для привычного клиента",
        "config": {
            "cst_id": "vip_user_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 18:00:00",
            "amount": 95000,  # В пределах обычного для клиента
            "target_id": "merchant_007",
            
            "last_phone_model": "iPhone 15 Pro Max",
            "last_os": "iOS 17",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 18,
            "sessions_unique_30d": 65,
            "daily_logins_avg_7d": 4.5,
            "daily_logins_avg_30d": 4.2,
            "login_freq_change_7_vs_30": 1.07,
            "login_share_7_of_30": 0.28,
            
            "avg_interval_30d": 55000,
            "std_interval_30d": 90000,
            "var_interval_30d": 8100000000,
            "ewm_interval_7d": 52000,
            "burstiness": 0.16,
            "fano_factor": 147.3,
            "zscore_interval_7_vs_30": 0.08,
            
            "device_tenure_days": 365,  # Год использования
            "cst_amount_mean_past": 92000,  # Привык к таким суммам
            "cst_txn_count_past": 150,
            "amount_rolling_mean_7d": 94000,  # Почти равно текущей
            "amount_rolling_std_7d": 8000,
            "txn_last_1h": 0,
            "txn_last_24h": 2,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.003,
            "cst_weekend_tx_share": 0.21,
            "hours_since_prev_trans": 58000,
        }
    },
    
    "scenario_8_perfect_bot": {
        "name": "Идеальный бот (слишком регулярный)",
        "config": {
            "cst_id": "bot_suspect_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 14:00:00",
            "amount": 10000,  # Всегда одинаковая сумма
            "target_id": "merchant_008",
            
            "last_phone_model": "Samsung Galaxy A52",
            "last_os": "Android 12",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 7,  # Ровно раз в день
            "sessions_unique_30d": 30,  # Ровно раз в день
            "daily_logins_avg_7d": 1.0,  # Точно раз в день
            "daily_logins_avg_30d": 1.0,
            "login_freq_change_7_vs_30": 1.0,  # Идеально
            "login_share_7_of_30": 0.233,
            
            "avg_interval_30d": 86400,  # Ровно 24 часа
            "std_interval_30d": 100,  # Почти нет вариации!
            "var_interval_30d": 10000,  # Очень низкая вариативность
            "ewm_interval_7d": 86400,
            "burstiness": 0.001,  # ОЧЕНЬ низкая burst-ность
            "fano_factor": 0.12,  # Почти 0!
            "zscore_interval_7_vs_30": 0.0,
            
            "device_tenure_days": 90,
            "cst_amount_mean_past": 10000,  # Всегда одинаково
            "cst_txn_count_past": 30,
            "amount_rolling_mean_7d": 10000,  # Идеально
            "amount_rolling_std_7d": 0,  # НЕТ вариации!
            "txn_last_1h": 0,
            "txn_last_24h": 0,  # Всегда ровно через 24 часа
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.0,
            "cst_weekend_tx_share": 0.0,  # Никогда в выходные
            "hours_since_prev_trans": 86400,  # Ровно 24 часа
        }
    },
    
    "scenario_9_weekend_shopper": {
        "name": "Выходной шопоголик",
        "config": {
            "cst_id": "weekend_shopper_001",
            "trans_date": "2025-02-15",  # Суббота
            "trans_datetime": "2025-02-15 11:00:00",
            "amount": 25000,
            "target_id": "merchant_009",
            
            "last_phone_model": "iPhone 13",
            "last_os": "iOS 16",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 4,  # Только в выходные
            "sessions_unique_30d": 16,
            "daily_logins_avg_7d": 1.5,
            "daily_logins_avg_30d": 1.2,
            "login_freq_change_7_vs_30": 1.25,
            "login_share_7_of_30": 0.25,
            
            "avg_interval_30d": 168000,  # ~7 дней (раз в неделю)
            "std_interval_30d": 86400,  # Вариация +- день
            "var_interval_30d": 7464960000,
            "ewm_interval_7d": 165000,
            "burstiness": 0.12,
            "fano_factor": 44.4,
            "zscore_interval_7_vs_30": 0.05,
            
            "device_tenure_days": 400,
            "cst_amount_mean_past": 23000,
            "cst_txn_count_past": 40,
            "amount_rolling_mean_7d": 24000,
            "amount_rolling_std_7d": 3500,
            "txn_last_1h": 0,
            "txn_last_24h": 0,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.0,
            "cst_weekend_tx_share": 0.95,  # ТОЛЬКО в выходные!
            "hours_since_prev_trans": 168000,  # Неделя назад
        }
    },
    
    # ========== ФРОДОВЫЕ СЦЕНАРИИ ==========
    
    "fraud_1_new_device_high_amount": {
        "name": "🚨 ФРОД: Новое устройство + Огромная сумма",
        "config": {
            "cst_id": "fraud_new_device_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 12:30:00",  # День (самый высокий fraud rate!)
            "amount": 450000,  # ОГРОМНАЯ СУММА!
            "target_id": "merchant_fraud_001",
            
            "last_phone_model": "Samsung Galaxy S23 Ultra",  # НОВОЕ ДОРОГОЕ устройство
            "last_os": "Android 14",
            "os_ver_count_30d": 2,  # Недавно сменил
            "phone_model_count_30d": 2,  # СМЕНИЛ УСТРОЙСТВО!
            
            "sessions_unique_7d": 2,  # Очень мало активности
            "sessions_unique_30d": 8,
            "daily_logins_avg_7d": 0.5,  # Почти не заходит
            "daily_logins_avg_30d": 0.6,
            "login_freq_change_7_vs_30": 0.83,
            "login_share_7_of_30": 0.25,
            
            "avg_interval_30d": 200000,  # Редко совершает транзакции
            "std_interval_30d": 150000,
            "var_interval_30d": 22500000000,
            "ewm_interval_7d": 180000,
            "burstiness": 0.25,
            "fano_factor": 112.5,
            "zscore_interval_7_vs_30": 0.3,
            
            "device_tenure_days": 1,  # ТОЛЬКО 1 ДЕНЬ!
            "cst_amount_mean_past": 12000,  # Обычно тратит мало
            "cst_txn_count_past": 25,
            "amount_rolling_mean_7d": 11500,
            "amount_rolling_std_7d": 2000,
            "txn_last_1h": 0,
            "txn_last_24h": 0,
            "is_new_phone_model_for_client": 1,  # НОВОЕ УСТРОЙСТВО!
            "is_new_os_for_client": 1,
            "cst_night_tx_share": 0.02,  # Обычно не делает ночные
            "cst_weekend_tx_share": 0.20,
            "hours_since_prev_trans": 250000,  # Давно не было транзакций
        }
    },
    
    "fraud_2_account_takeover": {
        "name": "🚨 ФРОД: Взлом аккаунта",
        "config": {
            "cst_id": "fraud_takeover_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 14:15:00",  # День
            "amount": 650000,  # Median фрода ~₸100k, мы берем 95% перцентиль
            "target_id": "merchant_fraud_002",
            
            "last_phone_model": "Xiaomi Redmi Note 12",  # Дешевое устройство
            "last_os": "Android 12",
            "os_ver_count_30d": 3,  # Часто меняет!
            "phone_model_count_30d": 3,  # МНОГО СМЕН!
            
            "sessions_unique_7d": 15,  # Резкий скачок активности!
            "sessions_unique_30d": 25,
            "daily_logins_avg_7d": 8.0,  # ОЧЕНЬ ВЫСОКАЯ активность!
            "daily_logins_avg_30d": 2.0,
            "login_freq_change_7_vs_30": 4.0,  # ОГРОМНЫЙ СКАЧОК!
            "login_share_7_of_30": 0.60,
            
            "avg_interval_30d": 50000,
            "std_interval_30d": 180000,  # ОГРОМНАЯ вариация
            "var_interval_30d": 32400000000,
            "ewm_interval_7d": 15000,  # Очень частые транзакции последнее время
            "burstiness": 0.56,  # ВЫСОКАЯ burst-ность!
            "fano_factor": 648.0,
            "zscore_interval_7_vs_30": 1.2,
            
            "device_tenure_days": 2,  # Новое устройство
            "cst_amount_mean_past": 8000,  # Обычно маленькие суммы
            "cst_txn_count_past": 50,
            "amount_rolling_mean_7d": 9000,
            "amount_rolling_std_7d": 1500,
            "txn_last_1h": 3,  # МНОГО транзакций за час!
            "txn_last_24h": 12,  # И за день!
            "is_new_phone_model_for_client": 1,
            "is_new_os_for_client": 1,
            "cst_night_tx_share": 0.01,  # Обычно не ночью
            "cst_weekend_tx_share": 0.18,
            "hours_since_prev_trans": 1800,  # 30 минут назад!
        }
    },
    
    "fraud_3_first_transaction_fraud": {
        "name": "🚨 ФРОД: Первая транзакция = Фрод",
        "config": {
            "cst_id": "fraud_first_tx_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 13:00:00",  # День
            "amount": 350000,  # СРАЗУ ОГРОМНАЯ СУММА!
            "target_id": "merchant_fraud_003",
            
            "last_phone_model": "OtherPhoneModel",  # Редкое устройство
            "last_os": "Android 11",  # Старая ОС
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 1,  # ПЕРВАЯ сессия
            "sessions_unique_30d": 1,
            "daily_logins_avg_7d": 0.14,  # Почти нет логинов
            "daily_logins_avg_30d": 0.03,
            "login_freq_change_7_vs_30": 4.67,
            "login_share_7_of_30": 1.0,
            
            "avg_interval_30d": 999999,  # Нет истории
            "std_interval_30d": 0,
            "var_interval_30d": 0,
            "ewm_interval_7d": 999999,
            "burstiness": 0.0,
            "fano_factor": 0.0,
            "zscore_interval_7_vs_30": 0.0,
            
            "device_tenure_days": 0,  # НОЛЬ ДНЕЙ!
            "cst_amount_mean_past": 75000,  # Нет истории - глобальное среднее
            "cst_txn_count_past": 0,  # ПЕРВАЯ ТРАНЗАКЦИЯ!
            "amount_rolling_mean_7d": 75000,
            "amount_rolling_std_7d": 0,
            "txn_last_1h": 0,
            "txn_last_24h": 0,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.0,
            "cst_weekend_tx_share": 0.0,
            "hours_since_prev_trans": 999999,
        }
    },
    
    "fraud_4_rapid_fire": {
        "name": "🚨 ФРОД: Rapid Fire (много за раз)",
        "config": {
            "cst_id": "fraud_rapid_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 11:45:00",
            "amount": 280000,  # Средняя фродовая сумма x 1.3
            "target_id": "merchant_fraud_004",
            
            "last_phone_model": "Samsung Galaxy A52",
            "last_os": "Android 12",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 25,  # ОГРОМНОЕ количество!
            "sessions_unique_30d": 80,
            "daily_logins_avg_7d": 12.0,  # Очень высокая активность
            "daily_logins_avg_30d": 5.0,
            "login_freq_change_7_vs_30": 2.4,
            "login_share_7_of_30": 0.31,
            
            "avg_interval_30d": 15000,  # ~4 часа - очень часто
            "std_interval_30d": 25000,
            "var_interval_30d": 625000000,
            "ewm_interval_7d": 8000,  # ~2 часа
            "burstiness": 0.40,  # Высокая burst-ность
            "fano_factor": 41.7,
            "zscore_interval_7_vs_30": 1.5,
            
            "device_tenure_days": 45,
            "cst_amount_mean_past": 8000,
            "cst_txn_count_past": 200,  # Много транзакций
            "amount_rolling_mean_7d": 9000,
            "amount_rolling_std_7d": 2000,
            "txn_last_1h": 5,  # 5 ТРАНЗАКЦИЙ ЗА ЧАС!
            "txn_last_24h": 25,  # 25 ЗА ДЕНЬ!
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.15,  # Много ночных транзакций
            "cst_weekend_tx_share": 0.30,
            "hours_since_prev_trans": 900,  # 15 МИНУТ назад!
        }
    },
    
    "fraud_5_amount_spike": {
        "name": "🚨 ФРОД: Резкий скачок суммы",
        "config": {
            "cst_id": "fraud_spike_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 15:30:00",  # День
            "amount": 550000,  # ОГРОМНЫЙ СКАЧОК!
            "target_id": "merchant_fraud_005",
            
            "last_phone_model": "iPhone 11",
            "last_os": "iOS 15",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 10,
            "sessions_unique_30d": 40,
            "daily_logins_avg_7d": 3.0,
            "daily_logins_avg_30d": 2.8,
            "login_freq_change_7_vs_30": 1.07,
            "login_share_7_of_30": 0.25,
            
            "avg_interval_30d": 70000,
            "std_interval_30d": 100000,
            "var_interval_30d": 10000000000,
            "ewm_interval_7d": 68000,
            "burstiness": 0.18,
            "fano_factor": 142.8,
            "zscore_interval_7_vs_30": 0.1,
            
            "device_tenure_days": 150,
            "cst_amount_mean_past": 5500,  # Обычно ОЧЕНЬ маленькие суммы
            "cst_txn_count_past": 80,
            "amount_rolling_mean_7d": 5800,
            "amount_rolling_std_7d": 800,
            "txn_last_1h": 0,
            "txn_last_24h": 2,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.003,
            "cst_weekend_tx_share": 0.20,
            "hours_since_prev_trans": 70000,
        }
    },
    
    "fraud_6_old_device_high_amount": {
        "name": "🚨 ФРОД: Старое устройство + Огромная сумма",
        "config": {
            "cst_id": "fraud_old_device_001",
            "trans_date": "2025-02-10",
            "trans_datetime": "2025-02-10 16:30:00",  # День
            "amount": 720000,  # 95% перцентиль фродов
            "target_id": "merchant_fraud_006",
            
            "last_phone_model": "Samsung Galaxy A10",  # Старое устройство
            "last_os": "Android 9",  # ОЧЕНЬ старая ОС!
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 5,
            "sessions_unique_30d": 15,
            "daily_logins_avg_7d": 1.5,
            "daily_logins_avg_30d": 1.2,
            "login_freq_change_7_vs_30": 1.25,
            "login_share_7_of_30": 0.33,
            
            "avg_interval_30d": 150000,
            "std_interval_30d": 120000,
            "var_interval_30d": 14400000000,
            "ewm_interval_7d": 140000,
            "burstiness": 0.22,
            "fano_factor": 96.0,
            "zscore_interval_7_vs_30": 0.2,
            
            "device_tenure_days": 500,  # Очень старое устройство!
            "cst_amount_mean_past": 7000,
            "cst_txn_count_past": 35,
            "amount_rolling_mean_7d": 7200,
            "amount_rolling_std_7d": 1200,
            "txn_last_1h": 0,
            "txn_last_24h": 1,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.02,
            "cst_weekend_tx_share": 0.25,
            "hours_since_prev_trans": 160000,
        }
    },
    
    # ========== VIP КЛИЕНТЫ (РЕАЛЬНЫЕ ИЗ ДАТАСЕТА) ==========
    
    "vip_1_ultra_high_spender": {
        "name": "💎 VIP: Ультра-богатый клиент (средняя ₸266k)",
        "config": {
            "cst_id": "453670216",  # Реальный VIP из датасета
            "trans_date": "2025-08-01",
            "trans_datetime": "2025-08-01 14:30:00",
            "amount": 290000,  # Близко к средней (+9%)
            "target_id": "merchant_vip_001",
            
            "last_phone_model": "iPhone 15 Pro Max",  # Дорогое устройство
            "last_os": "iOS 17",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 14,
            "sessions_unique_30d": 50,
            "daily_logins_avg_7d": 4.0,
            "daily_logins_avg_30d": 3.8,
            "login_freq_change_7_vs_30": 1.05,
            "login_share_7_of_30": 0.28,
            
            "avg_interval_30d": 220000,  # ~61 час между транзакциями (раз в 2-3 дня)
            "std_interval_30d": 90000,   # ~25 часов
            "var_interval_30d": 8100000000,
            "ewm_interval_7d": 200000,
            "burstiness": 0.14,
            "fano_factor": 36.8,
            "zscore_interval_7_vs_30": 0.08,
            
            "device_tenure_days": 365,  # Год использования
            "cst_amount_mean_past": 266000,  # СРЕДНЯЯ СУММА ₸266k!
            "cst_txn_count_past": 8,
            "amount_rolling_mean_7d": 270000,  # Близко к текущей
            "amount_rolling_std_7d": 120000,   # Большая вариация нормальна для VIP
            "txn_last_1h": 0,
            "txn_last_24h": 0,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.001,  # Почти никогда ночью
            "cst_weekend_tx_share": 0.15,
            "hours_since_prev_trans": 250000,  # ~7 дней (нормально для VIP)
        }
    },
    
    "vip_2_consistent_big_spender": {
        "name": "💎 VIP: Стабильный богатый клиент (средняя ₸180k)",
        "config": {
            "cst_id": "vip_consistent_001",
            "trans_date": "2025-08-01",
            "trans_datetime": "2025-08-01 11:00:00",
            "amount": 185000,  # Близко к его средней
            "target_id": "merchant_vip_002",
            
            "last_phone_model": "Samsung Galaxy S23 Ultra",
            "last_os": "Android 14",
            "os_ver_count_30d": 1,
            "phone_model_count_30d": 1,
            
            "sessions_unique_7d": 16,
            "sessions_unique_30d": 60,
            "daily_logins_avg_7d": 4.2,
            "daily_logins_avg_30d": 4.0,
            "login_freq_change_7_vs_30": 1.05,
            "login_share_7_of_30": 0.27,
            
            "avg_interval_30d": 150000,  # ~42 часа
            "std_interval_30d": 70000,   # ~19 часов
            "var_interval_30d": 4900000000,
            "ewm_interval_7d": 145000,
            "burstiness": 0.13,
            "fano_factor": 32.7,
            "zscore_interval_7_vs_30": 0.05,
            
            "device_tenure_days": 180,
            "cst_amount_mean_past": 180000,  # СРЕДНЯЯ СУММА ₸180k!
            "cst_txn_count_past": 25,
            "amount_rolling_mean_7d": 182000,  # Очень близко к текущей
            "amount_rolling_std_7d": 15000,    # Маленькая вариация (стабильный)
            "txn_last_1h": 0,
            "txn_last_24h": 1,
            "is_new_phone_model_for_client": 0,
            "is_new_os_for_client": 0,
            "cst_night_tx_share": 0.002,
            "cst_weekend_tx_share": 0.18,
            "hours_since_prev_trans": 155000,  # ~43 часа
        }
    },
}

# Используем первый сценарий по умолчанию
TRANSACTION_CONFIG = TEST_SCENARIOS["scenario_1_typical_user"]["config"]


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
    data["device_tenure_days_raw"] = (
        (data["trans_datetime"] - data["device_first_seen"]).dt.total_seconds() / 86400.0
    )
    # ИСПРАВЛЕНИЕ: Cap на 180 днях (после этого начинается fraud zone)
    # Проблема: 180-365 дней = fraud rate 2.03%
    # Решение: Все что больше 180 дней = 180 (убираем негативное влияние долгого tenure)
    data["device_tenure_days"] = data["device_tenure_days_raw"].clip(lower=0, upper=180)
    
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
    
    # === VIP / HIGH SPENDER DETECTION ===
    # Если клиент привык к большим суммам, то большая транзакция НЕ подозрительна
    amount_90p = data["amount"].quantile(0.90)
    amount_95p = data["amount"].quantile(0.95)
    
    data["is_high_spender"] = (data["cst_amount_mean_past"] > amount_90p).astype(int)
    data["is_ultra_high_spender"] = (data["cst_amount_mean_past"] > amount_95p).astype(int)
    
    # ПРОЦЕНТНОЕ отклонение от средней (ключевая фича!)
    data["amount_deviation_pct"] = (np.abs(data["amount"] - data["cst_amount_mean_past"]) / 
                                    (data["cst_amount_mean_past"] + 1e-3) * 100)
    data["amount_deviation_pct"] = data["amount_deviation_pct"].clip(0, 1000)
    
    # Транзакция соответствует истории (в пределах +/-50%)
    data["amount_matches_history"] = (data["amount_deviation_pct"] < 50).astype(int)
    
    # Очень близко к средней (в пределах +/-20%)
    data["amount_very_close_to_avg"] = (data["amount_deviation_pct"] < 20).astype(int)
    
    # Текущая сумма МЕНЬШЕ средней (безопасно)
    data["amount_below_average"] = (data["amount"] < data["cst_amount_mean_past"]).astype(int)
    
    # Ratio текущей суммы к средней (для VIP должен быть близок к 1.0)
    data["amount_vs_history_ratio"] = data["amount"] / (data["cst_amount_mean_past"] + 1e-3)
    data["amount_vs_history_ratio"] = data["amount_vs_history_ratio"].clip(0.01, 100)
    
    # КРИТИЧЕСКАЯ ФИЧА: для VIP клиентов большие суммы НОРМАЛЬНЫ если в пределах истории
    data["vip_safe_transaction"] = (
        (data["is_high_spender"] == 1) & 
        (data["amount_deviation_pct"] < 100)  # В пределах 2x от средней
    ).astype(int)
    
    # === INTERACTIONS ===
    data["night_x_high_amount"] = data["is_night_tx"] * data["is_high_amount_vs_client"]
    data["new_device_x_high_amount"] = data["is_new_phone_model_for_client"] * data["is_high_amount_vs_client"]
    data["weekend_x_high_amount"] = data["is_weekend"] * data["is_high_amount_vs_client"]
    
    # VIP interactions
    data["high_spender_x_high_amount"] = data["is_high_spender"] * (data["amount"] > data["amount"].quantile(0.90)).astype(int)
    
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
    emb_artifacts["global_amount_90p"] = float(data["amount"].quantile(0.90))
    emb_artifacts["global_amount_95p"] = float(data["amount"].quantile(0.95))
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
    
    # 2.5 Device tenure (cap на 180 днях)
    if "device_tenure_days" in tx_config:
        tenure_days_raw = tx_config["device_tenure_days"]
        # Cap на 180 (после этого fraud zone)
        tx_row["device_tenure_days"] = min(tenure_days_raw, 180)
    
    # 2.6 Amount-derived features (если базовые фичи уже в конфиге)
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
    
    # 2.7 Rolling amount features (если есть в конфиге)
    if "amount_rolling_mean_7d" in tx_config and "amount_rolling_std_7d" in tx_config:
        amt_mean_7d = tx_config["amount_rolling_mean_7d"]
        amt_std_7d = tx_config["amount_rolling_std_7d"]
        
        tx_row["amount_deviation_rolling"] = abs(tx_row["amount"].iloc[0] - amt_mean_7d)
        tx_row["amount_deviation_rolling_ratio"] = tx_row["amount_deviation_rolling"] / (amt_mean_7d + 1e-3)
        tx_row["zscore_amount"] = (tx_row["amount"].iloc[0] - amt_mean_7d) / (amt_std_7d + 1e-3)
        tx_row["zscore_amount"] = tx_row["zscore_amount"].clip(-10, 10)
        tx_row["amount_over_rolling_mean_7d"] = tx_row["amount"] / (amt_mean_7d + 1e-3)
    
    # 2.8 Ratio features
    if "sessions_unique_7d" in tx_config and "sessions_unique_30d" in tx_config:
        tx_row["sessions_7d_vs_30d_ratio"] = tx_config["sessions_unique_7d"] / (tx_config["sessions_unique_30d"] + 1)
    if "daily_logins_avg_7d" in tx_config and "daily_logins_avg_30d" in tx_config:
        tx_row["logins_7d_vs_30d_ratio"] = tx_config["daily_logins_avg_7d"] / (tx_config["daily_logins_avg_30d"] + 1)
    
    # 2.9 VIP / High spender features (КРИТИЧЕСКИ ВАЖНО ДЛЯ VIP!)
    if "cst_amount_mean_past" in tx_row.columns:
        global_amount_90p = emb_artifacts.get("global_amount_90p", 50000.0) if emb_artifacts else 50000.0
        global_amount_95p = emb_artifacts.get("global_amount_95p", 200000.0) if emb_artifacts else 200000.0
        
        amt_mean = tx_row["cst_amount_mean_past"].iloc[0]
        amt_current = tx_row["amount"].iloc[0]
        
        tx_row["is_high_spender"] = int(amt_mean > global_amount_90p)
        tx_row["is_ultra_high_spender"] = int(amt_mean > global_amount_95p)
        
        # ПРОЦЕНТНОЕ отклонение (ключевая фича!)
        amt_deviation_pct = np.abs(amt_current - amt_mean) / (amt_mean + 1e-3) * 100
        tx_row["amount_deviation_pct"] = np.clip(amt_deviation_pct, 0, 1000)
        
        # Флаги соответствия истории
        tx_row["amount_matches_history"] = int(amt_deviation_pct < 50)
        tx_row["amount_very_close_to_avg"] = int(amt_deviation_pct < 20)
        tx_row["amount_below_average"] = int(amt_current < amt_mean)
        
        # Ratio
        tx_row["amount_vs_history_ratio"] = np.clip(amt_current / (amt_mean + 1e-3), 0.01, 100)
        
        # КРИТИЧЕСКАЯ ФИЧА: VIP safe transaction
        tx_row["vip_safe_transaction"] = int(
            (tx_row["is_high_spender"].iloc[0] == 1) and 
            (amt_deviation_pct < 100)
        )
        
        # VIP interaction
        is_large_amt = int(amt_current > global_amount_90p)
        tx_row["high_spender_x_high_amount"] = int(tx_row["is_high_spender"].iloc[0]) * is_large_amt
    
    # 2.10 Interaction features
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
    
    # 2.11 Additional transformations
    if "std_interval_30d" in tx_config:
        tx_row["std_interval_30d_log"] = np.log1p(max(tx_config["std_interval_30d"], 0))
    
    # 2.12 Activity profile score
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
    
    # === АНАЛИЗ ПОРОГОВ ===
    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)
    print(f"\nMeta-model F1 threshold: {results['meta_f1_thr']:.4f}")
    print(f"Meta-model Cost threshold: {results['meta_cost_thr']:.4f}")
    
    # Посмотрим на распределение вероятностей для легитимных и фрод
    meta_oof = results["meta_oof_probas"]
    legit_probs = meta_oof[y_full == 0]
    fraud_probs = meta_oof[y_full == 1]
    
    print(f"\nLegitimate transactions probability distribution:")
    print(f"  Mean: {legit_probs.mean():.4f}")
    print(f"  Median: {np.median(legit_probs):.4f}")
    print(f"  95th percentile: {np.percentile(legit_probs, 95):.4f}")
    print(f"  99th percentile: {np.percentile(legit_probs, 99):.4f}")
    
    print(f"\nFraud transactions probability distribution:")
    print(f"  Mean: {fraud_probs.mean():.4f}")
    print(f"  Median: {np.median(fraud_probs):.4f}")
    print(f"  5th percentile: {np.percentile(fraud_probs, 5):.4f}")
    
    # Используем cost-optimized порог
    threshold = results["meta_cost_thr"]
    
    print(f"\n✅ Using COST-OPTIMIZED threshold: {threshold:.4f}")
    print(f"   (This minimizes business cost: FP_cost * 0.1 + FN_cost * 1.0)")
    
    # Оценка на этом пороге
    meta_metrics = evaluate_predictions(
        y_full, 
        meta_oof, 
        threshold, 
        amounts_full
    )
    
    print(f"\nPerformance at cost-optimized threshold:")
    print(f"  Precision: {meta_metrics.precision:.4f}")
    print(f"  Recall: {meta_metrics.recall:.4f}")
    print(f"  F1: {meta_metrics.f1:.4f}")
    print(f"  ROC-AUC: {meta_metrics.roc_auc:.4f}")
    if meta_metrics.business_metrics:
        bm = meta_metrics.business_metrics
        print(f"  Fraud Prevention Rate: {bm.fraud_prevention_rate:.2%}")
        print(f"  Total Cost: ₸{bm.total_cost:,.0f}")
    
    # 3. ТЕСТ НА СЛУЧАЙНЫХ ЛЕГИТИМНЫХ ТРАНЗАКЦИЯХ ИЗ ДАТАСЕТА
    print("\n" + "=" * 80)
    print("TESTING ON RANDOM LEGITIMATE TRANSACTIONS FROM DATASET")
    print("=" * 80)
    
    # Берем 20 случайных легитимных транзакций (разные суммы)
    legit_data = data[data["label"] == 0].sample(n=20, random_state=123)
    
    print(f"\nTesting {len(legit_data)} random legitimate transactions:")
    print(f"Threshold: {threshold:.4f}")
    print()
    print(f"{'Transaction ID':<15} {'Amount':>12} {'Probability':>12} {'Status':>10}")
    print("-" * 55)
    
    blocked_count = 0
    for idx, row in legit_data.iterrows():
        # Используем OOF predictions для этой транзакции
        oof_idx = data.index.get_loc(idx)
        prob = meta_oof[oof_idx]
        is_blocked = prob >= threshold
        
        if is_blocked:
            blocked_count += 1
            
        status = "🚫 BLOCK" if is_blocked else "✅ PASS"
        amt_str = f"₸{row['amount']:,.0f}"
        prob_str = f"{prob*100:.2f}%"
        print(f"{row['trans_id']:<15} {amt_str:>12} {prob_str:>12} {status:>10}")
    
    print("-" * 55)
    print(f"\nSummary: {blocked_count}/{len(legit_data)} legitimate transactions blocked ({blocked_count/len(legit_data)*100:.1f}%)")
    print(f"Expected FPR at this threshold: ~{(1 - meta_metrics.precision):.2%}")
    
    # 4. ТЕСТИРОВАНИЕ ВСЕХ СЦЕНАРИЕВ
    print("\n" + "=" * 80)
    print("TESTING ALL SCENARIOS")
    print("=" * 80)
    
    scenario_results = []
    
    for scenario_key, scenario_data in TEST_SCENARIOS.items():
        scenario_name = scenario_data["name"]
        scenario_config = scenario_data["config"]
        
        # Предсказание для сценария
        result = predict_single_transaction(
            scenario_config,
            models,
            results["meta_model"],
            data,
            emb_artifacts,
            categorical_cols,
            list(X_full.columns),
            feature_baselines,
            threshold,
            client_history=None
        )
        
        scenario_results.append({
            "name": scenario_name,
            "config": scenario_config,
            "result": result
        })
    
    # Выводим результаты всех сценариев в таблице
    print(f"\nThreshold: {threshold:.4f}")
    print(f"Legitimate median: {np.median(legit_probs)*100:.2f}%")
    print(f"Legitimate 95th percentile: {np.percentile(legit_probs, 95)*100:.2f}%")
    print()
    print("-" * 120)
    print(f"{'Scenario':<40} {'Amount':>10} {'Probability':>12} {'Decision':>10} {'vs Median':>12} {'Top Risk Factor':<30}")
    print("-" * 120)
    
    for sr in scenario_results:
        name = sr["name"]
        amount = sr["config"]["amount"]
        prob = sr["result"]["final_prediction"]["fraud_probability"]
        decision = "🚫 BLOCK" if sr["result"]["final_prediction"]["is_fraud"] else "✅ ALLOW"
        vs_median = f"+{(prob - np.median(legit_probs))*100:.1f}%" if prob > np.median(legit_probs) else f"{(prob - np.median(legit_probs))*100:.1f}%"
        
        top_risk = sr["result"]["explanation"]["top_risk_factors"][0]
        risk_str = f"{top_risk['feature'][:25]} ({top_risk['importance']*100:.1f}%)"
        
        prob_str = f"{prob*100:.2f}%"
        amount_str = f"₸{amount:,}"
        
        # Цвет для вероятности
        if prob < 0.3:
            prob_color = ""  # Зеленый (низкий риск)
        elif prob < 0.6:
            prob_color = ""  # Желтый (средний)
        else:
            prob_color = ""  # Красный (высокий)
        
        print(f"{name:<40} {amount_str:>10} {prob_str:>12} {decision:>10} {vs_median:>12} {risk_str:<30}")
    
    print("-" * 120)
    
    # Детальный вывод для каждого сценария
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF EACH SCENARIO")
    print("=" * 80)
    
    for sr in scenario_results:
        print(f"\n{'='*80}")
        print(f"📊 {sr['name'].upper()}")
        print(f"{'='*80}")
        
        config = sr["config"]
        result = sr["result"]
        final = result["final_prediction"]
        
        print(f"\n📋 Transaction:")
        print(f"   Amount: ₸{config['amount']:,}")
        if "cst_amount_mean_past" in config:
            amt_mean = config['cst_amount_mean_past']
            amt_current = config['amount']
            deviation_pct = abs(amt_current - amt_mean) / (amt_mean + 1e-3) * 100
            print(f"   Customer average: ₸{amt_mean:,}")
            print(f"   Deviation: {deviation_pct:+.1f}% from average")
        print(f"   Time: {config['trans_datetime']}")
        print(f"   Device: {config['last_phone_model']}")
        print(f"   Customer age: {config['cst_txn_count_past']} past transactions")
        print(f"   Device tenure: {config['device_tenure_days']} days")
        
        print(f"\n🎯 Result:")
        prob_pct = final['fraud_percentage']
        print(f"   Probability: {prob_pct:.2f}% {'(HIGH RISK!)' if prob_pct > 50 else '(MEDIUM)' if prob_pct > 30 else '(LOW RISK)'}")
        print(f"   Decision: {' 🚫 BLOCK' if final['is_fraud'] else '✅ ALLOW'}")
        print(f"   vs Legitimate median: {(final['fraud_probability'] - np.median(legit_probs))*100:+.1f}%")
        
        preds = result["predictions"]
        print(f"\n📊 Models:")
        print(f"   LGBM: {preds['lgbm']*100:5.2f}%  |  XGB: {preds['xgb']*100:5.2f}%  |  CatBoost: {preds['cat']*100:5.2f}%")
        
        print(f"\n💡 Top 3 Risk Factors:")
        for i, factor in enumerate(result["explanation"]["top_risk_factors"][:3], 1):
            val_str = f" = {factor['value']:.2f}" if factor['value'] is not None and isinstance(factor['value'], (int, float)) else ""
            print(f"   {i}. {factor['feature']}{val_str} (importance: {factor['importance']*100:.2f}%)")
    
    # 5. ИТОГОВЫЕ ВЫВОДЫ И АНОМАЛИИ
    print("\n" + "=" * 80)
    print("SUMMARY & ANOMALIES")
    print("=" * 80)
    
    # Сортируем сценарии по вероятности
    sorted_scenarios = sorted(scenario_results, key=lambda x: x["result"]["final_prediction"]["fraud_probability"])
    
    print(f"\n📊 Ranking (lowest to highest risk):")
    for i, sr in enumerate(sorted_scenarios, 1):
        prob = sr["result"]["final_prediction"]["fraud_probability"]
        decision = "🚫 BLOCK" if sr["result"]["final_prediction"]["is_fraud"] else "✅ ALLOW"
        print(f"   {i}. {sr['name']:<40} {prob*100:5.2f}% {decision}")
    
    # Анализ аномалий
    print(f"\n🔍 Anomaly Analysis:")
    
    # Какие сценарии блокируются?
    blocked = [sr for sr in scenario_results if sr["result"]["final_prediction"]["is_fraud"]]
    allowed = [sr for sr in scenario_results if not sr["result"]["final_prediction"]["is_fraud"]]
    
    print(f"\n   Blocked scenarios: {len(blocked)}/{len(scenario_results)}")
    if blocked:
        for sr in blocked:
            prob = sr["result"]["final_prediction"]["fraud_probability"]
            print(f"      - {sr['name']}: {prob*100:.2f}%")
    
    print(f"\n   Allowed scenarios: {len(allowed)}/{len(scenario_results)}")
    if allowed:
        for sr in allowed:
            prob = sr["result"]["final_prediction"]["fraud_probability"]
            print(f"      - {sr['name']}: {prob*100:.2f}%")
    
    # Найдем аномалии
    print(f"\n⚠️  Potential Issues:")
    
    # Проверим новых клиентов
    new_customer_scenarios = [sr for sr in scenario_results if sr["config"].get("cst_txn_count_past", 100) < 5]
    if new_customer_scenarios:
        avg_prob_new = np.mean([sr["result"]["final_prediction"]["fraud_probability"] for sr in new_customer_scenarios])
        print(f"   - New customers average probability: {avg_prob_new*100:.2f}%")
        if avg_prob_new > np.median(legit_probs) * 1.5:
            print(f"     ⚠️  WARNING: New customers are flagged as higher risk!")
    
    # Проверим ночные транзакции
    night_scenarios = [sr for sr in scenario_results if "03:00:00" in sr["config"]["trans_datetime"]]
    if night_scenarios:
        avg_prob_night = np.mean([sr["result"]["final_prediction"]["fraud_probability"] for sr in night_scenarios])
        print(f"   - Night transactions average probability: {avg_prob_night*100:.2f}%")
        if avg_prob_night > np.median(legit_probs) * 2:
            print(f"     ⚠️  WARNING: Night transactions heavily penalized!")
    
    # Проверим смену устройств
    device_change_scenarios = [sr for sr in scenario_results if sr["config"].get("is_new_phone_model_for_client", 0) == 1]
    if device_change_scenarios:
        avg_prob_device = np.mean([sr["result"]["final_prediction"]["fraud_probability"] for sr in device_change_scenarios])
        print(f"   - Device change average probability: {avg_prob_device*100:.2f}%")
        if avg_prob_device > np.median(legit_probs) * 1.5:
            print(f"     ⚠️  WARNING: Device changes heavily penalized!")
    
    print(f"\n✅ RECOMMENDATIONS:")
    print(f"   - Use threshold: {threshold:.4f}")
    print(f"   - Expected precision: {meta_metrics.precision:.2%}")
    print(f"   - Expected recall: {meta_metrics.recall:.2%}")
    print(f"   - Legitimate median probability: {np.median(legit_probs)*100:.2f}%")
    print(f"   - Monitor edge cases: new customers, night transactions, device changes")


if __name__ == "__main__":
    main()
