#!/usr/bin/env python3
"""
Скрипт для отладки - смотрим какие фичи попадают в модель
"""

import requests
import json
import pandas as pd

# Отправляем опасную транзакцию с ПОЛНЫМИ данными
транзакция = {
    "cst_id": "CST_HACKER_001",
    "target_id": "TGT_FRAUD_999",
    "amount": 2000000,
    
    # История клиента
    "cst_amount_mean_past": 30000,
    "cst_txn_count_past": 3,
    "amount_over_mean_past": 66.0,
    
    # История получателя
    "target_fraud_rate_past_smooth": 0.45,
    "target_txn_count_past": 2,
    
    # Флаги
    "is_new_target_for_client": 1,
    "is_new_phone_model_for_client": 1,
    "is_new_os_for_client": 1,
    "is_first_tx_for_client": 0,
    
    # Время
    "is_night_tx": 1,
    "is_weekend": 1,
    "hours_since_prev_trans": 0.1,
    
    # Поведение
    "sessions_unique_7d": 1,
    "sessions_unique_30d": 2,
    "daily_logins_avg_7d": 0.2,
    "daily_logins_avg_30d": 0.1,
}

print("=" * 80)
print("🔍 ОТЛАДКА: ПРОВЕРЯЕМ ЧТО ПОПАДАЕТ В МОДЕЛЬ")
print("=" * 80)

print("\n📤 ОТПРАВЛЯЕМ ТРАНЗАКЦИЮ:")
print(json.dumps(транзакция, indent=2, ensure_ascii=False))

response = requests.post(
    "http://localhost:8000/score",
    json=транзакция
)

print("\n📥 ОТВЕТ ОТ СЕРВЕРА:")
result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))

print("\n" + "=" * 80)
print("🤔 АНАЛИЗ:")
print("=" * 80)
print(f"\n⚠️  Вероятность: {result['probability']:.1%}")
print(f"⚠️  Решение: {result['decision']}")
print(f"\n❓ ПОЧЕМУ ТАК МАЛО?")
print("\nВозможные причины:")
print("1. Модель не видит все переданные поля")
print("2. В TxRequest не хватает полей")
print("3. Поля заполняются дефолтными значениями")
print("\nПроверим API код...")

