#!/usr/bin/env python3
"""
Скрипт для отправки транзакций на Fraud Detection API
Примеры использования для проверки мошенничества
"""

import requests
import json
from datetime import datetime


# ============================================================================
# НАСТРОЙКИ
# ============================================================================

API_URL = "http://localhost:8000"  # Адрес API сервера


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def отправить_транзакцию(транзакция: dict):
    """
    Отправляет транзакцию на проверку в API
    
    Параметры:
        транзакция (dict): Словарь с данными транзакции
    
    Возвращает:
        dict: Ответ от сервера с решением
    """
    print("\n" + "=" * 80)
    print("📤 ОТПРАВКА ТРАНЗАКЦИИ НА ПРОВЕРКУ")
    print("=" * 80)
    
    try:
        # Отправляем POST запрос
        response = requests.post(
            f"{API_URL}/score",
            json=транзакция,
            timeout=10
        )
        
        # Проверяем статус
        if response.status_code == 200:
            result = response.json()
            показать_результат(транзакция, result)
            return result
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"   {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Не удалось подключиться к API")
        print("   Убедитесь что сервер запущен: python api/main.py")
        return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def показать_результат(транзакция: dict, результат: dict):
    """
    Красиво показывает результат проверки транзакции
    """
    print("\n📊 ИНФОРМАЦИЯ О ТРАНЗАКЦИИ:")
    print(f"   💳 Клиент:        {транзакция['cst_id']}")
    print(f"   👤 Получатель:    {транзакция['target_id']}")
    print(f"   💰 Сумма:         ₸{транзакция['amount']:,.0f}")
    
    print("\n🎯 РЕЗУЛЬТАТ ПРОВЕРКИ:")
    
    # Решение с эмодзи
    decision = результат['decision']
    if decision == "BLOCK":
        emoji = "🚫"
        decision_ru = "БЛОКИРОВАТЬ"
        color = "КРАСНЫЙ"
    elif decision == "REVIEW":
        emoji = "⚠️"
        decision_ru = "НА ПРОВЕРКУ"
        color = "ЖЁЛТЫЙ"
    else:
        emoji = "✅"
        decision_ru = "РАЗРЕШИТЬ"
        color = "ЗЕЛЁНЫЙ"
    
    print(f"   {emoji} Решение:       {decision_ru} ({color})")
    print(f"   📈 Вероятность:   {результат['probability']:.1%}")
    print(f"   🏷️  Сегмент:       {результат['segment']}")
    print(f"   🎚️  Порог (fraud): {результат['threshold_fraud']:.1%}")
    print(f"   🎚️  Порог (review): {результат['threshold_review']:.1%}")
    
    # Причины
    if 'reasons' in результат and результат['reasons']:
        print(f"\n💡 ПРИЧИНЫ (найдено {len(результат['reasons'])} факторов риска):")
        for i, причина in enumerate(результат['reasons'], 1):
            print(f"   {i}. {причина}")
    
    # Объяснение решения
    if 'decision_text' in результат:
        print(f"\n📝 ОБЪЯСНЕНИЕ:")
        print(f"   {результат['decision_text']}")


# ============================================================================
# ПРИМЕРЫ ТРАНЗАКЦИЙ
# ============================================================================

def пример_1_нормальная_транзакция():
    """
    Пример 1: Обычная нормальная транзакция
    Клиент переводит обычную сумму своему знакомому получателю
    """
    print("\n" + "🟢" * 40)
    print("ПРИМЕР 1: НОРМАЛЬНАЯ БЕЗОПАСНАЯ ТРАНЗАКЦИЯ")
    print("🟢" * 40)
    
    транзакция = {
        # === ОБЯЗАТЕЛЬНЫЕ ПОЛЯ ===
        "cst_id": "CST_12345",              # ID клиента
        "target_id": "TGT_67890",           # ID получателя
        "amount": 50000,                    # Сумма транзакции в тенге
        
        # === ИСТОРИЯ КЛИЕНТА ===
        "cst_amount_mean_past": 45000,      # Средняя сумма переводов клиента (обычно 45к)
        "cst_txn_count_past": 150,          # Количество прошлых транзакций (опытный клиент)
        "amount_over_mean_past": 1.1,       # Сумма в 1.1 раза больше средней (норма)
        
        # === ИСТОРИЯ ПОЛУЧАТЕЛЯ ===
        "target_fraud_rate_past_smooth": 0.005,  # Fraud rate получателя 0.5% (низкий)
        "target_txn_count_past": 200,       # Получатель получал 200 переводов (надёжный)
        
        # === ФЛАГИ НОВИЗНЫ ===
        "is_new_target_for_client": 0,      # 0 = знакомый получатель
        "is_new_phone_model_for_client": 0, # 0 = обычный телефон
        "is_new_os_for_client": 0,          # 0 = обычная ОС
        "is_first_tx_for_client": 0,        # 0 = не первая транзакция
        
        # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
        "is_night_tx": 0,                   # 0 = днём (не ночью)
        "is_weekend": 0,                    # 0 = будний день
        "hours_since_prev_trans": 24.0,     # 24 часа с последней транзакции (норма)
        
        # === ПОВЕДЕНИЕ ===
        "sessions_unique_7d": 15,           # 15 сессий за неделю
        "sessions_unique_30d": 60,          # 60 сессий за месяц
        "daily_logins_avg_7d": 3.0,         # В среднем 3 входа в день
        "daily_logins_avg_30d": 2.8,        # Стабильное поведение
    }
    
    отправить_транзакцию(транзакция)


def пример_2_подозрительная_транзакция():
    """
    Пример 2: Подозрительная транзакция
    Новый получатель, большая сумма, ночью, с нового устройства
    """
    print("\n" + "🟡" * 40)
    print("ПРИМЕР 2: ПОДОЗРИТЕЛЬНАЯ ТРАНЗАКЦИЯ (НА ПРОВЕРКУ)")
    print("🟡" * 40)
    
    транзакция = {
        # === ОБЯЗАТЕЛЬНЫЕ ПОЛЯ ===
        "cst_id": "CST_99999",
        "target_id": "TGT_SUSPICIOUS_123",
        "amount": 500000,                   # Большая сумма - 500к
        
        # === ИСТОРИЯ КЛИЕНТА ===
        "cst_amount_mean_past": 250000,      # Обычно переводит 50к
        "cst_txn_count_past": 120,           # Мало транзакций
        "amount_over_mean_past": 2.0,      # В 10 раз больше обычного! 🚨
        
        # === ИСТОРИЯ ПОЛУЧАТЕЛЯ ===
        "target_fraud_rate_past_smooth": 0.08,  # Fraud rate 8% - подозрительно! 🚨
        "target_txn_count_past": 5,         # Получатель новый (всего 5 переводов)
        
        # === ФЛАГИ НОВИЗНЫ ===
        "is_new_target_for_client": 1,      # 1 = новый получатель для клиента! 🚨
        "is_new_phone_model_for_client": 0, # 1 = новый телефон! 🚨
        "is_new_os_for_client": 0,
        "is_first_tx_for_client": 0,
        
        # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
        "is_night_tx": 1,                   # 1 = ночью (02:00)! 🚨
        "is_weekend": 1,                    # 1 = выходной! 🚨
        "hours_since_prev_trans": 0.5,      # Всего 30 минут с последней транзакции
        
        # === ПОВЕДЕНИЕ ===
        "sessions_unique_7d": 120,            # Мало активности
        "sessions_unique_30d": 324,
        "daily_logins_avg_7d": 0.5,         # Почти не заходит
        "daily_logins_avg_30d": 0.3,
    }
    
    отправить_транзакцию(транзакция)


def пример_3_очень_опасная_транзакция():
    """
    Пример 3: Очень опасная транзакция
    Все признаки мошенничества
    """
    print("\n" + "🔴" * 40)
    print("ПРИМЕР 3: ОЧЕНЬ ОПАСНАЯ ТРАНЗАКЦИЯ (БЛОКИРОВКА)")
    print("🔴" * 40)
    
    транзакция = {
        # === ОБЯЗАТЕЛЬНЫЕ ПОЛЯ ===
        "cst_id": "CST_HACKER_001",
        "target_id": "TGT_FRAUD_999",
        "amount": 2000000,                  # ОЧЕНЬ большая сумма - 2 миллиона!
        
        # === ИСТОРИЯ КЛИЕНТА ===
        "cst_amount_mean_past": 30000,      # Обычно переводит 30к
        "cst_txn_count_past": 3,            # Почти новый клиент
        "amount_over_mean_past": 66.0,      # В 66 раз больше обычного!!! 🚨🚨🚨
        
        # === ИСТОРИЯ ПОЛУЧАТЕЛЯ ===
        "target_fraud_rate_past_smooth": 0.45,  # Fraud rate 45%!!! 🚨🚨🚨
        "target_txn_count_past": 2,         # Получатель совсем новый
        
        # === ФЛАГИ НОВИЗНЫ ===
        "is_new_target_for_client": 1,      # 1 = новый получатель! 🚨
        "is_new_phone_model_for_client": 1, # 1 = новый телефон! 🚨
        "is_new_os_for_client": 1,          # 1 = новая ОС! 🚨
        "is_first_tx_for_client": 0,
        
        # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
        "is_night_tx": 1,                   # 1 = глубокая ночь! 🚨
        "is_weekend": 1,                    # 1 = воскресенье! 🚨
        "hours_since_prev_trans": 0.1,      # Всего 6 минут с прошлой транзакции! 🚨
        
        # === ПОВЕДЕНИЕ ===
        "sessions_unique_7d": 1,            # Почти не заходил
        "sessions_unique_30d": 2,           # Подозрительно мало активности
        "daily_logins_avg_7d": 0.2,         # Почти не пользуется
        "daily_logins_avg_30d": 0.1,
    }
    
    отправить_транзакцию(транзакция)


def пример_4_минимальные_данные():
    """
    Пример 4: Минимальный набор данных
    Только обязательные поля - остальное API заполнит по умолчанию
    """
    print("\n" + "⚪" * 40)
    print("ПРИМЕР 4: МИНИМАЛЬНЫЕ ДАННЫЕ (ТОЛЬКО ОБЯЗАТЕЛЬНЫЕ ПОЛЯ)")
    print("⚪" * 40)
    
    транзакция = {
        "cst_id": "CST_SIMPLE",
        "target_id": "TGT_SIMPLE",
        "amount": 100000,
    }
    
    print("\n💡 API автоматически подставит значения по умолчанию для остальных полей")
    
    отправить_транзакцию(транзакция)


def пример_5_пакетная_проверка():
    """
    Пример 5: Пакетная проверка нескольких транзакций сразу
    """
    print("\n" + "🔵" * 40)
    print("ПРИМЕР 5: ПАКЕТНАЯ ПРОВЕРКА (5 ТРАНЗАКЦИЙ)")
    print("🔵" * 40)
    
    транзакции = [
        {
            "cst_id": f"CST_BATCH_{i}",
            "target_id": f"TGT_BATCH_{i}",
            "amount": 100000 + i * 50000,
            "cst_amount_mean_past": 100000,
            "target_fraud_rate_past_smooth": 0.01 + i * 0.02,
        }
        for i in range(5)
    ]
    
    print(f"\n📦 Отправляем {len(транзакции)} транзакций одним запросом...\n")
    
    try:
        response = requests.post(
            f"{API_URL}/score_batch",
            json=транзакции,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ РЕЗУЛЬТАТЫ ПАКЕТНОЙ ПРОВЕРКИ:")
            print(f"   Обработано транзакций: {result['count']}\n")
            
            for i, res in enumerate(result['results'], 1):
                decision_emoji = {
                    "BLOCK": "🚫",
                    "REVIEW": "⚠️",
                    "ACCEPT": "✅"
                }.get(res['decision'], "❓")
                
                print(f"   {i}. {decision_emoji} {res['cst_id']} → {res['target_id']}")
                print(f"      Сумма: ₸{res['amount']:,.0f} | "
                      f"Решение: {res['decision']} | "
                      f"Вероятность: {res['probability']:.1%}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"   {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")


# ============================================================================
# ГЛАВНОЕ МЕНЮ
# ============================================================================

def показать_меню():
    """Показывает меню с примерами"""
    print("\n" + "=" * 80)
    print("🔐 FRAUD DETECTION API - ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ")
    print("=" * 80)
    print("\nВыберите пример транзакции для проверки:\n")
    print("  1. 🟢 Нормальная безопасная транзакция")
    print("  2. 🟡 Подозрительная транзакция (на проверку)")
    print("  3. 🔴 Очень опасная транзакция (блокировка)")
    print("  4. ⚪ Минимальные данные (только обязательные поля)")
    print("  5. 🔵 Пакетная проверка (5 транзакций)")
    print("  6. ℹ️  Информация о API (health check)")
    print("  0. 🚪 Выход")
    print("\n" + "=" * 80)


def проверить_api():
    """Проверяет работоспособность API"""
    print("\n" + "=" * 80)
    print("ℹ️  ИНФОРМАЦИЯ О API")
    print("=" * 80)
    
    try:
        # Health check
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print("\n✅ API работает!")
            print(f"   🏷️  Модель:     {info['model_name']}")
            print(f"   📦 Версия:     {info['model_version']}")
            print(f"   📊 CV AUC:     {info['cv_auc']:.4f}")
            print(f"   🌐 URL:        {API_URL}")
        else:
            print(f"\n⚠️  API вернул статус: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("\n❌ API не доступен!")
        print("   Запустите сервер: python api/main.py")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


def main():
    """Главная функция"""
    
    while True:
        показать_меню()
        
        try:
            выбор = input("Введите номер (0-6): ").strip()
            
            if выбор == "1":
                пример_1_нормальная_транзакция()
            elif выбор == "2":
                пример_2_подозрительная_транзакция()
            elif выбор == "3":
                пример_3_очень_опасная_транзакция()
            elif выбор == "4":
                пример_4_минимальные_данные()
            elif выбор == "5":
                пример_5_пакетная_проверка()
            elif выбор == "6":
                проверить_api()
            elif выбор == "0":
                print("\n👋 До свидания!")
                break
            else:
                print("\n❌ Неверный выбор. Попробуйте снова.")
            
            input("\n⏎ Нажмите Enter для продолжения...")
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   🔐 FRAUD DETECTION API - CLIENT                           ║
║                                                                              ║
║              Скрипт для отправки транзакций на проверку                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Проверяем доступность API
    проверить_api()
    
    # Запускаем главное меню
    main()

