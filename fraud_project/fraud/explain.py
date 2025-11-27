"""
Explain Module
Generates human-readable explanations for fraud scores
"""

from typing import Dict, Any, List


def explain_tx(tx_row: Dict[str, Any]) -> List[str]:
    """
    Generate explanations for why a transaction was flagged as risky.
    
    Args:
        tx_row: Dict with transaction data and computed features
    
    Returns:
        List of explanation strings
    """
    reasons = []
    
    # Amount anomaly
    a_over = tx_row.get("amount_over_mean_past")
    if a_over is not None and a_over >= 5:
        reasons.append(f"⚠️ Сумма в {a_over:.1f} раза выше средней для клиента")
    elif a_over is not None and a_over >= 3:
        reasons.append(f"⚠️ Сумма в {a_over:.1f} раза выше обычной")
    
    # New target
    if tx_row.get("is_new_target_for_client") == 1:
        reasons.append("🆕 Перевод на нового получателя")
    
    # Night transaction
    if tx_row.get("is_night_tx") == 1:
        reasons.append("🌙 Транзакция совершена ночью (00:00-06:00)")
    
    # Weekend transaction
    if tx_row.get("is_weekend") == 1:
        reasons.append("📅 Транзакция в выходные")
    
    # New device
    if tx_row.get("is_new_phone_model_for_client") == 1:
        reasons.append("📱 Используется новый телефон/устройство")
    
    # New OS
    if tx_row.get("is_new_os_for_client") == 1:
        reasons.append("💻 Используется новая операционная система")
    
    # High fraud target
    t_rate = tx_row.get("target_fraud_rate_past_smooth")
    if t_rate is not None and t_rate > 0.05:
        reasons.append(f"⚡ Получатель имеет высокий уровень фрода ({t_rate*100:.1f}%)")
    elif t_rate is not None and t_rate > 0.02:
        reasons.append(f"⚡ Получатель имеет повышенный уровень фрода ({t_rate*100:.1f}%)")
    
    # Risk score
    risk_score = tx_row.get("risk_score")
    if risk_score is not None and risk_score >= 3:
        reasons.append(f"🔴 Высокий общий риск-счёт: {risk_score}/4")
    elif risk_score is not None and risk_score >= 2:
        reasons.append(f"🟡 Средний риск-счёт: {risk_score}/4")
    
    # Time since last transaction
    hours_since = tx_row.get("hours_since_prev_trans")
    if hours_since is not None and hours_since < 0.1:  # less than 6 minutes
        reasons.append("⚡ Очень быстрая последовательность транзакций")
    
    # Login spike
    if tx_row.get("is_login_spike") == 1:
        reasons.append("📈 Резкий всплеск активности входов")
    
    # First transaction
    if tx_row.get("is_first_tx_for_client") == 1:
        reasons.append("🆕 Первая транзакция клиента")
    
    # High absolute amount
    if tx_row.get("is_high_amount_global") == 1:
        amount = tx_row.get("amount", 0)
        reasons.append(f"💰 Очень высокая сумма (₸{amount:,.0f})")
    
    # If no specific reasons, give generic one
    if not reasons:
        reasons.append("🔍 Комбинация поведенческих признаков указывает на повышенный риск")
    
    return reasons


def explain_decision(
    tx_row: Dict[str, Any],
    probability: float,
    decision: str,
    segment: str,
    threshold_fraud: float,
    threshold_review: float
) -> Dict[str, Any]:
    """
    Generate full explanation for the decision
    
    Returns:
        Dict with explanation details
    """
    
    reasons = explain_tx(tx_row)
    
    # Decision explanation
    if decision == "BLOCK":
        decision_text = f"🚫 БЛОКИРОВАТЬ - вероятность фрода {probability:.1%} превышает порог {threshold_fraud:.1%}"
    elif decision == "REVIEW":
        decision_text = f"⚠️ НА ПРОВЕРКУ - вероятность фрода {probability:.1%} требует ручной проверки"
    else:
        decision_text = f"✅ РАЗРЕШИТЬ - вероятность фрода {probability:.1%} ниже порога {threshold_review:.1%}"
    
    return {
        "decision": decision,
        "decision_text": decision_text,
        "probability": probability,
        "segment": segment,
        "threshold_fraud": threshold_fraud,
        "threshold_review": threshold_review,
        "reasons": reasons,
        "risk_factors_count": len(reasons)
    }

