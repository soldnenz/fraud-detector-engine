#!/usr/bin/env python3
"""
Test API Script
Tests the fraud detection API with sample transactions
"""

import requests
import json


def test_health():
    """Test health endpoint"""
    print("=" * 60)
    print("Testing /health endpoint")
    print("=" * 60)
    
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_score_low_risk():
    """Test scoring a low-risk transaction"""
    print("=" * 60)
    print("Testing /score - LOW RISK transaction")
    print("=" * 60)
    
    tx = {
        "cst_id": "CST_NORMAL_001",
        "target_id": "TGT_KNOWN_001",
        "amount": 50000,  # Normal amount
        "last_phone_model": "iPhone 13",
        "last_os": "iOS 15",
        "cst_amount_mean_past": 45000,  # Consistent with history
        "cst_txn_count_past": 50,  # Established customer
        "amount_over_mean_past": 1.1,  # Slightly above mean
        "target_fraud_rate_past_smooth": 0.001,  # Low-risk target
        "target_txn_count_past": 100,  # Established target
        "is_new_target_for_client": 0,
        "is_new_phone_model_for_client": 0,
        "is_night_tx": 0,
        "is_weekend": 0,
        "sessions_unique_7d": 10,
        "sessions_unique_30d": 30,
        "daily_logins_avg_7d": 3.0,
        "daily_logins_avg_30d": 2.8
    }
    
    response = requests.post("http://localhost:8000/score", json=tx)
    print(f"Status: {response.status_code}")
    result = response.json()
    
    print(f"\n🎯 Decision: {result['decision']}")
    print(f"📊 Probability: {result['probability']:.1%}")
    print(f"📍 Segment: {result['segment']}")
    print(f"\n💡 Reasons:")
    for reason in result.get('reasons', []):
        print(f"   - {reason}")
    print()


def test_score_high_risk():
    """Test scoring a high-risk transaction"""
    print("=" * 60)
    print("Testing /score - HIGH RISK transaction")
    print("=" * 60)
    
    tx = {
        "cst_id": "CST_SUSPICIOUS_001",
        "target_id": "TGT_NEW_RISKY_001",
        "amount": 25000,  # Very high amount
        "last_phone_model": "Unknown",
        "last_os": "Unknown",
        "cst_amount_mean_past": 40000,  # Much lower than current
        "cst_txn_count_past": 5,  # New customer
        "amount_over_mean_past": 20.0,  # 20x higher than usual!
        "target_fraud_rate_past_smooth": 0.15,  # High-risk target
        "target_txn_count_past": 2,  # New target
        "is_new_target_for_client": 1,  # New target for this customer
        "is_new_phone_model_for_client": 1,  # New device
        "is_new_os_for_client": 1,  # New OS
        "is_night_tx": 1,  # Night transaction
        "is_weekend": 1,  # Weekend
        "sessions_unique_7d": 1,
        "sessions_unique_30d": 2,
        "daily_logins_avg_7d": 0.5,
        "daily_logins_avg_30d": 0.3
    }
    
    response = requests.post("http://localhost:8000/score", json=tx)
    print(f"Status: {response.status_code}")
    result = response.json()
    
    print(f"\n🎯 Decision: {result['decision']}")
    print(f"📊 Probability: {result['probability']:.1%}")
    print(f"📍 Segment: {result['segment']}")
    print(f"🔴 Risk Score: {result.get('risk_factors_count', 0)} factors")
    print(f"\n💡 Reasons:")
    for reason in result.get('reasons', []):
        print(f"   - {reason}")
    print()


def test_batch():
    """Test batch scoring"""
    print("=" * 60)
    print("Testing /score_batch")
    print("=" * 60)
    
    transactions = [
        {
            "cst_id": f"CST_{i:03d}",
            "target_id": f"TGT_{i:03d}",
            "amount": 100000 + i * 10000,
            "cst_amount_mean_past": 100000,
            "cst_txn_count_past": 10,
            "target_fraud_rate_past_smooth": 0.01
        }
        for i in range(5)
    ]
    
    response = requests.post("http://localhost:8000/score_batch", json=transactions)
    print(f"Status: {response.status_code}")
    result = response.json()
    
    print(f"\nProcessed {result['count']} transactions:")
    for r in result['results']:
        print(f"  {r['cst_id']} -> {r['target_id']}: {r['decision']} (p={r['probability']:.3f})")
    print()


if __name__ == "__main__":
    print("\n🧪 FRAUD DETECTION API TESTS\n")
    
    try:
        test_health()
        test_score_low_risk()
        test_score_high_risk()
        test_batch()
        
        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running:")
        print("   python api/main.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

