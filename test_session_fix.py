#!/usr/bin/env python3
"""
Test script to verify session state fixes are working.
"""
import requests
import json

API_BASE_URL = "http://localhost:8000"
SESSION_ID = "test-session-fix"

def test_prediction_and_context():
    """Test that predictions are stored and accessible for other endpoints."""
    
    # Sample prediction data
    sample_data = {
        "annual_revenue": 1000000,
        "annual_spend": 500000,
        "avg_payment_delay_days": 12,
        "contract_value": 250000,
        "contract_duration_months": 12,
        "past_disputes": 2,
        "delivery_score": 85,
        "financial_stability_index": 0.75,
        "relationship_years": 5,
        "credit_score": 720,
        "txn_count": 10,
        "avg_txn_amount": 25000,
        "avg_delay": 12,
        "late_ratio": 0.2,
        "dispute_rate": 0.1,
        "avg_delivery_quality": 88,
        "clause_risk_score": 25,
        "region": "North America",
        "industry": "IT Services",
        "contract_criticality": "Medium"
    }
    
    print("🔍 Testing Session State Fixes")
    print("=" * 50)
    
    # 1. Make a prediction
    print("\n1. Making single prediction...")
    response = requests.post(
        f"{API_BASE_URL}/predict/single",
        json=sample_data,
        headers={"X-Session-ID": SESSION_ID},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction: {result['prediction']} ({result['confidence']:.1%})")
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        return
    
    # 2. Test SHAP explanation
    print("\n2. Testing SHAP explanation...")
    response = requests.post(
        f"{API_BASE_URL}/explain/prediction",
        json=sample_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ SHAP explanation: {result.get('success')}")
    else:
        print(f"❌ SHAP explanation failed: {response.status_code}")
    
    # 3. Test LLM recommendations
    print("\n3. Testing LLM recommendations...")
    rec_data = {
        "prediction_result": {"prediction": "medium", "confidence": 0.75},
        "shap_explanation": {"top_negative": {"payment_delay": -0.2}},
        "input_data": sample_data
    }
    
    response = requests.post(
        f"{API_BASE_URL}/llm/recommendations",
        json=rec_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ LLM recommendations: {result.get('success')}")
        if result.get('success'):
            recs = result['data'].get('recommendations', [])
            print(f"   Generated {len(recs)} recommendations")
    else:
        print(f"❌ LLM recommendations failed: {response.status_code}")
    
    # 4. Test chat with context
    print("\n4. Testing chat with context...")
    chat_data = {
        "message": "What should I do about this supplier?",
        "session_id": SESSION_ID,
        "include_context": True
    }
    
    response = requests.post(
        f"{API_BASE_URL}/llm/chat",
        json=chat_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Chat response: {result.get('success')}")
        if result.get('success'):
            response_text = result['data']['response'][:100] + "..."
            print(f"   Response: {response_text}")
    else:
        print(f"❌ Chat failed: {response.status_code}")
    
    # 5. Test session stats
    print("\n5. Testing session statistics...")
    response = requests.get(
        f"{API_BASE_URL}/history/stats",
        params={"session_id": SESSION_ID},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Session stats: {result['total_predictions']} predictions")
        print(f"   Risk distribution: {result['risk_distribution']}")
    else:
        print(f"❌ Session stats failed: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("✅ Session state testing completed!")
    print("\n💡 Frontend should now properly share data between pages.")

if __name__ == "__main__":
    test_prediction_and_context()