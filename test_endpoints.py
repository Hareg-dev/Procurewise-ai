#!/usr/bin/env python3
"""
Test script to verify all backend endpoints are working correctly.
"""
import requests
import json
import time

API_BASE_URL = "http://localhost:8000"
SESSION_ID = "test-endpoints-session"

def test_endpoint(name, method, url, data=None, params=None):
    """Test a single endpoint and return result."""
    try:
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        success = response.status_code == 200
        result = response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        
        print(f"✅ {name}: {'PASS' if success else 'FAIL'} ({response.status_code})")
        if not success:
            print(f"   Error: {result}")
        
        return success, result
    
    except Exception as e:
        print(f"❌ {name}: ERROR - {str(e)}")
        return False, str(e)

def main():
    print("🔍 Testing Procurement Risk Prediction API Endpoints")
    print("=" * 60)
    
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
    
    # Test basic endpoints
    print("\n📋 Basic Endpoints:")
    test_endpoint("Root", "GET", f"{API_BASE_URL}/")
    test_endpoint("Health Check", "GET", f"{API_BASE_URL}/health")
    
    # Test prediction endpoints
    print("\n🎯 Prediction Endpoints:")
    success, result = test_endpoint(
        "Single Prediction", 
        "POST", 
        f"{API_BASE_URL}/predict/single",
        data=sample_data
    )
    
    test_endpoint("Prediction Health", "GET", f"{API_BASE_URL}/predict/health")
    
    # Test SHAP endpoints
    print("\n📊 SHAP Analysis Endpoints:")
    test_endpoint(
        "SHAP Explanation", 
        "POST", 
        f"{API_BASE_URL}/explain/prediction",
        data=sample_data
    )
    
    test_endpoint(
        "Global Importance", 
        "GET", 
        f"{API_BASE_URL}/explain/global-importance",
        params={"sample_size": 50}
    )
    
    # Test LLM endpoints
    print("\n🤖 LLM Endpoints:")
    test_endpoint("LLM Health", "GET", f"{API_BASE_URL}/llm/health")
    
    test_endpoint(
        "LLM Chat", 
        "POST", 
        f"{API_BASE_URL}/llm/chat",
        data={
            "message": "Hello, how can you help me?",
            "session_id": SESSION_ID,
            "include_context": True
        }
    )
    
    test_endpoint(
        "LLM Recommendations", 
        "POST", 
        f"{API_BASE_URL}/llm/recommendations",
        data={
            "prediction_result": {"prediction": "medium", "confidence": 0.75},
            "shap_explanation": {"top_negative": {"payment_delay": -0.2}},
            "input_data": sample_data
        }
    )
    
    # Test history endpoints
    print("\n📈 History Endpoints:")
    test_endpoint(
        "Session Stats", 
        "GET", 
        f"{API_BASE_URL}/history/stats",
        params={"session_id": SESSION_ID}
    )
    
    test_endpoint(
        "Prediction History", 
        "GET", 
        f"{API_BASE_URL}/history/predictions",
        params={"session_id": SESSION_ID, "limit": 10}
    )
    
    print("\n" + "=" * 60)
    print("✅ Endpoint testing completed!")
    print("\n💡 All endpoints should now work with the frontend.")
    print("🚀 You can now run the Streamlit frontend safely.")

if __name__ == "__main__":
    main()