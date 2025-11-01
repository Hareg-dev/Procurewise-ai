#!/usr/bin/env python3
"""
Test script for the FastAPI backend endpoints.
"""
import sys
from pathlib import Path
import requests
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running.")
        return False

def test_single_prediction():
    """Test single prediction endpoint."""
    print("Testing single prediction...")
    
    test_data = {
        "annual_revenue": 1000000,
        "annual_spend": 500000,
        "avg_payment_delay_days": 15,
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
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/single",
            json=test_data,
            headers={"X-Session-ID": "test-session-123"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single prediction successful")
            print(f"   Risk Level: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Processing Time: {result['processing_time']:.3f}s")
            return True
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
        return False

def test_session_creation():
    """Test session creation endpoint."""
    print("Testing session creation...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/history/session",
            json={"user_id": "test-user"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Session creation successful")
            print(f"   Session ID: {result['session_id']}")
            return result['session_id']
        else:
            print(f"❌ Session creation failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return None

def main():
    """Main test function."""
    print("=" * 60)
    print("PROCUREMENT RISK PREDICTION API - BACKEND TESTING")
    print("=" * 60)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    tests_passed = 0
    total_tests = 3
    
    # Test health check
    if test_health_check():
        tests_passed += 1
    
    # Test session creation
    session_id = test_session_creation()
    if session_id:
        tests_passed += 1
    
    # Test single prediction
    if test_single_prediction():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"TESTS COMPLETED: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All backend tests passed!")
        print("The API is ready for frontend integration.")
    else:
        print("❌ Some tests failed. Check the API server.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()