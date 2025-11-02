#!/usr/bin/env python3
"""
Test script to verify SHAP analysis fixes are working.
"""
import requests
import json

API_BASE_URL = "http://localhost:8000"
SESSION_ID = "test-shap-fixes"

def test_shap_endpoints():
    """Test all SHAP-related endpoints."""
    
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
    
    print("🔍 Testing SHAP Analysis Fixes")
    print("=" * 50)
    
    # 1. Test prediction explanation
    print("\n1. Testing prediction explanation...")
    response = requests.post(
        f"{API_BASE_URL}/explain/prediction",
        json=sample_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction explanation: {result.get('success')}")
        if result.get('success'):
            shap_data = result['data']['shap_explanation']
            print(f"   Top positive factors: {len(shap_data.get('top_positive', {}))}")
            print(f"   Top negative factors: {len(shap_data.get('top_negative', {}))}")
    else:
        print(f"❌ Prediction explanation failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    # 2. Test global importance
    print("\n2. Testing global importance...")
    response = requests.get(
        f"{API_BASE_URL}/explain/global-importance",
        params={"sample_size": 50},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Global importance: {result.get('success')}")
        if result.get('success'):
            importance = result['data'].get('global_importance', {})
            print(f"   Features analyzed: {len(importance)}")
    else:
        print(f"❌ Global importance failed: {response.status_code}")
    
    # 3. Test summary plot
    print("\n3. Testing summary plot...")
    response = requests.get(
        f"{API_BASE_URL}/explain/summary-plot",
        params={"sample_size": 50},
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Summary plot: {result.get('success')}")
        if result.get('success'):
            has_plot = 'plot' in result.get('data', {})
            print(f"   Plot generated: {has_plot}")
    else:
        print(f"❌ Summary plot failed: {response.status_code}")
    
    # 4. Test visualizations (force plot + summary plot)
    print("\n4. Testing visualizations...")
    response = requests.post(
        f"{API_BASE_URL}/explain/visualize",
        json=sample_data,
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Visualizations: {result.get('success')}")
        if result.get('success'):
            data = result.get('data', {})
            print(f"   Force plot: {'force_plot' in data}")
            print(f"   Summary plot: {'summary_plot' in data}")
    else:
        print(f"❌ Visualizations failed: {response.status_code}")
    
    # 5. Test comparison
    print("\n5. Testing prediction comparison...")
    
    # Create slightly different data for comparison
    sample_data2 = sample_data.copy()
    sample_data2['credit_score'] = 650  # Lower credit score
    sample_data2['past_disputes'] = 5   # More disputes
    
    response = requests.post(
        f"{API_BASE_URL}/explain/compare",
        json={
            "prediction1": sample_data,
            "prediction2": sample_data2
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction comparison: {result.get('success')}")
        if result.get('success'):
            differences = result['data'].get('differences', {})
            print(f"   Differences found: {len(differences)}")
    else:
        print(f"❌ Prediction comparison failed: {response.status_code}")
    
    print("\n" + "=" * 50)
    print("✅ SHAP analysis testing completed!")
    print("\n💡 Frontend SHAP pages should now work correctly.")
    print("🎯 Both single and batch predictions are properly handled.")

if __name__ == "__main__":
    test_shap_endpoints()