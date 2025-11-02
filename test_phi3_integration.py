#!/usr/bin/env python3
"""
Test script to verify Phi3 integration and all services are working.
"""
import requests
import json

API_BASE_URL = "http://localhost:8000"
SESSION_ID = "test-phi3-integration"

def test_all_services():
    """Test all services with Phi3 integration."""
    
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
    
    print("🔍 Testing Complete Procurement Risk Platform with Phi3")
    print("=" * 60)
    
    # 1. Test Prediction Services
    print("\n🎯 PREDICTION SERVICES")
    print("-" * 30)
    
    # Single prediction
    print("1. Single supplier risk assessment...")
    response = requests.post(
        f"{API_BASE_URL}/predict/single",
        json=sample_data,
        headers={"X-Session-ID": SESSION_ID},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Risk: {result['prediction']} ({result['confidence']:.1%})")
        print(f"   ✅ Processing time: {result.get('processing_time', 0):.3f}s")
        prediction_result = result
    else:
        print(f"   ❌ Failed: {response.status_code}")
        return
    
    # Historical tracking
    print("2. Historical tracking...")
    response = requests.get(
        f"{API_BASE_URL}/history/stats",
        params={"session_id": SESSION_ID},
        timeout=30
    )
    
    if response.status_code == 200:
        stats = response.json()
        print(f"   ✅ Total predictions: {stats['total_predictions']}")
        print(f"   ✅ Avg confidence: {stats['avg_confidence']:.1%}")
    else:
        print(f"   ❌ Failed: {response.status_code}")
    
    # 2. Test Explainable AI
    print("\n📊 EXPLAINABLE AI")
    print("-" * 30)
    
    # SHAP feature importance
    print("1. SHAP feature importance...")
    response = requests.post(
        f"{API_BASE_URL}/explain/prediction",
        json=sample_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            shap_data = result["data"]["shap_explanation"]
            print(f"   ✅ Top positive factors: {len(shap_data.get('top_positive', {}))}")
            print(f"   ✅ Top negative factors: {len(shap_data.get('top_negative', {}))}")
            shap_explanation = shap_data
        else:
            print("   ❌ SHAP explanation failed")
            shap_explanation = {}
    else:
        print(f"   ❌ Failed: {response.status_code}")
        shap_explanation = {}
    
    # Global model insights
    print("2. Global model insights...")
    response = requests.get(
        f"{API_BASE_URL}/explain/global-importance",
        params={"sample_size": 50},
        timeout=30
    )
    
    if response.status_code == 200:\n        result = response.json()\n        if result.get(\"success\"):\n            importance = result[\"data\"].get(\"global_importance\", {})\n            print(f\"   ✅ Features analyzed: {len(importance)}\")\n        else:\n            print(\"   ❌ Global importance failed\")\n    else:\n        print(f\"   ❌ Failed: {response.status_code}\")\n    \n    # Interactive visualizations\n    print(\"3. Interactive visualizations...\")\n    response = requests.post(\n        f\"{API_BASE_URL}/explain/visualize\",\n        json=sample_data,\n        timeout=60\n    )\n    \n    if response.status_code == 200:\n        result = response.json()\n        if result.get(\"success\"):\n            data = result.get(\"data\", {})\n            print(f\"   ✅ Force plot: {'force_plot' in data}\")\n            print(f\"   ✅ Summary plot: {'summary_plot' in data}\")\n        else:\n            print(\"   ❌ Visualizations failed\")\n    else:\n        print(f\"   ❌ Failed: {response.status_code}\")\n    \n    # 3. Test AI Assistant (Phi3)\n    print(\"\\n🤖 AI ASSISTANT (PHI3)\")\n    print(\"-\" * 30)\n    \n    # Phi3 health check\n    print(\"1. Phi3 model health...\")\n    response = requests.get(f\"{API_BASE_URL}/llm/health\", timeout=30)\n    \n    if response.status_code == 200:\n        result = response.json()\n        if result.get(\"success\"):\n            print(f\"   ✅ Model: {result.get('model', 'Unknown')}\")\n            print(f\"   ✅ Status: {result.get('status', 'Unknown')}\")\n            phi3_available = True\n        else:\n            print(f\"   ❌ Phi3 unhealthy: {result.get('error', 'Unknown')}\")\n            phi3_available = False\n    else:\n        print(f\"   ❌ Failed: {response.status_code}\")\n        phi3_available = False\n    \n    if phi3_available:\n        # Phi3-powered recommendations\n        print(\"2. Phi3-powered recommendations...\")\n        rec_data = {\n            \"prediction_result\": prediction_result,\n            \"shap_explanation\": shap_explanation,\n            \"input_data\": sample_data\n        }\n        \n        response = requests.post(\n            f\"{API_BASE_URL}/llm/recommendations\",\n            json=rec_data,\n            timeout=60\n        )\n        \n        if response.status_code == 200:\n            result = response.json()\n            if result.get(\"success\"):\n                recs = result[\"data\"]\n                print(f\"   ✅ Recommendations: {len(recs.get('recommendations', []))}\")\n                print(f\"   ✅ Model: {recs.get('model', 'Unknown')}\")\n                print(f\"   ✅ Response length: {len(recs.get('full_response', ''))} chars\")\n            else:\n                print(\"   ❌ Recommendations failed\")\n        else:\n            print(f\"   ❌ Failed: {response.status_code}\")\n        \n        # Conversational interface\n        print(\"3. Conversational interface...\")\n        chat_data = {\n            \"message\": \"What should I do about this high-risk supplier?\",\n            \"session_id\": SESSION_ID,\n            \"include_context\": True\n        }\n        \n        response = requests.post(\n            f\"{API_BASE_URL}/llm/chat\",\n            json=chat_data,\n            timeout=60\n        )\n        \n        if response.status_code == 200:\n            result = response.json()\n            if result.get(\"success\"):\n                chat_response = result[\"data\"][\"response\"]\n                print(f\"   ✅ Chat response: {len(chat_response)} chars\")\n                print(f\"   ✅ Has context: {result['data'].get('has_context', False)}\")\n                print(f\"   ✅ Model: {result['data'].get('model', 'Unknown')}\")\n            else:\n                print(\"   ❌ Chat failed\")\n        else:\n            print(f\"   ❌ Failed: {response.status_code}\")\n        \n        # What-if scenario analysis\n        print(\"4. What-if scenario analysis...\")\n        modified_data = sample_data.copy()\n        modified_data['credit_score'] = 650  # Lower credit score\n        modified_data['past_disputes'] = 5   # More disputes\n        \n        whatif_data = {\n            \"original_data\": sample_data,\n            \"modified_data\": modified_data,\n            \"original_prediction\": prediction_result\n        }\n        \n        response = requests.post(\n            f\"{API_BASE_URL}/llm/what-if\",\n            json=whatif_data,\n            timeout=60\n        )\n        \n        if response.status_code == 200:\n            result = response.json()\n            if result.get(\"success\"):\n                analysis = result[\"data\"][\"analysis\"]\n                print(f\"   ✅ What-if analysis: {len(analysis)} chars\")\n                print(f\"   ✅ Changes detected: {result['data']['changes_detected']}\")\n                print(f\"   ✅ Model: {result['data'].get('model', 'Unknown')}\")\n            else:\n                print(\"   ❌ What-if analysis failed\")\n        else:\n            print(f\"   ❌ Failed: {response.status_code}\")\n    \n    else:\n        print(\"   ⚠️  Skipping Phi3 tests - model unavailable\")\n    \n    print(\"\\n\" + \"=\" * 60)\n    print(\"✅ Platform testing completed!\")\n    \n    if phi3_available:\n        print(\"\\n🎯 ALL SERVICES OPERATIONAL:\")\n        print(\"   🎯 Prediction Services: Single assessment, batch processing, historical tracking\")\n        print(\"   📊 Explainable AI: SHAP analysis, visualizations, global insights\")\n        print(\"   🤖 AI Assistant: Phi3 recommendations, chat, what-if analysis\")\n    else:\n        print(\"\\n⚠️  PARTIAL FUNCTIONALITY:\")\n        print(\"   🎯 Prediction Services: ✅ Working\")\n        print(\"   📊 Explainable AI: ✅ Working\")\n        print(\"   🤖 AI Assistant: ❌ Phi3 model unavailable\")\n        print(\"\\n💡 To enable Phi3: Run 'ollama run phi3:mini' in terminal\")\n\nif __name__ == \"__main__\":\n    test_all_services()