"""
Automated endpoint testing for FastAPI backend.
"""
import pytest
import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

class TestEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Wait for server to be ready
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                if i == max_retries - 1:
                    pytest.skip("Backend server not available")
                time.sleep(2)
    
    def get_sample_input(self) -> Dict[str, Any]:
        """Get sample input data for testing."""
        return {
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
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Procurement Risk Prediction API"
    
    def test_single_prediction(self):
        """Test single prediction endpoint."""
        input_data = self.get_sample_input()
        
        response = requests.post(
            f"{BASE_URL}/predict/single",
            json=input_data,
            headers={"X-Session-ID": "test-session"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["prediction"] in ["low", "medium", "high"]
        assert 0 <= data["confidence"] <= 1
        
        print(f"✅ Prediction: {data['prediction']} (confidence: {data['confidence']:.2f})")
    
    def test_prediction_health(self):
        """Test prediction service health."""
        response = requests.get(f"{BASE_URL}/predict/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] == True
    
    def test_session_creation(self):
        """Test session creation."""
        response = requests.post(
            f"{BASE_URL}/history/session",
            json={"user_id": "test-user"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "created_at" in data
        
        print(f"✅ Session created: {data['session_id']}")
    
    def test_llm_health(self):
        """Test LLM service health."""
        response = requests.get(f"{BASE_URL}/llm/health")
        
        # LLM might not be available, so we check for expected responses
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ LLM service is healthy")
            else:
                print(f"⚠️ LLM service issue: {data.get('error', 'Unknown')}")
        else:
            print("⚠️ LLM service not available")
    
    def test_chat_endpoint(self):
        """Test chat endpoint."""
        chat_data = {
            "message": "Hello, can you help me understand procurement risks?",
            "session_id": "test-session",
            "include_context": True
        }
        
        response = requests.post(f"{BASE_URL}/llm/chat", json=chat_data)
        
        # Chat might fail if Ollama is not running
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✅ Chat response: {data['data']['response'][:100]}...")
            else:
                print(f"⚠️ Chat failed: {data.get('error', 'Unknown')}")
        else:
            print("⚠️ Chat endpoint not available")

if __name__ == "__main__":
    # Run tests manually
    test_instance = TestEndpoints()
    test_instance.setup()
    
    print("🧪 Running API Endpoint Tests")
    print("=" * 50)
    
    try:
        test_instance.test_health_endpoint()
        print("✅ Health endpoint test passed")
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
    
    try:
        test_instance.test_root_endpoint()
        print("✅ Root endpoint test passed")
    except Exception as e:
        print(f"❌ Root endpoint test failed: {e}")
    
    try:
        test_instance.test_single_prediction()
        print("✅ Single prediction test passed")
    except Exception as e:
        print(f"❌ Single prediction test failed: {e}")
    
    try:
        test_instance.test_prediction_health()
        print("✅ Prediction health test passed")
    except Exception as e:
        print(f"❌ Prediction health test failed: {e}")
    
    try:
        test_instance.test_session_creation()
        print("✅ Session creation test passed")
    except Exception as e:
        print(f"❌ Session creation test failed: {e}")
    
    try:
        test_instance.test_llm_health()
        print("✅ LLM health test completed")
    except Exception as e:
        print(f"❌ LLM health test failed: {e}")
    
    try:
        test_instance.test_chat_endpoint()
        print("✅ Chat endpoint test completed")
    except Exception as e:
        print(f"❌ Chat endpoint test failed: {e}")
    
    print("=" * 50)
    print("🎉 API testing completed!")