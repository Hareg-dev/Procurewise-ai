"""
Integration testing for complete data flow: Frontend → Backend → Model → SHAP → LLM → Database.
"""
import requests
import sqlite3
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
DB_PATH = Path(__file__).parent.parent / "backend.db"

class IntegrationTester:
    """Integration testing for complete system flow."""
    
    def __init__(self):
        self.session_id = None
        self.test_results = []
    
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "status": status,
            "details": details
        })
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def check_backend_health(self) -> bool:
        """Check if backend is healthy."""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            return response.status_code == 200 and response.json().get("status") == "healthy"
        except:
            return False
    
    def create_test_session(self) -> bool:
        """Create a test session."""
        try:
            response = requests.post(
                f"{BASE_URL}/history/session",
                json={"user_id": "integration-test-user"}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data["session_id"]
                return True
            return False
        except Exception as e:
            self.log_result("Session Creation", False, str(e))
            return False
    
    def test_single_prediction_flow(self) -> dict:
        """Test complete single prediction flow."""
        test_input = {
            "annual_revenue": 2500000,
            "annual_spend": 1200000,
            "avg_payment_delay_days": 25,
            "contract_value": 600000,
            "contract_duration_months": 24,
            "past_disputes": 5,
            "delivery_score": 70,
            "financial_stability_index": 0.60,
            "relationship_years": 8,
            "credit_score": 650,
            "txn_count": 25,
            "avg_txn_amount": 48000,
            "avg_delay": 20,
            "late_ratio": 0.4,
            "dispute_rate": 0.2,
            "avg_delivery_quality": 75,
            "clause_risk_score": 45,
            "region": "Europe",
            "industry": "Manufacturing",
            "contract_criticality": "High"
        }
        
        try:
            # Make prediction
            response = requests.post(
                f"{BASE_URL}/predict/single",
                json=test_input,
                headers={"X-Session-ID": self.session_id}
            )
            
            if response.status_code == 200:
                prediction_data = response.json()
                self.log_result(
                    "Single Prediction", 
                    True, 
                    f"Risk: {prediction_data['prediction']}, Confidence: {prediction_data['confidence']:.2f}"
                )
                return prediction_data
            else:
                self.log_result("Single Prediction", False, f"HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.log_result("Single Prediction", False, str(e))
            return None
    
    def test_llm_explanation(self, input_data: dict) -> str:
        """Test LLM explanation generation."""
        try:
            response = requests.post(
                f"{BASE_URL}/llm/explain-prediction",
                json=input_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    explanation = data["data"]["llm_explanation"]
                    self.log_result(
                        "LLM Explanation", 
                        True, 
                        f"Generated {len(explanation)} characters"
                    )
                    return explanation
                else:
                    self.log_result("LLM Explanation", False, "API returned success=False")
            else:
                self.log_result("LLM Explanation", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            self.log_result("LLM Explanation", False, str(e))
        
        return None
    
    def test_chat_interaction(self) -> bool:
        """Test chat interaction with LLM."""
        test_questions = [
            "What are the main risk factors in procurement?",
            "How can I reduce supplier risk?",
            "Explain SHAP values in simple terms"
        ]
        
        successful_chats = 0
        
        for question in test_questions:
            try:
                response = requests.post(
                    f"{BASE_URL}/llm/chat",
                    json={
                        "message": question,
                        "session_id": self.session_id,
                        "include_context": True
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        successful_chats += 1
                        
            except Exception:
                continue
        
        success = successful_chats > 0
        self.log_result(
            "Chat Interaction", 
            success, 
            f"{successful_chats}/{len(test_questions)} successful"
        )
        return success
    
    def test_database_logging(self) -> bool:
        """Test database logging functionality."""
        try:
            if not DB_PATH.exists():
                self.log_result("Database Logging", False, "Database file not found")
                return False
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ["user_sessions", "prediction_history"]
            missing_tables = [t for t in expected_tables if t not in tables]
            
            if missing_tables:
                self.log_result(
                    "Database Logging", 
                    False, 
                    f"Missing tables: {missing_tables}"
                )
                return False
            
            # Check for recent entries
            cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE session_id = ?", (self.session_id,))
            prediction_count = cursor.fetchone()[0]
            
            conn.close()
            
            success = prediction_count > 0
            self.log_result(
                "Database Logging", 
                success, 
                f"Found {prediction_count} predictions logged"
            )
            return success
            
        except Exception as e:
            self.log_result("Database Logging", False, str(e))
            return False
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction functionality."""
        # Create test CSV content
        csv_content = """annual_revenue,annual_spend,avg_payment_delay_days,contract_value,contract_duration_months,past_disputes,delivery_score,financial_stability_index,relationship_years,credit_score,txn_count,avg_txn_amount,avg_delay,late_ratio,dispute_rate,avg_delivery_quality,clause_risk_score,region,industry,contract_criticality
1000000,500000,15,250000,12,2,85,0.75,5,720,10,25000,12,0.2,0.1,88,25,North America,IT Services,Medium
2500000,1200000,25,600000,24,5,70,0.60,8,650,25,48000,20,0.4,0.2,75,45,Europe,Manufacturing,High"""
        
        try:
            files = {"file": ("test_batch.csv", csv_content, "text/csv")}
            headers = {"X-Session-ID": self.session_id}
            
            response = requests.post(
                f"{BASE_URL}/predict/batch",
                files=files,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                total_records = data.get("total_records", 0)
                successful = data.get("successful_predictions", 0)
                
                success = total_records > 0 and successful > 0
                self.log_result(
                    "Batch Prediction", 
                    success, 
                    f"{successful}/{total_records} successful predictions"
                )
                return success
            else:
                self.log_result("Batch Prediction", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_result("Batch Prediction", False, str(e))
            return False
    
    def run_integration_tests(self):
        """Run complete integration test suite."""
        print("🧪 Starting Integration Testing")
        print("=" * 60)
        
        # Check backend health
        if not self.check_backend_health():
            print("❌ Backend is not healthy. Please start the backend server.")
            return
        
        self.log_result("Backend Health", True, "API server is responding")
        
        # Create test session
        if not self.create_test_session():
            print("❌ Could not create test session. Stopping tests.")
            return
        
        self.log_result("Session Creation", True, f"Session ID: {self.session_id}")
        
        # Test single prediction flow
        test_input = {
            "annual_revenue": 2500000,
            "annual_spend": 1200000,
            "avg_payment_delay_days": 25,
            "contract_value": 600000,
            "contract_duration_months": 24,
            "past_disputes": 5,
            "delivery_score": 70,
            "financial_stability_index": 0.60,
            "relationship_years": 8,
            "credit_score": 650,
            "txn_count": 25,
            "avg_txn_amount": 48000,
            "avg_delay": 20,
            "late_ratio": 0.4,
            "dispute_rate": 0.2,
            "avg_delivery_quality": 75,
            "clause_risk_score": 45,
            "region": "Europe",
            "industry": "Manufacturing",
            "contract_criticality": "High"
        }
        
        prediction_result = self.test_single_prediction_flow()
        
        # Test LLM explanation
        if prediction_result:
            explanation = self.test_llm_explanation(test_input)
            if explanation:
                print(f"\n📝 LLM Explanation Sample:")
                print(f"   {explanation[:200]}...")
        
        # Test chat interaction
        self.test_chat_interaction()
        
        # Test batch prediction
        self.test_batch_prediction()
        
        # Test database logging
        time.sleep(1)  # Allow time for DB writes
        self.test_database_logging()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 Integration Test Summary")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if "✅" in result["status"])
        total = len(self.test_results)
        
        for result in self.test_results:
            print(f"{result['status']}: {result['test']}")
            if result['details']:
                print(f"   {result['details']}")
        
        print(f"\n🎯 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All integration tests passed!")
        else:
            print("⚠️ Some tests failed. Check the details above.")

if __name__ == "__main__":
    tester = IntegrationTester()
    tester.run_integration_tests()