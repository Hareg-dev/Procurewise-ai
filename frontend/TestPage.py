"""
Manual testing page for QA and sample data validation.
"""
import streamlit as st
import pandas as pd
import requests
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout import setup_page_config, render_header, render_sidebar, render_footer, show_loading, show_success, show_error, show_info

def load_test_data():
    """Load test data samples."""
    test_samples = [
        {
            "name": "Low Risk Supplier (Biotech)",
            "data": {
                "annual_revenue": 750000,
                "annual_spend": 400000,
                "avg_payment_delay_days": 8,
                "contract_value": 180000,
                "contract_duration_months": 18,
                "past_disputes": 1,
                "delivery_score": 92,
                "financial_stability_index": 0.85,
                "relationship_years": 3,
                "credit_score": 780,
                "txn_count": 8,
                "avg_txn_amount": 22500,
                "avg_delay": 6,
                "late_ratio": 0.1,
                "dispute_rate": 0.05,
                "avg_delivery_quality": 95,
                "clause_risk_score": 15,
                "region": "Asia-Pacific",
                "industry": "Biotech",
                "contract_criticality": "Low"
            }
        },
        {
            "name": "Medium Risk Supplier (IT Services)",
            "data": {
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
        },
        {
            "name": "High Risk Supplier (Logistics)",
            "data": {
                "annual_revenue": 500000,
                "annual_spend": 300000,
                "avg_payment_delay_days": 45,
                "contract_value": 120000,
                "contract_duration_months": 6,
                "past_disputes": 8,
                "delivery_score": 55,
                "financial_stability_index": 0.35,
                "relationship_years": 2,
                "credit_score": 580,
                "txn_count": 15,
                "avg_txn_amount": 20000,
                "avg_delay": 35,
                "late_ratio": 0.6,
                "dispute_rate": 0.4,
                "avg_delivery_quality": 60,
                "clause_risk_score": 65,
                "region": "LATAM",
                "industry": "Logistics",
                "contract_criticality": "High"
            }
        }
    ]
    return test_samples

def test_api_endpoint(endpoint: str, method: str = "GET", data: dict = None):
    """Test API endpoint and return response."""
    base_url = "http://localhost:8000"
    
    try:
        if method == "GET":
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
        elif method == "POST":
            response = requests.post(f"{base_url}{endpoint}", json=data, timeout=30)
        
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "data": response.json() if response.status_code == 200 else None,
            "error": response.text if response.status_code != 200 else None
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status_code": None,
            "data": None,
            "error": "Cannot connect to backend API. Is the server running?"
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "data": None,
            "error": str(e)
        }

def display_system_status():
    """Display system component status."""
    st.markdown("## 🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🖥️ Backend API")
        health_result = test_api_endpoint("/health")
        
        if health_result["success"]:
            st.success("✅ Online")
            st.json(health_result["data"])
        else:
            st.error("❌ Offline")
            st.text(health_result["error"])
    
    with col2:
        st.markdown("### 🤖 ML Model")
        model_result = test_api_endpoint("/predict/health")
        
        if model_result["success"]:
            st.success("✅ Loaded")
            data = model_result["data"]
            st.text(f"Model loaded: {data.get('model_loaded', False)}")
            st.text(f"Features: {data.get('feature_count', 0)}")
        else:
            st.error("❌ Not Available")
    
    with col3:
        st.markdown("### 🧠 LLM Service")
        llm_result = test_api_endpoint("/llm/health")
        
        if llm_result["success"] and llm_result["data"].get("success"):
            st.success("✅ Connected")
            st.text(f"Model: {llm_result['data'].get('model', 'Unknown')}")
        else:
            st.warning("⚠️ Ollama Not Available")
            if llm_result["data"]:
                st.text(llm_result["data"].get("error", "Unknown error"))

def display_sample_testing():
    """Display sample data testing interface."""
    st.markdown("## 🧪 Sample Data Testing")
    
    test_samples = load_test_data()
    
    # Sample selection
    sample_names = [sample["name"] for sample in test_samples]
    selected_sample = st.selectbox("Choose a test sample:", sample_names)
    
    # Get selected sample data
    sample_data = next(sample["data"] for sample in test_samples if sample["name"] == selected_sample)
    
    # Display sample data
    st.markdown("### 📋 Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Financial Data:**")
        st.text(f"Annual Revenue: ${sample_data['annual_revenue']:,}")
        st.text(f"Annual Spend: ${sample_data['annual_spend']:,}")
        st.text(f"Contract Value: ${sample_data['contract_value']:,}")
        st.text(f"Credit Score: {sample_data['credit_score']}")
        
        st.markdown("**Performance Data:**")
        st.text(f"Delivery Score: {sample_data['delivery_score']}")
        st.text(f"Payment Delay: {sample_data['avg_payment_delay_days']} days")
        st.text(f"Dispute Rate: {sample_data['dispute_rate']:.1%}")
    
    with col2:
        st.markdown("**Relationship Data:**")
        st.text(f"Relationship Years: {sample_data['relationship_years']}")
        st.text(f"Past Disputes: {sample_data['past_disputes']}")
        st.text(f"Financial Stability: {sample_data['financial_stability_index']:.2f}")
        
        st.markdown("**Contract Data:**")
        st.text(f"Region: {sample_data['region']}")
        st.text(f"Industry: {sample_data['industry']}")
        st.text(f"Criticality: {sample_data['contract_criticality']}")
    
    # Test prediction
    if st.button("🔍 Test Prediction", use_container_width=True):
        with show_loading("Making prediction..."):
            result = test_api_endpoint("/predict/single", "POST", sample_data)
            
            if result["success"]:
                prediction_data = result["data"]
                
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    risk_level = prediction_data["prediction"]
                    risk_icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                    st.metric("Risk Level", f"{risk_icons.get(risk_level, '⚪')} {risk_level.upper()}")
                
                with col2:
                    confidence = prediction_data["confidence"]
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    processing_time = prediction_data.get("processing_time", 0)
                    st.metric("Processing Time", f"{processing_time:.3f}s")
                
                # Probabilities
                st.markdown("### 📊 Risk Probabilities")
                probabilities = prediction_data["probabilities"]
                
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                
                with prob_col1:
                    st.metric("🟢 Low Risk", f"{probabilities.get('low', 0):.1%}")
                
                with prob_col2:
                    st.metric("🟡 Medium Risk", f"{probabilities.get('medium', 0):.1%}")
                
                with prob_col3:
                    st.metric("🔴 High Risk", f"{probabilities.get('high', 0):.1%}")
                
                show_success("Prediction completed successfully!")
                
            else:
                show_error(f"Prediction failed: {result['error']}")

def display_llm_testing():
    """Display LLM testing interface."""
    st.markdown("## 🤖 LLM Testing")
    
    # Test questions
    test_questions = [
        "What are the main factors that increase procurement risk?",
        "How can I reduce supplier payment delays?",
        "Explain the difference between high and medium risk suppliers",
        "What contract terms help mitigate procurement risks?",
        "How should I handle a supplier with declining performance?"
    ]
    
    selected_question = st.selectbox("Choose a test question:", ["Custom question..."] + test_questions)
    
    if selected_question == "Custom question...":
        user_question = st.text_input("Enter your question:")
    else:
        user_question = selected_question
    
    if st.button("💬 Ask LLM", use_container_width=True) and user_question:
        with show_loading("Getting LLM response..."):
            chat_data = {
                "message": user_question,
                "session_id": "test-session",
                "include_context": True
            }
            
            result = test_api_endpoint("/llm/chat", "POST", chat_data)
            
            if result["success"] and result["data"].get("success"):
                response = result["data"]["response"]
                
                st.markdown("### 🤖 LLM Response")
                st.markdown(response)
                
                show_success("LLM response received!")
                
            else:
                error_msg = result["data"].get("error", result["error"]) if result["data"] else result["error"]
                show_error(f"LLM request failed: {error_msg}")

def display_batch_testing():
    """Display batch testing interface."""
    st.markdown("## 📊 Batch Testing")
    
    # Load test CSV
    test_csv_path = Path(__file__).parent.parent / "tests" / "test_data" / "test_input_data.csv"
    
    if test_csv_path.exists():
        test_df = pd.read_csv(test_csv_path)
        
        st.markdown("### 📋 Test Data Preview")
        st.dataframe(test_df, use_container_width=True)
        
        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with show_loading("Processing batch prediction..."):
                # Convert DataFrame to CSV string
                csv_content = test_df.to_csv(index=False)
                
                try:
                    files = {"file": ("test_batch.csv", csv_content, "text/csv")}
                    headers = {"X-Session-ID": "test-batch-session"}
                    
                    response = requests.post(
                        "http://localhost:8000/predict/batch",
                        files=files,
                        headers=headers,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        batch_result = response.json()
                        
                        st.markdown("### 📊 Batch Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Records", batch_result["total_records"])
                        
                        with col2:
                            st.metric("Successful", batch_result["successful_predictions"])
                        
                        with col3:
                            st.metric("Failed", batch_result["failed_predictions"])
                        
                        with col4:
                            st.metric("Avg Confidence", f"{batch_result['avg_confidence']:.1%}")
                        
                        # Risk distribution
                        risk_dist = batch_result["risk_distribution"]
                        st.markdown("### 🎯 Risk Distribution")
                        
                        risk_col1, risk_col2, risk_col3 = st.columns(3)
                        
                        with risk_col1:
                            st.metric("🔴 High Risk", risk_dist["high"])
                        
                        with risk_col2:
                            st.metric("🟡 Medium Risk", risk_dist["medium"])
                        
                        with risk_col3:
                            st.metric("🟢 Low Risk", risk_dist["low"])
                        
                        show_success("Batch prediction completed!")
                        
                    else:
                        show_error(f"Batch prediction failed: HTTP {response.status_code}")
                        
                except Exception as e:
                    show_error(f"Batch prediction error: {str(e)}")
    else:
        show_error("Test data file not found")

def main():
    setup_page_config()
    render_header()
    render_sidebar()
    
    st.markdown("# 🧪 System Testing & QA")
    st.markdown("Manual testing interface for validating system components and functionality.")
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 System Status", "🧪 Sample Testing", "🤖 LLM Testing", "📊 Batch Testing"])
    
    with tab1:
        display_system_status()
    
    with tab2:
        display_sample_testing()
    
    with tab3:
        display_llm_testing()
    
    with tab4:
        display_batch_testing()
    
    render_footer()

if __name__ == "__main__":
    main()