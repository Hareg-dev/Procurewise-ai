"""
API Explorer page for testing and exploring all backend endpoints.
"""
import streamlit as st
import requests
import json

API_BASE_URL = "http://localhost:8000"

def show_api_explorer():
    st.markdown("# 🔧 API Explorer")
    st.markdown("*Test and explore all backend endpoints*")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction APIs", "SHAP APIs", "LLM APIs", "History APIs"])
    
    with tab1:
        show_prediction_apis()
    
    with tab2:
        show_shap_apis()
    
    with tab3:
        show_llm_apis()
    
    with tab4:
        show_history_apis()

def show_prediction_apis():
    st.markdown("### 🎯 Prediction Endpoints")
    
    # Health check
    if st.button("🔍 GET /predict/health"):
        try:
            response = requests.get(f"{API_BASE_URL}/predict/health")
            st.json({"status_code": response.status_code, "response": response.json()})
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Single prediction test
    st.markdown("#### POST /predict/single")
    
    with st.expander("Test Single Prediction"):
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
        
        st.json(sample_data)
        
        if st.button("🚀 Test Single Prediction"):
            try:
                response = requests.post(f"{API_BASE_URL}/predict/single", json=sample_data)
                st.json({"status_code": response.status_code, "response": response.json()})
            except Exception as e:
                st.error(f"Error: {str(e)}")

def show_shap_apis():
    st.markdown("### 📊 SHAP Endpoints")
    
    endpoints = [
        ("GET", "/explain/global-importance", "Get global feature importance"),
        ("POST", "/explain/prediction", "Explain single prediction"),
        ("GET", "/explain/summary-plot", "Generate summary plot"),
        ("POST", "/explain/visualize", "Generate visualizations")
    ]
    
    for method, endpoint, description in endpoints:
        if st.button(f"{method} {endpoint}"):
            try:
                if method == "GET":
                    if "summary-plot" in endpoint:
                        response = requests.get(f"{API_BASE_URL}{endpoint}?sample_size=30")
                    else:
                        response = requests.get(f"{API_BASE_URL}{endpoint}")
                else:
                    # POST endpoints need sample data
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
                    response = requests.post(f"{API_BASE_URL}{endpoint}", json=sample_data)
                
                result = {"status_code": response.status_code}
                try:
                    result["response"] = response.json()
                except:
                    result["response"] = response.text[:500] + "..." if len(response.text) > 500 else response.text
                
                st.json(result)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.caption(description)
        st.markdown("---")

def show_llm_apis():
    st.markdown("### 🤖 LLM Endpoints")
    
    # Health check
    if st.button("🔍 GET /llm/health"):
        try:
            response = requests.get(f"{API_BASE_URL}/llm/health")
            st.json({"status_code": response.status_code, "response": response.json()})
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Chat test
    st.markdown("#### POST /llm/chat")
    
    with st.expander("Test Chat"):
        chat_message = st.text_input("Chat message", value="What are the key risk factors in procurement?")
        session_id = st.text_input("Session ID", value=st.session_state.get('session_id', 'test_session'))
        
        if st.button("💬 Send Chat"):
            try:
                chat_data = {
                    "message": chat_message,
                    "session_id": session_id,
                    "include_context": True
                }
                response = requests.post(f"{API_BASE_URL}/llm/chat", json=chat_data)
                st.json({"status_code": response.status_code, "response": response.json()})
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Other LLM endpoints
    llm_endpoints = [
        ("POST", "/llm/explain-prediction", "AI prediction explanation"),
        ("POST", "/llm/recommendations", "Generate recommendations"),
        ("POST", "/llm/what-if", "What-if analysis"),
        ("GET", "/llm/chat-history/{session_id}", "Get chat history"),
        ("DELETE", "/llm/chat-history/{session_id}", "Clear chat history")
    ]
    
    for method, endpoint, description in llm_endpoints:
        if st.button(f"{method} {endpoint}"):
            try:
                session_id = st.session_state.get('session_id', 'test_session')
                
                if method == "GET":
                    url = f"{API_BASE_URL}{endpoint.replace('{session_id}', session_id)}"
                    response = requests.get(url)
                elif method == "DELETE":
                    url = f"{API_BASE_URL}{endpoint.replace('{session_id}', session_id)}"
                    response = requests.delete(url)
                else:
                    # POST endpoints need sample data
                    if "explain-prediction" in endpoint:
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
                    elif "recommendations" in endpoint:
                        sample_data = {
                            "input_data": {"industry": "IT Services", "contract_value": 250000},
                            "prediction_result": {"prediction": "medium", "confidence": 0.75},
                            "shap_explanation": {"top_negative": {"credit_score": -0.1}, "top_positive": {"delivery_score": 0.08}}
                        }
                    elif "what-if" in endpoint:
                        sample_data = {
                            "original_data": {"credit_score": 720, "delivery_score": 85},
                            "modified_data": {"credit_score": 650, "delivery_score": 90},
                            "original_prediction": {"prediction": "medium", "confidence": 0.75}
                        }
                    
                    response = requests.post(f"{API_BASE_URL}{endpoint}", json=sample_data)
                
                st.json({"status_code": response.status_code, "response": response.json()})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.caption(description)
        st.markdown("---")

def show_history_apis():
    st.markdown("### 📜 History Endpoints")
    
    session_id = st.session_state.get('session_id', 'test_session')
    
    # Session creation
    st.markdown("#### POST /history/session")
    
    with st.expander("Create Session"):
        user_id = st.text_input("User ID (optional)", value="test_user")
        
        if st.button("🆕 Create Session"):
            try:
                session_data = {"user_id": user_id if user_id else None}
                response = requests.post(f"{API_BASE_URL}/history/session", json=session_data)
                st.json({"status_code": response.status_code, "response": response.json()})
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Other history endpoints
    history_endpoints = [
        ("GET", "/history/predictions", "Get prediction history"),
        ("GET", "/history/batch-summaries", "Get batch summaries"),
        ("GET", "/history/stats", "Get session statistics")
    ]
    
    for method, endpoint, description in history_endpoints:
        if st.button(f"{method} {endpoint}"):
            try:
                params = {"session_id": session_id}
                if "predictions" in endpoint:
                    params.update({"limit": 10, "offset": 0})
                elif "batch-summaries" in endpoint:
                    params["limit"] = 5
                
                response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
                st.json({"status_code": response.status_code, "response": response.json()})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.caption(description)
        st.markdown("---")