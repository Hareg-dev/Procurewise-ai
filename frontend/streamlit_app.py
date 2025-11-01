"""
Simple Streamlit app launcher for the Procurement Risk Prediction Platform.
"""
import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Procurement Risk Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple page routing
def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79 0%, #2e7d9a 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; text-align: center;">
            🔍 Procurement Risk Prediction Platform
        </h1>
        <p style="color: #e0e0e0; margin: 0.5rem 0 0 0; text-align: center;">
            AI-Powered Risk Assessment with Explainable Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Home", "Single Prediction", "Batch Prediction", "Analysis", "Recommendations", "Chat"]
        )
        
        # API Status
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        
        try:
            import requests
            health_response = requests.get("http://localhost:8000/health", timeout=3)
            if health_response.status_code == 200:
                st.success("🟢 Backend Online")
            else:
                st.error("🔴 Backend Error")
        except:
            st.error("🔴 Backend Offline")
            
        try:
            llm_response = requests.get("http://localhost:8000/llm/health", timeout=3)
            if llm_response.status_code == 200 and llm_response.json().get("success"):
                st.success("🟢 Phi3 Online")
            else:
                st.warning("🟡 Phi3 Offline")
        except:
            st.warning("🟡 LLM Offline")
    
    # Page content based on selection
    if page == "Home":
        show_home_page()
    elif page == "Single Prediction":
        show_single_prediction()
    elif page == "Batch Prediction":
        show_batch_prediction()
    elif page == "Analysis":
        show_analysis()
    elif page == "Recommendations":
        show_recommendations()
    elif page == "Chat":
        show_chat()

def show_home_page():
    st.markdown("## 🏠 Welcome to the Procurement Risk Prediction Platform")
    
    st.markdown("""
    This AI-powered platform helps you assess procurement risks using advanced machine learning 
    and explainable AI techniques. Get instant risk predictions with detailed explanations 
    of the factors driving each assessment.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Single Prediction
        - Input supplier details manually
        - Get instant risk assessment
        - View SHAP explanations
        - Understand key risk factors
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Batch Analysis
        - Upload CSV files
        - Process multiple suppliers
        - Get aggregated insights
        - Download results
        """)
    
    with col3:
        st.markdown("""
        ### 📈 Analytics Dashboard
        - View prediction history
        - Analyze risk trends
        - Interactive visualizations
        - Export reports
        """)

def show_single_prediction():
    st.markdown("# 🎯 Single Supplier Risk Prediction")
    st.info("⚠️ Backend API connection required for predictions. Please ensure the FastAPI server is running on localhost:8000")
    
    # Simple form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            annual_revenue = st.number_input("Annual Revenue ($)", value=1000000)
            annual_spend = st.number_input("Annual Spend ($)", value=500000)
            contract_value = st.number_input("Contract Value ($)", value=250000)
            credit_score = st.number_input("Credit Score", value=720)
            delivery_score = st.slider("Delivery Score", 0, 100, 85)
        
        with col2:
            region = st.selectbox("Region", ["North America", "Europe", "Asia-Pacific", "LATAM"])
            industry = st.selectbox("Industry", ["IT Services", "Manufacturing", "Biotech", "Logistics"])
            contract_criticality = st.selectbox("Contract Criticality", ["Low", "Medium", "High"])
            relationship_years = st.number_input("Relationship Years", value=5)
            past_disputes = st.number_input("Past Disputes", value=2)
        
        # Additional required fields
        financial_stability_index = st.slider("Financial Stability Index", 0.0, 1.0, 0.75)
        txn_count = st.number_input("Transaction Count", value=10)
        avg_txn_amount = st.number_input("Avg Transaction Amount", value=25000)
        avg_delay = st.number_input("Average Delay (days)", value=12)
        late_ratio = st.slider("Late Ratio", 0.0, 1.0, 0.2)
        dispute_rate = st.slider("Dispute Rate", 0.0, 1.0, 0.1)
        avg_delivery_quality = st.slider("Avg Delivery Quality", 0, 100, 88)
        clause_risk_score = st.slider("Clause Risk Score", 0, 100, 25)
        contract_duration_months = st.number_input("Contract Duration (months)", value=12)
        
        submitted = st.form_submit_button("🔍 Predict Risk")
        
        if submitted:
            # Prepare input data
            input_data = {
                "annual_revenue": annual_revenue,
                "annual_spend": annual_spend,
                "avg_payment_delay_days": avg_delay,
                "contract_value": contract_value,
                "contract_duration_months": contract_duration_months,
                "past_disputes": past_disputes,
                "delivery_score": delivery_score,
                "financial_stability_index": financial_stability_index,
                "relationship_years": relationship_years,
                "credit_score": credit_score,
                "txn_count": txn_count,
                "avg_txn_amount": avg_txn_amount,
                "avg_delay": avg_delay,
                "late_ratio": late_ratio,
                "dispute_rate": dispute_rate,
                "avg_delivery_quality": avg_delivery_quality,
                "clause_risk_score": clause_risk_score,
                "region": region,
                "industry": industry,
                "contract_criticality": contract_criticality
            }
            
            # Make API call
            try:
                import requests
                with st.spinner("Making prediction..."):
                    response = requests.post(
                        "http://localhost:8000/predict/single",
                        json=input_data,
                        headers={"X-Session-ID": "streamlit-session"},
                        timeout=30
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Prediction completed successfully!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        risk_level = result["prediction"]
                        risk_icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                        st.metric("Risk Level", f"{risk_icons.get(risk_level, '⚪')} {risk_level.upper()}")
                    
                    with col2:
                        confidence = result["confidence"]
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col3:
                        processing_time = result.get("processing_time", 0)
                        st.metric("Processing Time", f"{processing_time:.3f}s")
                    
                    # Show probabilities
                    st.markdown("### 📊 Risk Probabilities")
                    probabilities = result["probabilities"]
                    
                    prob_col1, prob_col2, prob_col3 = st.columns(3)
                    
                    with prob_col1:
                        st.metric("🟢 Low Risk", f"{probabilities.get('low', 0):.1%}")
                    
                    with prob_col2:
                        st.metric("🟡 Medium Risk", f"{probabilities.get('medium', 0):.1%}")
                    
                    with prob_col3:
                        st.metric("🔴 High Risk", f"{probabilities.get('high', 0):.1%}")
                        
                else:
                    st.error(f"❌ Prediction failed: HTTP {response.status_code}")
                    st.text(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend API. Please ensure the FastAPI server is running on localhost:8000")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_batch_prediction():
    st.markdown("# 📊 Batch Risk Prediction")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file:
        if st.button("🚀 Process Batch"):
            try:
                import requests
                with st.spinner("Processing batch prediction..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    headers = {"X-Session-ID": "streamlit-batch-session"}
                    
                    response = requests.post(
                        "http://localhost:8000/predict/batch",
                        files=files,
                        headers=headers,
                        timeout=60
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Batch processing completed!")
                    
                    # Display summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", result["total_records"])
                    
                    with col2:
                        st.metric("Successful", result["successful_predictions"])
                    
                    with col3:
                        st.metric("Failed", result["failed_predictions"])
                    
                    with col4:
                        st.metric("Avg Confidence", f"{result['avg_confidence']:.1%}")
                    
                    # Risk distribution
                    st.markdown("### 🎯 Risk Distribution")
                    risk_dist = result["risk_distribution"]
                    
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    
                    with risk_col1:
                        st.metric("🔴 High Risk", risk_dist["high"])
                    
                    with risk_col2:
                        st.metric("🟡 Medium Risk", risk_dist["medium"])
                    
                    with risk_col3:
                        st.metric("🟢 Low Risk", risk_dist["low"])
                        
                else:
                    st.error(f"❌ Batch processing failed: HTTP {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend API. Please ensure the FastAPI server is running on localhost:8000")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_analysis():
    st.markdown("# 📈 Analytics Dashboard")
    st.info("⚠️ Backend API connection required for analytics data.")
    
    # Mock metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", "0")
    with col2:
        st.metric("Avg Confidence", "0%")
    with col3:
        st.metric("High Risk Count", "0")
    with col4:
        st.metric("Batch Predictions", "0")

def show_recommendations():
    st.markdown("# 💡 AI Recommendations")
    st.markdown("*Powered by Phi3 Mini*")
    
    st.markdown("Get AI-powered procurement recommendations based on your risk predictions.")
    
    # Sample data for demonstration
    if st.button("🎯 Generate Sample Recommendations"):
        sample_data = {
            "prediction_result": {"prediction": "high", "confidence": 0.85},
            "shap_explanation": {"top_negative": {"payment_delay": -0.3, "delivery_score": -0.2}},
            "input_data": {"industry": "Manufacturing", "contract_value": 500000}
        }
        
        try:
            import requests
            with st.spinner("Generating AI recommendations..."):
                response = requests.post(
                    "http://localhost:8000/llm/recommendations",
                    json=sample_data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    recommendations = result["data"]
                    
                    st.success("✅ Recommendations generated!")
                    st.markdown("### 🤖 AI Recommendations:")
                    st.markdown(recommendations.get("full_response", "No recommendations available"))
                else:
                    st.error("Failed to generate recommendations")
            else:
                st.error(f"API Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.info("💡 **Tip**: Make a prediction first, then come here for personalized recommendations based on your specific supplier data.")

def show_chat():
    st.markdown("# 💬 AI Chat Assistant")
    st.markdown("*Powered by Phi3 Mini*")
    
    # Simple chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI procurement assistant powered by Phi3. How can I help you today?"}
        ]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask me about procurement risks..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            try:
                import requests
                with st.spinner("🤖 Thinking..."):
                    chat_data = {
                        "message": prompt,
                        "session_id": "streamlit-chat-session",
                        "include_context": True
                    }
                    
                    response = requests.post(
                        "http://localhost:8000/llm/chat",
                        json=chat_data,
                        timeout=30
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        ai_response = result["data"]["response"]
                    else:
                        ai_response = f"I'm having trouble processing your request. Error: {result.get('error', 'Unknown error')}"
                else:
                    ai_response = "I'm currently unable to process your request. Please check if the backend is running."
                    
            except requests.exceptions.ConnectionError:
                ai_response = "I can't connect to my AI backend. Please ensure the FastAPI server is running."
            except Exception as e:
                ai_response = f"I encountered an error: {str(e)}"
            
            st.write(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

if __name__ == "__main__":
    main()