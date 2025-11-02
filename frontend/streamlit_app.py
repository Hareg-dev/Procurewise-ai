"""
Comprehensive Streamlit frontend for Procurement Risk Prediction Platform.
Integrates with all backend endpoints for complete functionality.
"""
import streamlit as st
import requests
import pandas as pd
import json
import base64
from io import BytesIO
import time
import uuid

# Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

SESSION_ID = st.session_state.session_id

# Page configuration
st.set_page_config(
    page_title="Procurement Risk Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_api_health():
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def check_llm_health():
    """Check LLM service health."""
    try:
        response = requests.get(f"{API_BASE_URL}/llm/health", timeout=3)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

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
            ["Home", "Single Prediction", "Batch Prediction", "SHAP Analysis", "AI Recommendations", "Chat Assistant", "History & Analytics"]
        )
        
        # System Status
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        
        api_healthy, api_info = check_api_health()
        if api_healthy:
            st.success("🟢 Backend API Online")
        else:
            st.error("🔴 Backend API Offline")
            
        llm_healthy, llm_info = check_llm_health()
        if llm_healthy:
            st.success("🟢 Phi3 LLM Online")
        else:
            st.warning("🟡 Phi3 LLM Offline")
        
        # Session Info
        st.markdown("---")
        st.markdown("### 📋 Session Info")
        st.text(f"Session: {SESSION_ID[:8]}...")
    
    # Route to appropriate page
    if page == "Home":
        show_home_page()
    elif page == "Single Prediction":
        show_single_prediction()
    elif page == "Batch Prediction":
        show_batch_prediction()
    elif page == "SHAP Analysis":
        show_shap_analysis()
    elif page == "AI Recommendations":
        show_ai_recommendations()
    elif page == "Chat Assistant":
        show_chat_assistant()
    elif page == "History & Analytics":
        show_history_analytics()

def show_home_page():
    st.markdown("## 🏠 Welcome to the Procurement Risk Prediction Platform")
    
    st.markdown("""
    This comprehensive AI platform provides end-to-end procurement risk assessment with:
    - **Machine Learning Predictions** using XGBoost
    - **Explainable AI** with SHAP analysis
    - **LLM-powered Recommendations** via Phi3 Mini
    - **Interactive Chat** for procurement insights
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Prediction Services
        - Single supplier risk assessment
        - Batch CSV processing
        - Real-time confidence scoring
        - Historical tracking
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Explainable AI
        - SHAP feature importance
        - Interactive visualizations
        - Prediction comparisons
        - Global model insights
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 AI Assistant
        - Phi3-powered recommendations
        - Conversational interface
        - Context-aware responses
        - What-if scenario analysis
        """)

def show_single_prediction():
    st.markdown("# 🎯 Single Supplier Risk Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Financial Metrics")
            annual_revenue = st.number_input("Annual Revenue ($)", value=1000000, min_value=1000)
            annual_spend = st.number_input("Annual Spend ($)", value=500000, min_value=1000)
            contract_value = st.number_input("Contract Value ($)", value=250000, min_value=1000)
            credit_score = st.number_input("Credit Score", value=720, min_value=300, max_value=850)
            
            st.markdown("#### Performance Metrics")
            delivery_score = st.slider("Delivery Score", 0, 100, 85)
            financial_stability_index = st.slider("Financial Stability Index", 0.0, 1.0, 0.75)
            avg_delivery_quality = st.slider("Avg Delivery Quality", 0, 100, 88)
        
        with col2:
            st.markdown("#### Categorical Data")
            region = st.selectbox("Region", ["North America", "Europe", "Asia-Pacific", "LATAM"])
            industry = st.selectbox("Industry", ["IT Services", "Manufacturing", "Biotech", "Logistics"])
            contract_criticality = st.selectbox("Contract Criticality", ["Low", "Medium", "High"])
            
            st.markdown("#### Relationship & Risk Metrics")
            relationship_years = st.number_input("Relationship Years", value=5, min_value=0)
            past_disputes = st.number_input("Past Disputes", value=2, min_value=0)
            contract_duration_months = st.number_input("Contract Duration (months)", value=12, min_value=1, max_value=120)
            
            st.markdown("#### Transaction Metrics")
            txn_count = st.number_input("Transaction Count", value=10, min_value=0)
            avg_txn_amount = st.number_input("Avg Transaction Amount", value=25000, min_value=0)
            avg_delay = st.number_input("Average Delay (days)", value=12, min_value=0)
            late_ratio = st.slider("Late Payment Ratio", 0.0, 1.0, 0.2)
            dispute_rate = st.slider("Dispute Rate", 0.0, 1.0, 0.1)
            clause_risk_score = st.slider("Clause Risk Score", 0, 100, 25)
        
        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)
        
        if submitted:
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
            
            make_single_prediction(input_data)

def make_single_prediction(input_data):
    """Make single prediction API call and display results."""
    try:
        with st.spinner("Making prediction..."):
            response = requests.post(
                f"{API_BASE_URL}/predict/single",
                json=input_data,
                headers={"X-Session-ID": SESSION_ID},
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            
            st.success("✅ Prediction completed successfully!")
            
            # Main results
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
            
            # Probabilities
            st.markdown("### 📊 Risk Probabilities")
            probabilities = result["probabilities"]
            
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            with prob_col1:
                st.metric("🟢 Low Risk", f"{probabilities.get('low', 0):.1%}")
            
            with prob_col2:
                st.metric("🟡 Medium Risk", f"{probabilities.get('medium', 0):.1%}")
            
            with prob_col3:
                st.metric("🔴 High Risk", f"{probabilities.get('high', 0):.1%}")
            
            # SHAP Values (if available)
            if result.get("shap_values"):
                st.markdown("### 🔍 Feature Importance (SHAP)")
                shap_df = pd.DataFrame(list(result["shap_values"].items()), columns=["Feature", "Impact"])
                shap_df = shap_df.sort_values("Impact", key=abs, ascending=False).head(10)
                st.dataframe(shap_df, use_container_width=True)
            
            # Store result for other pages
            st.session_state.last_prediction = {
                "input_data": input_data,
                "result": result
            }
            
            # Add to prediction history
            st.session_state.prediction_history.append({
                "type": "single",
                "timestamp": time.time(),
                "input_data": input_data,
                "result": result
            })
            
        else:
            st.error(f"❌ Prediction failed: HTTP {response.status_code}")
            st.text(response.text)
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Please ensure the server is running.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def show_batch_prediction():
    st.markdown("# 📊 Batch Risk Prediction")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file:
        # Show file preview
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown("### 📋 File Preview")
            st.dataframe(df.head(), use_container_width=True)
            st.info(f"File contains {len(df)} records with {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        
        if st.button("🚀 Process Batch", use_container_width=True):
            try:
                with st.spinner("Processing batch prediction..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    headers = {"X-Session-ID": SESSION_ID}
                    
                    response = requests.post(
                        f"{API_BASE_URL}/predict/batch",
                        files=files,
                        headers=headers,
                        timeout=120
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Batch processing completed!")
                    
                    # Summary metrics
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
                    
                    # Store batch result for other pages
                    st.session_state.last_prediction = {
                        "input_data": {"batch_file": uploaded_file.name},
                        "result": result,
                        "type": "batch"
                    }
                    
                    # Add to prediction history
                    st.session_state.prediction_history.append({
                        "type": "batch",
                        "timestamp": time.time(),
                        "input_data": {"batch_file": uploaded_file.name},
                        "result": result
                    })
                    
                    # Results table
                    if result.get("results"):
                        st.markdown("### 📋 Detailed Results")
                        results_df = pd.DataFrame(result["results"])
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name=f"batch_predictions_{int(time.time())}.csv",
                            mime="text/csv"
                        )
                        
                else:
                    st.error(f"❌ Batch processing failed: HTTP {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend API.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_shap_analysis():
    st.markdown("# 📈 SHAP Explainability Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Global Importance", "Prediction Explanation", "Visualizations"])
    
    with tab1:
        st.markdown("### 🌍 Global Feature Importance")
        
        sample_size = st.slider("Sample Size", 50, 500, 100)
        
        if st.button("📊 Generate Global Importance"):
            try:
                with st.spinner("Calculating global feature importance..."):
                    response = requests.get(
                        f"{API_BASE_URL}/explain/global-importance",
                        params={"sample_size": sample_size},
                        timeout=60
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        data = result["data"]
                        
                        # Display top features
                        top_features = data.get("top_features", {})
                        if top_features:
                            st.markdown("#### 🔝 Top 10 Most Important Features")
                            importance_df = pd.DataFrame(
                                list(top_features.items()), 
                                columns=["Feature", "Importance"]
                            )
                            st.bar_chart(importance_df.set_index("Feature"))
                            st.dataframe(importance_df, use_container_width=True)
                        
                    else:
                        st.error("Failed to generate global importance")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.markdown("### 🔍 Individual Prediction Explanation")
        
        if st.session_state.last_prediction is not None:
            prediction = st.session_state.last_prediction
            
            # Check if it's batch or single prediction
            is_batch = prediction.get('type') == 'batch'
            
            if is_batch:
                # Show batch info
                result = prediction['result']
                st.info(f"Batch prediction: {result.get('total_records', 0)} records processed")
                st.warning("📊 Batch predictions cannot be explained individually. Please make a single prediction for SHAP analysis.")
                
                # Show batch summary
                if 'risk_distribution' in result:
                    st.markdown("#### 📈 Batch Risk Distribution")
                    risk_dist = result['risk_distribution']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔴 High Risk", risk_dist.get('high', 0))
                    with col2:
                        st.metric("🟡 Medium Risk", risk_dist.get('medium', 0))
                    with col3:
                        st.metric("🟢 Low Risk", risk_dist.get('low', 0))
            else:
                # Single prediction - show explanation
                st.info(f"Current prediction: {prediction['result'].get('prediction', 'N/A')} risk (confidence: {prediction['result'].get('confidence', 0):.1%})")
                
                if st.button("🧠 Explain Last Prediction"):
                    try:
                        with st.spinner("Generating SHAP explanation..."):
                            response = requests.post(
                                f"{API_BASE_URL}/explain/prediction",
                                json=prediction["input_data"],
                                timeout=60
                            )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                data = result["data"]
                                
                                # SHAP explanation
                                shap_exp = data.get("shap_explanation", {})
                                if shap_exp and "top_positive" in shap_exp:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("#### 📈 Risk Increasing Factors")
                                        if shap_exp["top_negative"]:
                                            neg_df = pd.DataFrame(
                                                list(shap_exp["top_negative"].items()),
                                                columns=["Feature", "Impact"]
                                            )
                                            st.dataframe(neg_df, use_container_width=True)
                                        else:
                                            st.info("No significant risk-increasing factors")
                                    
                                    with col2:
                                        st.markdown("#### 📉 Risk Decreasing Factors")
                                        if shap_exp["top_positive"]:
                                            pos_df = pd.DataFrame(
                                                list(shap_exp["top_positive"].items()),
                                                columns=["Feature", "Impact"]
                                            )
                                            st.dataframe(pos_df, use_container_width=True)
                                        else:
                                            st.info("No significant risk-decreasing factors")
                                
                                # Summary
                                summary = data.get("summary", {})
                                if summary:
                                    st.markdown("#### 📝 Explanation Summary")
                                    for key, value in summary.items():
                                        if value and not value.startswith("error"):
                                            st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                            
                            else:
                                st.error("Failed to generate explanation")
                        else:
                            st.error(f"API Error: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("Make a prediction first to see SHAP explanation")
            st.markdown("**Available prediction methods:**")
            st.markdown("- Go to 'Single Prediction' page to analyze one supplier")
            st.markdown("- Go to 'Batch Prediction' page to upload a CSV file")
            st.markdown("- After making predictions, return here for SHAP analysis")
    
    with tab3:
        st.markdown("### 📊 SHAP Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Generate Summary Plot"):
                try:
                    with st.spinner("Generating SHAP summary plot..."):
                        response = requests.get(
                            f"{API_BASE_URL}/explain/summary-plot",
                            params={"sample_size": 50},
                            timeout=60
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            plot_data = result["data"].get("plot")
                            if plot_data:
                                try:
                                    # Decode and display plot
                                    image_data = base64.b64decode(plot_data)
                                    st.image(image_data, caption="SHAP Summary Plot")
                                except Exception as e:
                                    st.error(f"Error displaying plot: {str(e)}")
                            else:
                                st.warning("Plot generation unavailable - using fallback analysis")
                                st.info("SHAP visualization requires model initialization. Global importance is available above.")
                        else:
                            st.error("Failed to generate plot")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.info("Try the Global Importance analysis above instead")
                        
                except requests.exceptions.ConnectionError:
                    st.error("🚫 Cannot connect to Phi3 model. Please ensure the backend is running.")
                except Exception as e:
                    if "503" in str(e):
                        st.error("🚫 Phi3 model is currently unavailable. Please check Ollama service.")
                    else:
                        st.error(f"Phi3 Error: {str(e)}")
        
        with col2:
            can_generate_force_plot = (st.session_state.last_prediction is not None and 
                                     st.session_state.last_prediction.get('type') != 'batch')
            
            if st.button("🔍 Generate Force Plot", disabled=not can_generate_force_plot):
                if can_generate_force_plot:
                    try:
                        prediction = st.session_state.last_prediction
                        with st.spinner("Generating SHAP force plot..."):
                            response = requests.post(
                                f"{API_BASE_URL}/explain/visualize",
                                json=prediction["input_data"],
                                timeout=60
                            )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                force_plot = result["data"].get("force_plot")
                                if force_plot:
                                    try:
                                        # Decode and display plot
                                        image_data = base64.b64decode(force_plot)
                                        st.image(image_data, caption="SHAP Force Plot")
                                    except Exception as e:
                                        st.error(f"Error displaying plot: {str(e)}")
                                else:
                                    st.warning("Force plot generation unavailable - using fallback analysis")
                                    st.info("SHAP visualization requires model initialization. Use 'Explain Last Prediction' above.")
                            else:
                                st.error("Failed to generate force plot")
                        else:
                            st.error(f"API Error: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            if not can_generate_force_plot:
                if st.session_state.last_prediction is None:
                    st.info("Make a single prediction first to generate force plot")
                else:
                    st.info("Force plots are only available for single predictions, not batch predictions")

def show_ai_recommendations():
    st.markdown("# 💡 AI-Powered Recommendations")
    st.markdown("*Powered by Phi3 Mini - Real AI Analysis*")
    
    if st.session_state.last_prediction is not None:
        prediction = st.session_state.last_prediction
        
        # Show current prediction info
        st.info(f"Current prediction: {prediction['result'].get('prediction', 'N/A')} risk (confidence: {prediction['result'].get('confidence', 0):.1%})")
        
        if st.button("🤖 Generate AI Recommendations"):
            try:
                # First get SHAP explanation
                with st.spinner("Analyzing prediction..."):
                    shap_response = requests.post(
                        f"{API_BASE_URL}/explain/prediction",
                        json=prediction["input_data"],
                        timeout=30
                    )
                
                shap_explanation = {}
                if shap_response.status_code == 200:
                    shap_result = shap_response.json()
                    if shap_result.get("success"):
                        shap_explanation = shap_result["data"]["shap_explanation"]
                
                # Generate recommendations
                with st.spinner("Generating AI recommendations..."):
                    rec_data = {
                        "prediction_result": prediction["result"],
                        "shap_explanation": shap_explanation,
                        "input_data": prediction["input_data"]
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/llm/recommendations",
                        json=rec_data,
                        timeout=60
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        recommendations = result["data"]
                        
                        st.success("✅ Phi3 AI Recommendations Generated!")
                        
                        # Show model info
                        if recommendations.get("model"):
                            st.info(f"🤖 Generated by: {recommendations['model']} | Confidence: {recommendations.get('confidence', 0):.1%}")
                        
                        # Display recommendations
                        st.markdown("### 🎯 Recommended Actions")
                        
                        if recommendations.get("recommendations"):
                            for i, rec in enumerate(recommendations["recommendations"], 1):
                                st.markdown(f"**{i}.** {rec}")
                        
                        # Full response
                        if recommendations.get("full_response"):
                            st.markdown("### 🤖 Detailed Phi3 Analysis")
                            st.markdown(recommendations["full_response"])
                        
                    else:
                        st.error("Failed to generate recommendations")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("💡 Make a prediction first to get personalized Phi3 recommendations")
        st.markdown("**Available prediction methods:**")
        st.markdown("- Go to 'Single Prediction' page to analyze one supplier")
        st.markdown("- Go to 'Batch Prediction' page to upload a CSV file")
        st.markdown("- After making predictions, return here for real AI recommendations from Phi3")
        
        # Demo recommendations
        if st.button("🎯 Test Phi3 Recommendations"):
            demo_data = {
                "prediction_result": {"prediction": "high", "confidence": 0.85},
                "shap_explanation": {"top_negative": {"payment_delay": -0.3, "delivery_score": -0.2}},
                "input_data": {"industry": "Manufacturing", "contract_value": 500000}
            }
            
            try:
                with st.spinner("Generating demo recommendations..."):
                    response = requests.post(
                        f"{API_BASE_URL}/llm/recommendations",
                        json=demo_data,
                        timeout=30
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        st.success("✅ Phi3 Demo Recommendations:")
                        st.markdown(result["data"].get("full_response", "No recommendations available"))
                elif response.status_code == 503:
                    st.error("🚫 Phi3 model is currently unavailable. Please ensure Ollama is running with 'ollama run phi3:mini'")
                else:
                    st.error(f"API Error: {response.status_code}")
                
            except Exception as e:
                st.error(f"Error connecting to Phi3: {str(e)}")

def show_chat_assistant():
    st.markdown("# 💬 AI Chat Assistant")
    st.markdown("*Powered by Phi3 Mini - Real AI Conversations*")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Phi3-powered procurement assistant. I can provide real AI analysis of your supplier risks and recommendations. How can I help you today?"}
        ]
    
    # Show context if available
    if st.session_state.last_prediction is not None:
        prediction = st.session_state.last_prediction
        st.info(f"💡 Phi3 Context: Last prediction was {prediction['result'].get('prediction', 'N/A')} risk with {prediction['result'].get('confidence', 0):.1%} confidence - Ask me about it!")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about procurement risks..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            try:
                with st.spinner("🤖 Phi3 is thinking..."):
                    # Include prediction context if available
                    include_context = st.session_state.last_prediction is not None
                    
                    chat_data = {
                        "message": prompt,
                        "session_id": SESSION_ID,
                        "include_context": include_context
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/llm/chat",
                        json=chat_data,
                        timeout=60
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        ai_response = result["data"]["response"]
                        # Show model info
                        if result["data"].get("model"):
                            ai_response += f"\n\n*Response generated by {result['data']['model']}*"
                    else:
                        ai_response = "🚫 Phi3 is having trouble processing your request. Please try again."
                elif response.status_code == 503:
                    ai_response = "🚫 Phi3 model is currently unavailable. Please ensure Ollama is running with 'ollama run phi3:mini'."
                else:
                    ai_response = f"Phi3 service error (HTTP {response.status_code}). Please check if the backend is running."
                    
            except requests.exceptions.ConnectionError:
                ai_response = "🚫 Can't connect to Phi3 backend. Please ensure the FastAPI server is running on localhost:8000."
            except Exception as e:
                if "503" in str(e):
                    ai_response = "🚫 Phi3 model is currently unavailable. Please check Ollama service."
                else:
                    ai_response = f"Phi3 connection error: {str(e)}"
            
            st.write(ai_response)
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

def show_history_analytics():
    st.markdown("# 📊 History & Analytics")
    
    tab1, tab2 = st.tabs(["Session Statistics", "Prediction History"])
    
    with tab1:
        st.markdown("### 📈 Session Statistics")
        
        # Show current session info
        st.info(f"Session ID: {SESSION_ID[:8]}... | Predictions in memory: {len(st.session_state.prediction_history)}")
        
        if st.button("📊 Get Session Stats"):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/history/stats",
                    params={"session_id": SESSION_ID},
                    timeout=30
                )
                
                if response.status_code == 200:
                    stats = response.json()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Predictions", stats["total_predictions"])
                    
                    with col2:
                        st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
                    
                    with col3:
                        st.metric("Single Predictions", stats["prediction_types"]["single"])
                    
                    with col4:
                        st.metric("Batch Predictions", stats["prediction_types"]["batch"])
                    
                    # Risk distribution
                    if stats["risk_distribution"]:
                        st.markdown("#### Risk Distribution")
                        risk_df = pd.DataFrame(
                            list(stats["risk_distribution"].items()),
                            columns=["Risk Level", "Count"]
                        )
                        st.bar_chart(risk_df.set_index("Risk Level"))
                
                else:
                    st.error(f"Failed to get stats: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.markdown("### 📋 Prediction History")
        
        limit = st.slider("Number of records", 10, 100, 20)
        
        # Show local history first
        if st.session_state.prediction_history:
            st.markdown("#### 📋 Local Session History")
            local_history = []
            for i, pred in enumerate(st.session_state.prediction_history[-limit:]):
                local_history.append({
                    "Index": i+1,
                    "Type": pred["type"],
                    "Risk": pred["result"].get("prediction", "N/A"),
                    "Confidence": f"{pred['result'].get('confidence', 0):.1%}",
                    "Time": pd.to_datetime(pred["timestamp"], unit='s').strftime('%H:%M:%S')
                })
            
            if local_history:
                st.dataframe(pd.DataFrame(local_history), use_container_width=True)
        
        if st.button("📜 Get Database History"):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/history/predictions",
                    params={"session_id": SESSION_ID, "limit": limit},
                    timeout=30
                )
                
                if response.status_code == 200:
                    history = response.json()
                    
                    if history:
                        history_df = pd.DataFrame(history)
                        st.dataframe(history_df, use_container_width=True)
                    else:
                        st.info("No prediction history found for this session")
                
                else:
                    st.error(f"Failed to get history: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()