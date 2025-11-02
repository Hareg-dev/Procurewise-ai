"""
Advanced LLM features page for enhanced AI interactions.
"""
import streamlit as st
import requests
import json

API_BASE_URL = "http://localhost:8000"

def show_llm_advanced():
    st.markdown("# 🤖 Advanced AI Features")
    
    session_id = st.session_state.get('session_id', 'default')
    
    tab1, tab2, tab3, tab4 = st.tabs(["What-If Analysis", "Prediction Explanation", "Chat History", "LLM Recommendations"])
    
    with tab1:
        show_what_if_analysis()
    
    with tab2:
        show_prediction_explanation()
    
    with tab3:
        show_chat_history(session_id)
    
    with tab4:
        show_llm_recommendations()

def show_what_if_analysis():
    st.markdown("### 🔮 What-If Scenario Analysis")
    
    if st.session_state.get('last_prediction'):
        prediction = st.session_state.last_prediction
        
        st.info("Modify values below to see how changes would affect the risk assessment:")
        
        # Create form with current values
        with st.form("what_if_form"):
            col1, col2 = st.columns(2)
            
            original_data = prediction['input_data']
            modified_data = {}
            
            with col1:
                modified_data['credit_score'] = st.number_input("Credit Score", 
                    value=original_data.get('credit_score', 720), min_value=300, max_value=850)
                modified_data['delivery_score'] = st.slider("Delivery Score", 
                    0, 100, original_data.get('delivery_score', 85))
                modified_data['contract_value'] = st.number_input("Contract Value", 
                    value=original_data.get('contract_value', 250000))
            
            with col2:
                modified_data['past_disputes'] = st.number_input("Past Disputes", 
                    value=original_data.get('past_disputes', 2), min_value=0)
                modified_data['avg_payment_delay_days'] = st.number_input("Payment Delay Days", 
                    value=original_data.get('avg_payment_delay_days', 12), min_value=0)
                modified_data['financial_stability_index'] = st.slider("Financial Stability", 
                    0.0, 1.0, original_data.get('financial_stability_index', 0.75))
            
            # Copy unchanged values
            for key, value in original_data.items():
                if key not in modified_data:
                    modified_data[key] = value
            
            if st.form_submit_button("🔍 Analyze What-If Scenario"):
                try:
                    what_if_data = {
                        "original_data": original_data,
                        "modified_data": modified_data,
                        "original_prediction": prediction['result']
                    }
                    
                    with st.spinner("Analyzing scenario..."):
                        response = requests.post(f"{API_BASE_URL}/llm/what-if", json=what_if_data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success("✅ What-If Analysis Complete")
                            st.markdown("### 📊 Analysis Results")
                            st.markdown(result["data"]["analysis"])
                            
                            changes = result["data"]["changes_detected"]
                            st.info(f"🔄 {changes} parameter(s) modified")
                        else:
                            st.error("Analysis failed")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("Make a prediction first to use What-If analysis")

def show_prediction_explanation():
    st.markdown("### 🧠 AI Prediction Explanation")
    
    if st.session_state.get('last_prediction'):
        prediction = st.session_state.last_prediction
        
        if st.button("🤖 Get AI Explanation"):
            try:
                with st.spinner("Generating AI explanation..."):
                    response = requests.post(f"{API_BASE_URL}/llm/explain-prediction", 
                                           json=prediction['input_data'])
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        data = result["data"]
                        
                        st.success("✅ AI Explanation Generated")
                        
                        # Show prediction summary
                        pred_result = data["prediction"]
                        st.markdown(f"**Risk Level:** {pred_result['prediction'].upper()} ({pred_result['confidence']:.1%} confidence)")
                        
                        # Show AI explanation
                        st.markdown("### 🤖 AI Analysis")
                        st.markdown(data["llm_explanation"])
                        
                        # Show model info
                        st.info(f"Generated by: {data['model']}")
                    else:
                        st.error("Explanation generation failed")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("Make a prediction first to get AI explanation")

def show_chat_history(session_id):
    st.markdown("### 💬 Chat History Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📜 Get Chat History"):
            try:
                response = requests.get(f"{API_BASE_URL}/llm/chat-history/{session_id}")
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        history = result["data"]["history"]
                        
                        if history:
                            st.success(f"✅ Found {len(history)} chat messages")
                            
                            for i, exchange in enumerate(history[-5:], 1):  # Show last 5
                                with st.expander(f"Exchange {i} - {exchange.get('timestamp', 'Unknown time')}"):
                                    st.markdown(f"**User:** {exchange['user']}")
                                    st.markdown(f"**Assistant:** {exchange['assistant']}")
                        else:
                            st.info("No chat history found")
                    else:
                        st.error("Failed to get chat history")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear Chat History"):
            try:
                response = requests.delete(f"{API_BASE_URL}/llm/chat-history/{session_id}")
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        st.success("✅ Chat history cleared")
                        st.info(result["data"]["message"])
                    else:
                        st.error("Failed to clear chat history")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

def show_llm_recommendations():
    st.markdown("### 💡 Enhanced LLM Recommendations")
    
    if st.session_state.get('last_prediction'):
        prediction = st.session_state.last_prediction
        
        if st.button("🚀 Generate Enhanced Recommendations"):
            try:
                # Get SHAP explanation first
                shap_response = requests.post(f"{API_BASE_URL}/explain/prediction", 
                                            json=prediction['input_data'])
                
                shap_explanation = {}
                if shap_response.status_code == 200:
                    shap_result = shap_response.json()
                    if shap_result.get("success"):
                        shap_explanation = shap_result["data"]["shap_explanation"]
                
                # Generate recommendations
                rec_data = {
                    "input_data": prediction['input_data'],
                    "prediction_result": prediction['result'],
                    "shap_explanation": shap_explanation
                }
                
                with st.spinner("Generating enhanced recommendations..."):
                    response = requests.post(f"{API_BASE_URL}/llm/recommendations", json=rec_data)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        recommendations = result["data"]
                        
                        st.success("✅ Enhanced Recommendations Generated")
                        
                        # Show structured recommendations
                        if recommendations.get("recommendations"):
                            st.markdown("### 🎯 Action Items")
                            for i, rec in enumerate(recommendations["recommendations"], 1):
                                st.markdown(f"**{i}.** {rec}")
                        
                        # Show full analysis
                        st.markdown("### 🤖 Detailed Analysis")
                        st.markdown(recommendations.get("full_response", "No detailed analysis available"))
                        
                        # Show metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Level", recommendations.get("risk_level", "Unknown").title())
                        with col2:
                            st.metric("Confidence", f"{recommendations.get('confidence', 0):.1%}")
                        with col3:
                            st.info(f"Model: {recommendations.get('model', 'Unknown')}")
                    else:
                        st.error("Recommendation generation failed")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("Make a prediction first to get enhanced recommendations")