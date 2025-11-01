"""
Single prediction page for individual supplier risk assessment.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout import setup_page_config, render_header, render_sidebar, render_footer, show_loading, show_success, show_error
from components.shap_visuals import render_shap_explanation, render_shap_interpretation
from utils.api_client import APIClient

def create_input_form():
    """Create input form for single prediction."""
    st.markdown("## 📝 Supplier Information")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 Financial Information")
            annual_revenue = st.number_input("Annual Revenue ($)", min_value=1000, value=1000000, step=10000)
            annual_spend = st.number_input("Annual Spend ($)", min_value=1000, value=500000, step=10000)
            contract_value = st.number_input("Contract Value ($)", min_value=1000, value=250000, step=10000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720, step=10)
            financial_stability_index = st.slider("Financial Stability Index", 0.0, 1.0, 0.75, 0.01)
            
            st.markdown("### 📊 Performance Metrics")
            delivery_score = st.slider("Delivery Score", 0, 100, 85, 1)
            avg_payment_delay_days = st.number_input("Avg Payment Delay (days)", min_value=0, value=15, step=1)
            avg_delivery_quality = st.slider("Avg Delivery Quality", 0, 100, 88, 1)
            
        with col2:
            st.markdown("### 🤝 Relationship & Contract")
            relationship_years = st.number_input("Relationship Years", min_value=0, value=5, step=1)
            contract_duration_months = st.number_input("Contract Duration (months)", min_value=1, max_value=120, value=12, step=1)
            past_disputes = st.number_input("Past Disputes", min_value=0, value=2, step=1)
            clause_risk_score = st.slider("Contract Clause Risk Score", 0, 100, 25, 1)
            
            st.markdown("### 📈 Transaction Data")
            txn_count = st.number_input("Transaction Count", min_value=0, value=10, step=1)
            avg_txn_amount = st.number_input("Avg Transaction Amount ($)", min_value=0, value=25000, step=1000)
            avg_delay = st.number_input("Avg Delay (days)", min_value=0, value=12, step=1)
            late_ratio = st.slider("Late Payment Ratio", 0.0, 1.0, 0.2, 0.01)
            dispute_rate = st.slider("Dispute Rate", 0.0, 1.0, 0.1, 0.01)
        
        # Categorical inputs
        st.markdown("### 🌍 Classification")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            region = st.selectbox("Region", ["North America", "Europe", "Asia-Pacific", "LATAM"])
        
        with col4:
            industry = st.selectbox("Industry", ["IT Services", "Manufacturing", "Biotech", "Logistics", "Facilities"])
        
        with col5:
            contract_criticality = st.selectbox("Contract Criticality", ["Low", "Medium", "High"])
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                "annual_revenue": annual_revenue,
                "annual_spend": annual_spend,
                "avg_payment_delay_days": avg_payment_delay_days,
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
            
            return input_data
    
    return None

def display_prediction_results(result_data):
    """Display prediction results with SHAP explanation."""
    st.markdown("## 🎯 Prediction Results")
    
    prediction = result_data["prediction"]
    confidence = result_data["confidence"]
    probabilities = result_data["probabilities"]
    shap_values = result_data.get("shap_values", {})
    
    # Main result display
    col1, col2, col3 = st.columns(3)
    
    risk_colors = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🔴'
    }
    
    with col1:
        risk_icon = risk_colors.get(prediction.lower(), '⚪')
        st.metric(
            label="Risk Level",
            value=f"{risk_icon} {prediction.upper()}"
        )
    
    with col2:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.1%}"
        )
    
    with col3:
        processing_time = result_data.get("processing_time", 0)
        st.metric(
            label="Processing Time",
            value=f"{processing_time:.3f}s"
        )
    
    # Probability breakdown
    st.markdown("### 📊 Risk Probabilities")
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    
    with prob_col1:
        low_prob = probabilities.get('low', 0)
        st.metric("🟢 Low Risk", f"{low_prob:.1%}")
    
    with prob_col2:
        medium_prob = probabilities.get('medium', 0)
        st.metric("🟡 Medium Risk", f"{medium_prob:.1%}")
    
    with prob_col3:
        high_prob = probabilities.get('high', 0)
        st.metric("🔴 High Risk", f"{high_prob:.1%}")
    
    # SHAP explanation
    if shap_values:
        render_shap_explanation(shap_values, prediction, confidence)
        render_shap_interpretation(prediction, confidence, shap_values)
    else:
        st.warning("SHAP explanation not available for this prediction")

def main():
    setup_page_config()
    render_header()
    render_sidebar()
    
    st.markdown("# 🎯 Single Supplier Risk Prediction")
    st.markdown("Enter supplier details below to get an instant risk assessment with AI explanations.")
    
    # Input form
    input_data = create_input_form()
    
    # Process prediction if form submitted
    if input_data:
        api_client = APIClient()
        
        with show_loading("Analyzing supplier risk..."):
            result = api_client.predict_single(input_data)
        
        if result["success"]:
            show_success("Risk prediction completed successfully!")
            display_prediction_results(result["data"])
            
            # Store in session state for analysis page
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            st.session_state.prediction_history.append(result["data"])
            
        else:
            show_error(f"Prediction failed: {result['error']}")
            st.markdown("**Troubleshooting:**")
            st.markdown("- Check if the API server is running")
            st.markdown("- Verify all input values are valid")
            st.markdown("- Try again in a few moments")
    
    # Sample data button
    st.markdown("---")
    if st.button("📋 Load Sample Data", help="Fill form with sample supplier data"):
        st.info("Sample data loaded! Scroll up and click 'Predict Risk' to see results.")
        st.rerun()
    
    render_footer()

if __name__ == "__main__":
    main()