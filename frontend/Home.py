"""
Home page for Procurement Risk Prediction Platform.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout import setup_page_config, render_header, render_sidebar, render_footer
from components.charts import create_metrics_cards
from utils.api_client import APIClient

def main():
    setup_page_config()
    render_header()
    render_sidebar()
    
    # Main content
    st.markdown("## 🏠 Welcome to the Procurement Risk Prediction Platform")
    
    st.markdown("""
    This AI-powered platform helps you assess procurement risks using advanced machine learning 
    and explainable AI techniques. Get instant risk predictions with detailed explanations 
    of the factors driving each assessment.
    """)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Single Prediction
        - Input supplier details manually
        - Get instant risk assessment
        - View SHAP explanations
        - Understand key risk factors
        """)
        if st.button("Start Single Prediction", key="nav_single", use_container_width=True):
            st.session_state.current_page = "Predict_Single"
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 📊 Batch Analysis
        - Upload CSV files
        - Process multiple suppliers
        - Get aggregated insights
        - Download results
        """)
        if st.button("Upload Batch File", key="nav_batch", use_container_width=True):
            st.session_state.current_page = "Predict_Batch"
            st.rerun()
    
    with col3:
        st.markdown("""
        ### 📈 Analytics Dashboard
        - View prediction history
        - Analyze risk trends
        - Interactive visualizations
        - Export reports
        """)
        if st.button("View Analytics", key="nav_analytics", use_container_width=True):
            st.session_state.current_page = "Analysis"
            st.rerun()
    
    st.markdown("---")
    
    # Session statistics
    st.markdown("## 📊 Your Session Overview")
    
    api_client = APIClient()
    stats_result = api_client.get_session_stats()
    
    if stats_result["success"]:
        stats = stats_result["data"]
        create_metrics_cards(stats)
        
        # Recent activity
        if stats.get('total_predictions', 0) > 0:
            st.markdown("### 📋 Recent Activity")
            history_result = api_client.get_prediction_history(limit=5)
            
            if history_result["success"] and history_result["data"]:
                history_df = st.dataframe(
                    history_result["data"],
                    columns=['prediction_type', 'risk_level', 'confidence_score', 'created_at'],
                    use_container_width=True
                )
            else:
                st.info("No recent predictions found")
        else:
            st.info("No predictions made yet. Start with a single prediction or batch upload!")
    else:
        st.warning("Unable to load session statistics. Please check API connection.")
    
    # Platform information
    st.markdown("---")
    st.markdown("## 🔧 Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 AI Technology Stack:**
        - XGBoost machine learning model
        - SHAP explainable AI
        - MLflow experiment tracking
        - FastAPI backend
        - Streamlit frontend
        """)
    
    with col2:
        st.markdown("""
        **📋 Risk Assessment Factors:**
        - Financial stability metrics
        - Performance history
        - Contract terms analysis
        - Relationship duration
        - Industry risk factors
        """)
    
    render_footer()

if __name__ == "__main__":
    main()