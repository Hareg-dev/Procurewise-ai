"""
Explainability page for SHAP visualizations and model interpretations.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout import setup_page_config, render_header, render_sidebar, render_footer, show_loading, show_success, show_error, show_info
from utils.api_client import APIClient

def display_global_importance():
    """Display global feature importance analysis."""
    st.markdown("## 🌍 Global Feature Importance")
    st.markdown("Understanding which features are most important across all predictions.")
    
    api_client = APIClient()
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.slider("Sample Size for Analysis", 50, 500, 100, 50)
    
    with col2:
        if st.button("🔄 Generate Global Analysis", use_container_width=True):
            with show_loading("Analyzing global feature importance..."):
                try:
                    response = api_client._make_request("GET", "/explain/global-importance", params={"sample_size": sample_size})
                    
                    if response["success"]:
                        data = response["data"]
                        global_importance = data.get("global_importance", {})
                        
                        if global_importance:
                            # Display top features
                            st.markdown("### 📊 Top 10 Most Important Features")
                            
                            top_features = list(global_importance.items())[:10]
                            features, importance = zip(*top_features)
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.barh(range(len(features)), importance, color='steelblue', alpha=0.7)
                            ax.set_yticks(range(len(features)))
                            ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
                            ax.set_xlabel('Average SHAP Importance')
                            ax.set_title('Global Feature Importance (SHAP)')
                            
                            # Add value labels
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                                       f'{importance[i]:.3f}', ha='left', va='center')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Feature insights
                            st.markdown("### 🔍 Key Insights")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Most Critical Factors:**")
                                for i, (feature, score) in enumerate(top_features[:5]):
                                    st.markdown(f"{i+1}. **{feature.replace('_', ' ').title()}** ({score:.3f})")
                            
                            with col2:
                                st.markdown("**Analysis Summary:**")
                                st.markdown(f"- **Sample Size:** {data.get('sample_size', 0)} predictions")
                                st.markdown(f"- **Top Feature:** {top_features[0][0].replace('_', ' ').title()}")
                                st.markdown(f"- **Importance Range:** {min(importance):.3f} - {max(importance):.3f}")
                        else:
                            show_error("No global importance data available")
                    else:
                        show_error("Failed to generate global analysis")
                        
                except Exception as e:
                    show_error(f"Error: {str(e)}")

def display_prediction_explanation():
    """Display SHAP explanation for a specific prediction."""
    st.markdown("## 🎯 Prediction Explanation")
    st.markdown("Get detailed SHAP analysis for a specific supplier prediction.")
    
    # Input form for prediction
    with st.form("explanation_form"):
        st.markdown("### 📝 Supplier Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            annual_revenue = st.number_input("Annual Revenue ($)", value=1000000)
            annual_spend = st.number_input("Annual Spend ($)", value=500000)
            contract_value = st.number_input("Contract Value ($)", value=250000)
            credit_score = st.number_input("Credit Score", value=720)
            delivery_score = st.slider("Delivery Score", 0, 100, 85)
            relationship_years = st.number_input("Relationship Years", value=5)
        
        with col2:
            region = st.selectbox("Region", ["North America", "Europe", "Asia-Pacific", "LATAM"])
            industry = st.selectbox("Industry", ["IT Services", "Manufacturing", "Biotech", "Logistics", "Facilities"])
            contract_criticality = st.selectbox("Contract Criticality", ["Low", "Medium", "High"])
            past_disputes = st.number_input("Past Disputes", value=2)
            avg_payment_delay_days = st.number_input("Avg Payment Delay (days)", value=15)
            financial_stability_index = st.slider("Financial Stability Index", 0.0, 1.0, 0.75)
        
        # Additional fields (simplified)
        txn_count = st.number_input("Transaction Count", value=10)
        avg_txn_amount = st.number_input("Avg Transaction Amount", value=25000)
        avg_delay = st.number_input("Average Delay (days)", value=12)
        late_ratio = st.slider("Late Ratio", 0.0, 1.0, 0.2)
        dispute_rate = st.slider("Dispute Rate", 0.0, 1.0, 0.1)
        avg_delivery_quality = st.slider("Avg Delivery Quality", 0, 100, 88)
        clause_risk_score = st.slider("Clause Risk Score", 0, 100, 25)
        contract_duration_months = st.number_input("Contract Duration (months)", value=12)
        
        submitted = st.form_submit_button("🔍 Explain Prediction", use_container_width=True)
        
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
            
            api_client = APIClient()
            
            with show_loading("Generating SHAP explanation..."):
                try:
                    # Get LLM explanation
                    response = api_client._make_request("POST", "/llm/explain-prediction", json=input_data)
                    
                    if response["success"]:
                        data = response["data"]
                        prediction = data.get("prediction", {})
                        shap_explanation = data.get("shap_explanation", {})
                        llm_explanation = data.get("llm_explanation", "")
                        
                        # Display results
                        st.markdown("### 🎯 Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            risk_level = prediction.get("prediction", "unknown")
                            risk_icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                            st.metric("Risk Level", f"{risk_icons.get(risk_level, '⚪')} {risk_level.upper()}")
                        
                        with col2:
                            confidence = prediction.get("confidence", 0)
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col3:
                            processing_time = prediction.get("processing_time", 0)
                            st.metric("Processing Time", f"{processing_time:.3f}s")
                        
                        # SHAP Analysis
                        if shap_explanation and 'feature_impact' in shap_explanation:
                            st.markdown("### 📊 SHAP Feature Impact Analysis")
                            
                            feature_impact = shap_explanation['feature_impact']
                            sorted_features = sorted(feature_impact.items(), key=lambda x: abs(x[1]), reverse=True)
                            
                            # Top 10 features chart
                            top_features = sorted_features[:10]
                            features, impacts = zip(*top_features)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['red' if impact < 0 else 'green' for impact in impacts]
                            bars = ax.barh(range(len(features)), impacts, color=colors, alpha=0.7)
                            
                            ax.set_yticks(range(len(features)))
                            ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
                            ax.set_xlabel('SHAP Value (Impact on Prediction)')
                            ax.set_title('Feature Impact on Risk Prediction')
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            
                            # Add value labels
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                ax.text(width + (0.01 if width >= 0 else -0.01), 
                                       bar.get_y() + bar.get_height()/2,
                                       f'{impacts[i]:.3f}', 
                                       ha='left' if width >= 0 else 'right', va='center')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Feature breakdown
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🔴 Risk Increasing Factors**")
                                negative_features = [(f, v) for f, v in sorted_features if v < 0][:5]
                                for feature, value in negative_features:
                                    st.markdown(f"• **{feature.replace('_', ' ').title()}**: {value:.3f}")
                            
                            with col2:
                                st.markdown("**🟢 Risk Decreasing Factors**")
                                positive_features = [(f, v) for f, v in sorted_features if v > 0][:5]
                                for feature, value in positive_features:
                                    st.markdown(f"• **{feature.replace('_', ' ').title()}**: +{value:.3f}")
                        
                        # LLM Explanation
                        if llm_explanation:
                            st.markdown("### 🤖 AI Explanation")
                            st.markdown(llm_explanation)
                        
                        show_success("SHAP explanation generated successfully!")
                        
                    else:
                        show_error("Failed to generate explanation")
                        
                except Exception as e:
                    show_error(f"Error: {str(e)}")

def display_visualization_gallery():
    """Display SHAP visualization gallery."""
    st.markdown("## 🎨 SHAP Visualization Gallery")
    st.markdown("Interactive SHAP plots and visualizations.")
    
    api_client = APIClient()
    
    if st.button("📊 Generate Summary Plot", use_container_width=True):
        with show_loading("Generating SHAP summary plot..."):
            try:
                response = api_client._make_request("GET", "/explain/summary-plot", params={"sample_size": 50})
                
                if response["success"]:
                    plot_data = response["data"]["plot"]
                    
                    if plot_data:
                        # Decode and display plot
                        image_data = base64.b64decode(plot_data)
                        st.image(image_data, caption="SHAP Summary Plot", use_column_width=True)
                        
                        st.markdown("""
                        **How to read this plot:**
                        - Each dot represents a prediction
                        - X-axis shows SHAP value (impact on prediction)
                        - Color indicates feature value (red=high, blue=low)
                        - Features are ranked by importance (top to bottom)
                        """)
                    else:
                        show_error("Failed to generate plot")
                else:
                    show_error("Failed to generate summary plot")
                    
            except Exception as e:
                show_error(f"Error: {str(e)}")

def main():
    setup_page_config()
    render_header()
    render_sidebar()
    
    st.markdown("# 🔍 Model Explainability")
    st.markdown("Understand how the AI model makes predictions using SHAP (SHapley Additive exPlanations).")
    
    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["🌍 Global Analysis", "🎯 Prediction Explanation", "🎨 Visualizations"])
    
    with tab1:
        display_global_importance()
    
    with tab2:
        display_prediction_explanation()
    
    with tab3:
        display_visualization_gallery()
    
    # Information section
    st.markdown("---")
    st.markdown("## ℹ️ About SHAP")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **What is SHAP?**
        
        SHAP (SHapley Additive exPlanations) is a method to explain individual predictions by computing the contribution of each feature to the prediction.
        
        **Key Benefits:**
        - **Consistent**: Same feature contributions across different models
        - **Accurate**: Sum of contributions equals model output
        - **Interpretable**: Clear positive/negative impact visualization
        """)
    
    with col2:
        st.markdown("""
        **How to Use SHAP Results:**
        
        - **Positive SHAP values** → Increase predicted risk
        - **Negative SHAP values** → Decrease predicted risk  
        - **Larger absolute values** → More influential features
        - **Feature ranking** → Prioritize improvement efforts
        
        Use these insights to focus on the most impactful risk factors.
        """)
    
    render_footer()

# Add API client method for explainability endpoints
def _make_request(self, method: str, endpoint: str, **kwargs):
    """Make HTTP request to API."""
    import requests
    
    url = f"{self.base_url}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, **kwargs, timeout=30)
        elif method == "POST":
            response = requests.post(url, **kwargs, timeout=30)
        else:
            return {"success": False, "error": f"Unsupported method: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# Monkey patch the API client
APIClient._make_request = _make_request

if __name__ == "__main__":
    main()