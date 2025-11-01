"""
SHAP visualization components for explainable AI insights.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional

def render_shap_explanation(shap_values: Dict[str, float], prediction: str, confidence: float):
    """Render SHAP explanation with interpretation."""
    if not shap_values:
        st.warning("SHAP values not available for this prediction")
        return
    
    st.subheader("🔍 Feature Impact Analysis (SHAP)")
    
    # Sort features by absolute impact
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Create two columns for positive and negative impacts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 Risk Increasing Factors**")
        negative_features = [(f, v) for f, v in sorted_features if v < 0][:5]
        
        if negative_features:
            for feature, value in negative_features:
                st.markdown(f"• **{feature.replace('_', ' ').title()}**: {value:.3f}")
        else:
            st.info("No significant risk-increasing factors")
    
    with col2:
        st.markdown("**🟢 Risk Decreasing Factors**")
        positive_features = [(f, v) for f, v in sorted_features if v > 0][:5]
        
        if positive_features:
            for feature, value in positive_features:
                st.markdown(f"• **{feature.replace('_', ' ').title()}**: +{value:.3f}")
        else:
            st.info("No significant risk-decreasing factors")
    
    # Feature importance bar chart
    st.markdown("**📊 Feature Importance Visualization**")
    
    # Prepare data for plotting
    features = [f.replace('_', ' ').title() for f, _ in sorted_features[:10]]
    values = [v for _, v in sorted_features[:10]]
    colors = ['red' if v < 0 else 'green' for v in values]
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, values, color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_xlabel('SHAP Value (Impact on Prediction)')
    ax.set_title(f'Top 10 Feature Impacts for {prediction.upper()} Risk Prediction')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def render_shap_interpretation(prediction: str, confidence: float, shap_values: Dict[str, float]):
    """Provide human-readable interpretation of SHAP results."""
    st.subheader("🧠 AI Explanation")
    
    # Risk level interpretation
    risk_colors = {
        'low': '🟢',
        'medium': '🟡', 
        'high': '🔴'
    }
    
    risk_icon = risk_colors.get(prediction.lower(), '⚪')
    
    st.markdown(f"""
    **Prediction Summary:**
    - **Risk Level**: {risk_icon} **{prediction.upper()}**
    - **Confidence**: {confidence:.1%}
    """)
    
    if not shap_values:
        return
    
    # Find most impactful features
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    top_feature, top_impact = sorted_features[0] if sorted_features else (None, 0)
    
    if top_feature:
        impact_direction = "increases" if top_impact < 0 else "decreases"
        st.markdown(f"""
        **Key Insight:**
        The most influential factor is **{top_feature.replace('_', ' ').title()}** 
        which {impact_direction} the risk by {abs(top_impact):.3f} points.
        """)
    
    # Risk level specific advice
    if prediction.lower() == 'high':
        st.error("""
        **⚠️ High Risk Alert:**
        This supplier shows significant risk factors. Consider:
        - Enhanced due diligence
        - Additional contract safeguards
        - Regular monitoring
        """)
    elif prediction.lower() == 'medium':
        st.warning("""
        **🔍 Medium Risk Notice:**
        This supplier has moderate risk. Recommended actions:
        - Standard monitoring procedures
        - Review key risk factors
        - Consider mitigation strategies
        """)
    else:
        st.success("""
        **✅ Low Risk Assessment:**
        This supplier appears to be low risk. Maintain:
        - Regular relationship management
        - Standard contract terms
        - Periodic reviews
        """)

def create_shap_summary_table(shap_values: Dict[str, float]) -> pd.DataFrame:
    """Create a summary table of SHAP values."""
    if not shap_values:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {
            'Feature': feature.replace('_', ' ').title(),
            'SHAP Value': value,
            'Impact': 'Risk Increasing' if value < 0 else 'Risk Decreasing',
            'Magnitude': abs(value)
        }
        for feature, value in shap_values.items()
    ])
    
    return df.sort_values('Magnitude', ascending=False)