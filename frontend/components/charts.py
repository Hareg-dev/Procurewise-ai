"""
Reusable chart components for data visualization.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Set style
plt.style.use('default')
sns.set_palette("husl")

def plot_risk_distribution(risk_counts: Dict[str, int], title: str = "Risk Distribution"):
    """Plot risk level distribution as pie chart."""
    fig = go.Figure(data=[go.Pie(
        labels=list(risk_counts.keys()),
        values=list(risk_counts.values()),
        hole=0.3,
        marker_colors=['#ff6b6b', '#feca57', '#48dbfb']
    )])
    
    fig.update_layout(
        title=title,
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    
    return fig

def plot_confidence_distribution(confidences: List[float], title: str = "Confidence Score Distribution"):
    """Plot confidence score distribution as histogram."""
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title=title,
        labels={'x': 'Confidence Score', 'y': 'Count'},
        color_discrete_sequence=['#3742fa']
    )
    
    fig.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        showlegend=False,
        height=400
    )
    
    return fig

def plot_shap_summary(shap_values: Dict[str, float], title: str = "Feature Importance (SHAP)"):
    """Plot SHAP values as horizontal bar chart."""
    if not shap_values:
        return None
    
    # Sort by absolute value
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    features, values = zip(*sorted_features[:10])  # Top 10 features
    
    # Create color map based on positive/negative values
    colors = ['#ff6b6b' if v < 0 else '#48dbfb' for v in values]
    
    fig = go.Figure(data=[go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Features",
        height=500,
        showlegend=False
    )
    
    return fig

def plot_batch_results_summary(results_df: pd.DataFrame):
    """Create comprehensive batch results visualization."""
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        risk_counts = results_df['prediction'].value_counts().to_dict()
        fig_risk = plot_risk_distribution(risk_counts, "Batch Risk Distribution")
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Confidence distribution
        confidences = results_df['confidence'].dropna().tolist()
        fig_conf = plot_confidence_distribution(confidences, "Confidence Distribution")
        st.plotly_chart(fig_conf, use_container_width=True)

def plot_prediction_timeline(history_data: List[Dict]):
    """Plot prediction timeline."""
    if not history_data:
        st.info("No prediction history available")
        return
    
    df = pd.DataFrame(history_data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Group by date and risk level
    daily_counts = df.groupby([df['created_at'].dt.date, 'risk_level']).size().unstack(fill_value=0)
    
    fig = px.line(
        daily_counts.reset_index(),
        x='created_at',
        y=daily_counts.columns,
        title="Prediction Timeline by Risk Level"
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Predictions",
        height=400
    )
    
    return fig

def create_metrics_cards(stats: Dict[str, Any]):
    """Create metric cards for dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Predictions",
            value=stats.get('total_predictions', 0)
        )
    
    with col2:
        avg_conf = stats.get('avg_confidence', 0)
        st.metric(
            label="Avg Confidence",
            value=f"{avg_conf:.2f}"
        )
    
    with col3:
        risk_dist = stats.get('risk_distribution', {})
        high_risk = risk_dist.get('high', 0)
        st.metric(
            label="High Risk Count",
            value=high_risk,
            delta=None if high_risk == 0 else "⚠️"
        )
    
    with col4:
        pred_types = stats.get('prediction_types', {})
        batch_count = pred_types.get('batch', 0)
        st.metric(
            label="Batch Predictions",
            value=batch_count
        )