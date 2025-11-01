"""
Analysis page for interactive analytics and prediction history visualization.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout import setup_page_config, render_header, render_sidebar, render_footer, show_info
from components.charts import create_metrics_cards, plot_prediction_timeline, plot_risk_distribution, plot_confidence_distribution
from utils.api_client import APIClient

def display_session_overview():
    """Display session overview with key metrics."""
    st.markdown("## 📊 Session Overview")
    
    api_client = APIClient()
    stats_result = api_client.get_session_stats()
    
    if stats_result["success"]:
        stats = stats_result["data"]
        create_metrics_cards(stats)
        
        # Risk distribution visualization
        if stats.get('total_predictions', 0) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                risk_dist = stats.get('risk_distribution', {})
                if any(risk_dist.values()):
                    fig_risk = plot_risk_distribution(risk_dist, "Session Risk Distribution")
                    st.plotly_chart(fig_risk, use_container_width=True)
            
            with col2:
                pred_types = stats.get('prediction_types', {})
                if any(pred_types.values()):
                    fig_types = plot_risk_distribution(pred_types, "Prediction Types")
                    st.plotly_chart(fig_types, use_container_width=True)
        else:
            show_info("No predictions made yet. Start with single or batch predictions to see analytics.")
    else:
        st.error("Unable to load session statistics")

def display_prediction_history():
    """Display detailed prediction history with filtering."""
    st.markdown("## 📋 Prediction History")
    
    api_client = APIClient()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction_type = st.selectbox(
            "Prediction Type",
            ["All", "single", "batch"],
            key="history_type_filter"
        )
    
    with col2:
        limit = st.selectbox(
            "Records to Show",
            [10, 25, 50, 100],
            index=1,
            key="history_limit"
        )
    
    with col3:
        if st.button("🔄 Refresh History"):
            st.rerun()
    
    # Get history data
    history_result = api_client.get_prediction_history(limit=limit)
    
    if history_result["success"] and history_result["data"]:
        history_data = history_result["data"]
        df = pd.DataFrame(history_data)
        
        # Apply filters
        if prediction_type != "All":
            df = df[df['prediction_type'] == prediction_type]
        
        if not df.empty:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            
            with col2:
                avg_confidence = df['confidence_score'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col3:
                high_risk_count = len(df[df['risk_level'] == 'high'])
                st.metric("High Risk Count", high_risk_count)
            
            with col4:
                avg_processing_time = df['processing_time'].mean()
                st.metric("Avg Processing Time", f"{avg_processing_time:.3f}s")
            
            # Timeline visualization
            if len(df) > 1:
                st.markdown("### 📈 Prediction Timeline")
                timeline_fig = plot_prediction_timeline(history_data)
                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Detailed table
            st.markdown("### 📋 Detailed History")
            
            # Format datetime for display
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Display table with formatting
            st.dataframe(
                df[['prediction_type', 'risk_level', 'confidence_score', 'processing_time', 'created_at']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'prediction_type': 'Type',
                    'risk_level': 'Risk Level',
                    'confidence_score': st.column_config.ProgressColumn(
                        'Confidence',
                        min_value=0,
                        max_value=1,
                        format="%.1%"
                    ),
                    'processing_time': 'Processing Time (s)',
                    'created_at': 'Created At'
                }
            )
        else:
            show_info("No predictions match the selected filters.")
    else:
        show_info("No prediction history available.")

def display_batch_summaries():
    """Display batch prediction summaries."""
    st.markdown("## 📦 Batch Prediction Summaries")
    
    api_client = APIClient()
    
    # Get batch summaries (this endpoint needs to be implemented in the API)
    try:
        response = api_client.get_session_stats()  # Using stats for now
        
        if response["success"]:
            stats = response["data"]
            batch_count = stats.get('prediction_types', {}).get('batch', 0)
            
            if batch_count > 0:
                st.markdown(f"**Total Batch Predictions:** {batch_count}")
                
                # Placeholder for batch summaries
                st.info("Detailed batch summaries will be displayed here when available from the API.")
            else:
                show_info("No batch predictions found. Upload a CSV file to create batch predictions.")
        else:
            st.error("Unable to load batch summaries")
            
    except Exception as e:
        st.error(f"Error loading batch summaries: {str(e)}")

def display_risk_insights():
    """Display risk analysis insights."""
    st.markdown("## 🔍 Risk Analysis Insights")
    
    api_client = APIClient()
    stats_result = api_client.get_session_stats()
    
    if stats_result["success"]:
        stats = stats_result["data"]
        risk_dist = stats.get('risk_distribution', {})
        total_predictions = stats.get('total_predictions', 0)
        
        if total_predictions > 0:
            # Risk level analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Risk Level Analysis")
                
                high_risk_pct = (risk_dist.get('high', 0) / total_predictions) * 100
                medium_risk_pct = (risk_dist.get('medium', 0) / total_predictions) * 100
                low_risk_pct = (risk_dist.get('low', 0) / total_predictions) * 100
                
                st.markdown(f"- **High Risk:** {high_risk_pct:.1f}% ({risk_dist.get('high', 0)} suppliers)")
                st.markdown(f"- **Medium Risk:** {medium_risk_pct:.1f}% ({risk_dist.get('medium', 0)} suppliers)")
                st.markdown(f"- **Low Risk:** {low_risk_pct:.1f}% ({risk_dist.get('low', 0)} suppliers)")
                
                # Risk recommendations
                if high_risk_pct > 30:
                    st.warning("⚠️ High percentage of high-risk suppliers detected. Consider enhanced due diligence.")
                elif high_risk_pct > 15:
                    st.info("ℹ️ Moderate high-risk exposure. Monitor closely.")
                else:
                    st.success("✅ Low high-risk exposure. Good supplier portfolio health.")
            
            with col2:
                st.markdown("### 📊 Performance Metrics")
                
                avg_confidence = stats.get('avg_confidence', 0)
                
                st.markdown(f"- **Average Confidence:** {avg_confidence:.1%}")
                
                if avg_confidence > 0.8:
                    st.success("✅ High prediction confidence")
                elif avg_confidence > 0.6:
                    st.info("ℹ️ Moderate prediction confidence")
                else:
                    st.warning("⚠️ Low prediction confidence - review input data quality")
                
                # Prediction type breakdown
                pred_types = stats.get('prediction_types', {})
                single_count = pred_types.get('single', 0)
                batch_count = pred_types.get('batch', 0)
                
                st.markdown(f"- **Single Predictions:** {single_count}")
                st.markdown(f"- **Batch Predictions:** {batch_count}")
        else:
            show_info("No predictions available for analysis. Make some predictions first!")
    else:
        st.error("Unable to load risk insights")

def main():
    setup_page_config()
    render_header()
    render_sidebar()
    
    st.markdown("# 📈 Analytics Dashboard")
    st.markdown("Explore your prediction history, analyze risk patterns, and gain insights from your assessments.")
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 History", "📦 Batch Results", "🔍 Insights"])
    
    with tab1:
        display_session_overview()
    
    with tab2:
        display_prediction_history()
    
    with tab3:
        display_batch_summaries()
    
    with tab4:
        display_risk_insights()
    
    render_footer()

if __name__ == "__main__":
    main()