"""
Advanced Analytics page for comprehensive data analysis and insights.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def show_advanced_analytics():
    st.markdown("# 📊 Advanced Analytics")
    
    session_id = st.session_state.get('session_id', 'default')
    
    tab1, tab2, tab3 = st.tabs(["Session Analytics", "Batch Analysis", "System Health"])
    
    with tab1:
        if st.button("📈 Get Session Stats"):
            try:
                response = requests.get(f"{API_BASE_URL}/history/stats", params={"session_id": session_id})
                if response.status_code == 200:
                    stats = response.json()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", stats["total_predictions"])
                    with col2:
                        st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
                    with col3:
                        st.metric("Single/Batch", f"{stats['prediction_types']['single']}/{stats['prediction_types']['batch']}")
                    
                    if stats["risk_distribution"]:
                        fig = px.pie(values=list(stats["risk_distribution"].values()), 
                                   names=list(stats["risk_distribution"].keys()))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Failed: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        if st.button("📦 Get Batch Summaries"):
            try:
                response = requests.get(f"{API_BASE_URL}/history/batch-summaries", 
                                      params={"session_id": session_id})
                if response.status_code == 200:
                    summaries = response.json()
                    for summary in summaries:
                        with st.expander(f"{summary['file_name']} - {summary['total_records']} records"):
                            st.json(summary)
                else:
                    st.error(f"Failed: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Check Prediction Health"):
                try:
                    response = requests.get(f"{API_BASE_URL}/predict/health")
                    if response.status_code == 200:
                        st.success("✅ Healthy")
                        st.json(response.json())
                    else:
                        st.error("❌ Unhealthy")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("🤖 Check LLM Health"):
                try:
                    response = requests.get(f"{API_BASE_URL}/llm/health")
                    if response.status_code == 200:
                        st.success("✅ Healthy")
                        st.json(response.json())
                    else:
                        st.error("❌ Unhealthy")
                except Exception as e:
                    st.error(f"Error: {str(e)}")