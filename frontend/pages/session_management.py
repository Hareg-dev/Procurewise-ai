"""
Session Management page for handling user sessions and prediction history.
"""
import streamlit as st
import requests
import pandas as pd
import uuid
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def show_session_management():
    st.markdown("# 👤 Session Management")
    
    tab1, tab2, tab3 = st.tabs(["Current Session", "Prediction History", "Session Control"])
    
    with tab1:
        show_current_session()
    
    with tab2:
        show_prediction_history()
    
    with tab3:
        show_session_control()

def show_current_session():
    st.markdown("### 📋 Current Session Info")
    
    session_id = st.session_state.get('session_id', 'No session')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Session ID:** {session_id}")
        st.info(f"**Predictions in Memory:** {len(st.session_state.get('prediction_history', []))}")
    
    with col2:
        if st.button("📊 Get Session Statistics"):
            try:
                response = requests.get(f"{API_BASE_URL}/history/stats", 
                                      params={"session_id": session_id})
                
                if response.status_code == 200:
                    stats = response.json()
                    
                    st.success("✅ Session Stats Retrieved")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Total Predictions", stats["total_predictions"])
                    
                    with metric_col2:
                        st.metric("Average Confidence", f"{stats['avg_confidence']:.1%}")
                    
                    with metric_col3:
                        single_count = stats["prediction_types"]["single"]
                        batch_count = stats["prediction_types"]["batch"]
                        st.metric("Single/Batch", f"{single_count}/{batch_count}")
                    
                    # Risk distribution
                    if stats["risk_distribution"]:
                        st.markdown("#### Risk Distribution")
                        risk_df = pd.DataFrame(
                            list(stats["risk_distribution"].items()),
                            columns=["Risk Level", "Count"]
                        )
                        st.bar_chart(risk_df.set_index("Risk Level"))
                
                else:
                    st.error(f"Failed to get session stats: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

def show_prediction_history():
    st.markdown("### 📜 Prediction History")
    
    session_id = st.session_state.get('session_id', 'default')
    
    col1, col2 = st.columns(2)
    
    with col1:
        limit = st.number_input("Number of records", min_value=1, max_value=100, value=20)
        prediction_type = st.selectbox("Filter by type", ["All", "single", "batch"])
    
    with col2:
        offset = st.number_input("Skip records", min_value=0, value=0)
    
    if st.button("📋 Get Prediction History"):
        try:
            params = {
                "session_id": session_id,
                "limit": limit,
                "offset": offset
            }
            
            if prediction_type != "All":
                params["prediction_type"] = prediction_type
            
            response = requests.get(f"{API_BASE_URL}/history/predictions", params=params)
            
            if response.status_code == 200:
                predictions = response.json()
                
                if predictions:
                    st.success(f"✅ Retrieved {len(predictions)} predictions")
                    
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(predictions)
                    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Display as table
                    st.dataframe(df[['id', 'prediction_type', 'risk_level', 'confidence_score', 'created_at', 'processing_time']], 
                               use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download History CSV",
                        data=csv,
                        file_name=f"prediction_history_{session_id[:8]}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.info("No prediction history found")
            
            else:
                st.error(f"Failed to get history: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def show_session_control():
    st.markdown("### ⚙️ Session Control")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Create New Session")
        
        user_id = st.text_input("User ID (optional)", placeholder="Enter user identifier")
        
        if st.button("🆕 Create New Session"):
            try:
                session_data = {"user_id": user_id if user_id else None}
                
                response = requests.post(f"{API_BASE_URL}/history/session", json=session_data)
                
                if response.status_code == 200:
                    result = response.json()
                    new_session_id = result["session_id"]
                    
                    st.success(f"✅ New session created: {new_session_id}")
                    
                    # Update session state
                    st.session_state.session_id = new_session_id
                    st.session_state.prediction_history = []
                    st.session_state.last_prediction = None
                    
                    st.info("Session updated! You can now make predictions with the new session.")
                    
                else:
                    st.error(f"Failed to create session: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("#### Session Actions")
        
        current_session = st.session_state.get('session_id', 'No session')
        st.info(f"Current: {current_session[:8]}..." if len(current_session) > 8 else current_session)
        
        if st.button("🔄 Reset Local Session"):
            # Reset session state
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.prediction_history = []
            st.session_state.last_prediction = None
            st.session_state.messages = []
            
            st.success("✅ Local session reset")
            st.info(f"New session ID: {st.session_state.session_id[:8]}...")
        
        if st.button("🗑️ Clear Local History"):
            # Clear only history, keep session ID
            st.session_state.prediction_history = []
            st.session_state.last_prediction = None
            st.session_state.messages = []
            
            st.success("✅ Local history cleared")
    
    # Session info
    st.markdown("---")
    st.markdown("#### 📊 Session Information")
    
    info_data = {
        "Current Session ID": st.session_state.get('session_id', 'None'),
        "Local Predictions": len(st.session_state.get('prediction_history', [])),
        "Last Prediction": "Yes" if st.session_state.get('last_prediction') else "No",
        "Chat Messages": len(st.session_state.get('messages', [])) - 1 if st.session_state.get('messages') else 0  # -1 for initial message
    }
    
    for key, value in info_data.items():
        st.text(f"{key}: {value}")