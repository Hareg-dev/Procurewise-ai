"""
Shared layout components for consistent UI across pages.
"""
import streamlit as st
from utils.api_client import APIClient

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Procurement Risk Prediction",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_header():
    """Render consistent header across all pages."""
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

def render_sidebar():
    """Render navigation sidebar."""
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        
        # System Status
        try:
            api_client = APIClient()
            health = api_client.health_check()
            
            # Check LLM status
            try:
                llm_health = api_client._make_request("GET", "/llm/health")
                llm_status = llm_health.get("success", False)
            except:
                llm_status = False
                
        except Exception as e:
            health = {"status": "error", "error": str(e)}
            llm_status = False
        
        st.markdown("### 🔧 System Status")
        
        if health["status"] == "connected":
            st.success("🟢 Backend API")
        else:
            st.error("🔴 Backend API")
            st.caption(f"Error: {health.get('error', 'Unknown error')}")
        
        if llm_status:
            st.success("🟢 Phi3:mini")
        else:
            st.error("🔴 Phi3:mini")
            st.caption("LLM features unavailable")
        
        st.markdown("---")
        
        # Navigation menu
        pages = {
            "🏠 Home": "Home",
            "🎯 Single Prediction": "Predict_Single", 
            "📊 Batch Prediction": "Predict_Batch",
            "📈 Analysis": "Analysis",
            "🔍 Explainability": "Explainability",
            "💡 Recommendations": "Recommendations",
            "💬 Chat Assistant": "ChatAssistant"
        }
        
        st.markdown("### Pages")
        for display_name, page_name in pages.items():
            if st.button(display_name, key=f"nav_{page_name}", use_container_width=True):
                st.session_state.current_page = page_name
                st.rerun()
        
        st.markdown("---")
        
        # Session info
        try:
            if hasattr(api_client, 'session_id'):
                st.markdown("### 📋 Session Info")
                st.caption(f"Session ID: {api_client.session_id[:8]}...")
        except:
            pass

def render_footer():
    """Render consistent footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Procurement Risk Prediction Platform | Powered by XGBoost & SHAP</p>
    </div>
    """, unsafe_allow_html=True)

def show_loading(message: str = "Processing..."):
    """Show loading spinner with message."""
    return st.spinner(message)

def show_success(message: str):
    """Show success message."""
    st.success(f"✅ {message}")

def show_error(message: str):
    """Show error message."""
    st.error(f"❌ {message}")

def show_warning(message: str):
    """Show warning message."""
    st.warning(f"⚠️ {message}")

def show_info(message: str):
    """Show info message."""
    st.info(f"ℹ️ {message}")