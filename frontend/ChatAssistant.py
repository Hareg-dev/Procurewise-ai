"""
Chat Assistant page with Mistral LLM integration for procurement insights.
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout import setup_page_config, render_header, render_sidebar, render_footer, show_loading, show_success, show_error, show_info
from utils.api_client import APIClient

def initialize_chat_session():
    """Initialize chat session state."""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "👋 Hello! I'm your AI procurement assistant powered by Mistral 7B. I can help you analyze risks, interpret predictions, and provide strategic insights. How can I assist you today?",
                "timestamp": datetime.now()
            }
        ]
    
    if 'chat_session_id' not in st.session_state:
        import uuid
        st.session_state.chat_session_id = str(uuid.uuid4())

def display_chat_interface():
    """Display the main chat interface."""
    st.markdown("## 💬 AI Chat Assistant")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(f"*{message['timestamp'].strftime('%H:%M:%S')}*")
    
    # Chat input
    user_input = st.chat_input("Ask me about procurement risks, predictions, or strategies...")
    
    if user_input:
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Get AI response
        api_client = APIClient()
        
        with st.spinner("🤖 Thinking..."):
            try:
                response = api_client._make_request("POST", "/llm/chat", json={
                    "message": user_input,
                    "session_id": st.session_state.chat_session_id,
                    "include_context": True
                })
                
                if response["success"]:
                    ai_response = response["data"]["response"]
                else:
                    ai_response = f"I apologize, but I'm having trouble connecting to my AI backend. Error: {response.get('error', 'Unknown error')}"
                    
            except Exception as e:
                ai_response = f"I'm currently unable to process your request. Please ensure the backend API is running. Error: {str(e)}"
        
        # Add AI response
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.now()
        })
        
        st.rerun()

def display_chat_controls():
    """Display chat control buttons and options."""
    st.markdown("### 🎛️ Chat Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = st.session_state.chat_messages[:1]  # Keep welcome message
            st.rerun()
    
    with col2:
        if st.button("📋 Export Chat", use_container_width=True):
            chat_export = ""
            for msg in st.session_state.chat_messages:
                role = "You" if msg["role"] == "user" else "AI Assistant"
                chat_export += f"{role} ({msg['timestamp'].strftime('%H:%M:%S')}): {msg['content']}\n\n"
            
            st.download_button(
                label="💾 Download Chat History",
                data=chat_export,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("🔄 Check AI Status", use_container_width=True):
            api_client = APIClient()
            
            try:
                response = api_client._make_request("GET", "/llm/health")
                
                if response["success"] and response.get("status") == "healthy":
                    show_success("AI Assistant is online and ready!")
                    st.json(response)
                else:
                    show_error(f"AI Assistant is offline: {response.get('error', 'Unknown error')}")
            except Exception as e:
                show_error(f"Cannot check AI status: {str(e)}")

def display_suggested_questions():
    """Display suggested questions for users."""
    st.markdown("### 💡 Suggested Questions")
    
    suggestions = [
        "How can I reduce risk with high-risk suppliers?",
        "What do the SHAP values mean in my prediction?",
        "How should I diversify my supplier portfolio?",
        "What contract terms help mitigate payment risks?",
        "Explain the difference between high and medium risk suppliers",
        "What are the most important factors in procurement risk?",
        "How can I improve supplier performance?",
        "What should I do if a supplier's risk score increases?"
    ]
    
    # Display suggestions in a grid
    cols = st.columns(2)
    
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(f"💭 {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                # Add suggestion as user message
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": suggestion,
                    "timestamp": datetime.now()
                })
                st.rerun()

def display_what_if_analyzer():
    """Display what-if scenario analyzer."""
    st.markdown("### 🔮 What-If Scenario Analyzer")
    st.markdown("Analyze how changes to supplier data might affect risk predictions.")
    
    with st.expander("🧪 Run What-If Analysis", expanded=False):
        st.markdown("**Original Supplier Data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            orig_revenue = st.number_input("Original Annual Revenue", value=1000000, key="orig_revenue")
            orig_credit = st.number_input("Original Credit Score", value=650, key="orig_credit")
            orig_delivery = st.slider("Original Delivery Score", 0, 100, 70, key="orig_delivery")
        
        with col2:
            new_revenue = st.number_input("New Annual Revenue", value=1200000, key="new_revenue")
            new_credit = st.number_input("New Credit Score", value=720, key="new_credit")
            new_delivery = st.slider("New Delivery Score", 0, 100, 85, key="new_delivery")
        
        if st.button("🔍 Analyze What-If Scenario", use_container_width=True):
            original_data = {
                "annual_revenue": orig_revenue,
                "credit_score": orig_credit,
                "delivery_score": orig_delivery
            }
            
            modified_data = {
                "annual_revenue": new_revenue,
                "credit_score": new_credit,
                "delivery_score": new_delivery
            }
            
            api_client = APIClient()
            
            with show_loading("Analyzing scenario..."):
                try:
                    response = api_client._make_request("POST", "/llm/what-if", json={
                        "original_data": original_data,
                        "modified_data": modified_data,
                        "original_prediction": {"prediction": "medium"}  # Mock prediction
                    })
                    
                    if response["success"]:
                        analysis = response["data"]["analysis"]
                        st.markdown("**🤖 AI Analysis:**")
                        st.markdown(analysis)
                    else:
                        show_error("Failed to analyze scenario")
                        
                except Exception as e:
                    show_error(f"Error: {str(e)}")

def display_conversation_insights():
    """Display insights from conversation history."""
    st.markdown("### 📊 Conversation Insights")
    
    if len(st.session_state.chat_messages) > 1:
        user_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "user"]
        ai_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "assistant"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", len(st.session_state.chat_messages))
        
        with col2:
            st.metric("Your Questions", len(user_messages))
        
        with col3:
            st.metric("AI Responses", len(ai_messages))
        
        # Recent topics
        if user_messages:
            st.markdown("**Recent Topics:**")
            recent_messages = user_messages[-3:]
            for i, msg in enumerate(recent_messages, 1):
                st.markdown(f"{i}. {msg['content'][:50]}...")
    else:
        show_info("Start a conversation to see insights!")

def main():
    setup_page_config()
    render_header()
    render_sidebar()
    
    st.markdown("# 💬 AI Chat Assistant")
    st.markdown("*Powered by Mistral 7B via Ollama*")
    
    # Initialize chat session
    initialize_chat_session()
    
    # Main chat interface
    display_chat_interface()
    
    # Sidebar with additional features
    with st.sidebar:
        st.markdown("---")
        display_chat_controls()
        
        st.markdown("---")
        display_conversation_insights()
    
    # Additional features in tabs
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["💡 Suggestions", "🔮 What-If Analysis", "ℹ️ Help"])
    
    with tab1:
        display_suggested_questions()
    
    with tab2:
        display_what_if_analyzer()
    
    with tab3:
        st.markdown("## 🤖 How to Use the AI Assistant")
        
        st.markdown("""
        **Getting Started:**
        1. Type your question in the chat input below
        2. Ask about risk predictions, SHAP explanations, or procurement strategies
        3. Use suggested questions for inspiration
        4. Try what-if scenarios to explore different outcomes
        
        **What I Can Help With:**
        - 🎯 **Risk Analysis**: Explain prediction results and SHAP values
        - 💡 **Recommendations**: Suggest risk mitigation strategies
        - 📊 **Data Interpretation**: Help understand model outputs
        - 🔮 **Scenario Planning**: Analyze what-if situations
        - 📋 **Best Practices**: Share procurement industry insights
        
        **Tips for Better Conversations:**
        - Be specific about your questions
        - Mention specific suppliers or scenarios
        - Ask follow-up questions for deeper insights
        - Use the what-if analyzer for scenario planning
        """)
        
        st.markdown("### 🔧 Technical Requirements")
        st.markdown("""
        **For full functionality, ensure:**
        - ✅ FastAPI backend is running (localhost:8000)
        - ✅ Ollama is installed and running (localhost:11434)
        - ✅ Mistral 7B model is downloaded in Ollama
        
        **Check AI Status** using the button in the sidebar to verify connectivity.
        """)
    
    render_footer()

if __name__ == "__main__":
    main()