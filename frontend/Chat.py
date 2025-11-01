"""
Chat page - placeholder for future Mistral LLM chat interface.
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout import setup_page_config, render_header, render_sidebar, render_footer, show_info

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

def display_chat_interface():
    """Display the chat interface (placeholder)."""
    st.markdown("## 💬 AI Chat Assistant")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(f"*{message['timestamp'].strftime('%H:%M:%S')}*")
    
    # Chat input (placeholder)
    user_input = st.chat_input("Ask me about procurement risks, predictions, or strategies...")
    
    if user_input:
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Simulate AI response (placeholder)
        mock_response = generate_mock_response(user_input)
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": mock_response,
            "timestamp": datetime.now()
        })
        
        st.rerun()

def generate_mock_response(user_input: str) -> str:
    """Generate mock AI responses for demonstration."""
    user_input_lower = user_input.lower()
    
    if any(word in user_input_lower for word in ['risk', 'high', 'supplier']):
        return """🔍 **Risk Analysis Insight:**

Based on your query about supplier risk, here are key factors to consider:

• **Financial Stability**: Monitor credit scores and payment history
• **Performance Metrics**: Track delivery scores and dispute rates  
• **Contract Terms**: Ensure appropriate penalty clauses
• **Diversification**: Avoid over-concentration with single suppliers

Would you like me to analyze a specific supplier's risk profile or discuss mitigation strategies?"""
    
    elif any(word in user_input_lower for word in ['predict', 'prediction', 'model']):
        return """🤖 **Model Explanation:**

Our XGBoost model analyzes 20+ features to predict procurement risk:

**Key Features:**
- Financial metrics (revenue, credit score, stability index)
- Performance history (delivery scores, payment delays)
- Relationship factors (duration, dispute history)
- Contract characteristics (value, criticality, terms)

**SHAP Explanations** show which features most influence each prediction. High SHAP values indicate risk-reducing factors, while negative values suggest risk-increasing elements.

Need help interpreting a specific prediction result?"""
    
    elif any(word in user_input_lower for word in ['strategy', 'recommend', 'advice']):
        return """💡 **Strategic Recommendations:**

Here's a framework for procurement risk management:

**1. Risk Assessment**
- Regular supplier evaluations (quarterly)
- Automated risk scoring updates
- Industry benchmark comparisons

**2. Mitigation Strategies**
- Diversified supplier portfolio
- Tiered contract terms based on risk levels
- Performance-based incentives

**3. Monitoring & Response**
- Real-time risk alerts
- Escalation procedures for high-risk events
- Continuous improvement processes

What specific area would you like to explore further?"""
    
    elif any(word in user_input_lower for word in ['help', 'what', 'how']):
        return """🎯 **How I Can Help:**

I'm designed to assist with procurement risk management:

**Analysis & Insights:**
- Interpret prediction results and SHAP explanations
- Identify key risk factors and trends
- Compare supplier performance metrics

**Strategic Guidance:**
- Risk mitigation strategies
- Contract optimization suggestions
- Portfolio diversification advice

**Operational Support:**
- Best practice recommendations
- Compliance guidance
- Performance improvement plans

Try asking me about specific suppliers, risk factors, or strategic decisions!"""
    
    else:
        return """🤔 **I understand you're asking about procurement matters.**

I'm here to help with risk analysis, supplier evaluation, and strategic procurement decisions. 

Some example questions you might ask:
- "How can I reduce risk with high-risk suppliers?"
- "What do the SHAP values mean in my prediction?"
- "How should I diversify my supplier portfolio?"
- "What contract terms help mitigate payment risks?"

*Note: This is a demonstration interface. Full LLM integration coming in Phase 4!*"""

def display_chat_features():
    """Display planned chat features."""
    st.markdown("## 🚀 Planned Chat Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Core Capabilities
        - **Risk Analysis**: Deep dive into prediction results
        - **Strategy Consultation**: Personalized recommendations
        - **Data Interpretation**: Explain SHAP values and metrics
        - **Best Practices**: Industry-specific guidance
        
        ### 💬 Interaction Types
        - **Q&A**: Direct questions about risks and strategies
        - **Analysis**: "Analyze my supplier portfolio"
        - **Comparison**: "Compare these two suppliers"
        - **Planning**: "Help me create a risk management plan"
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Technical Features
        - **Context Awareness**: Remembers your session data
        - **Multi-turn Conversations**: Follow-up questions
        - **Document Analysis**: Upload contracts for review
        - **Export Options**: Save conversations and insights
        
        ### 🎨 User Experience
        - **Natural Language**: Conversational interface
        - **Rich Responses**: Charts, tables, and formatted text
        - **Quick Actions**: Suggested follow-up questions
        - **History**: Access previous conversations
        """)

def display_sample_conversations():
    """Display sample conversation examples."""
    st.markdown("## 💭 Sample Conversations")
    
    conversations = [
        {
            "title": "Risk Mitigation Strategy",
            "messages": [
                ("user", "I have a supplier with high risk score due to payment delays. What should I do?"),
                ("assistant", "Based on payment delay risks, I recommend: 1) Implement milestone payments, 2) Add penalty clauses, 3) Require guarantees, 4) Set up monitoring alerts. Would you like me to draft specific contract language?"),
                ("user", "Yes, please draft penalty clause language."),
                ("assistant", "Here's a sample penalty clause: 'For each day of delay beyond agreed delivery date, Supplier shall pay 0.5% of order value as liquidated damages, capped at 10% total order value.' Adjust percentages based on your risk tolerance.")
            ]
        },
        {
            "title": "SHAP Explanation",
            "messages": [
                ("user", "Can you explain why my supplier got a medium risk rating?"),
                ("assistant", "Looking at your SHAP values: Credit Score (+0.23) and Delivery Score (+0.18) reduce risk, but Payment Delays (-0.31) and Dispute Rate (-0.15) increase it. The net effect places them in medium risk. Focus on improving payment terms."),
                ("user", "How can I improve their payment behavior?"),
                ("assistant", "Try these approaches: 1) Early payment discounts (2% for 10-day payment), 2) Automated payment reminders, 3) Shorter payment terms (Net 15 vs Net 30), 4) Direct debit arrangements, 5) Performance-based contract renewals.")
            ]
        }
    ]
    
    for conv in conversations:
        with st.expander(f"📝 {conv['title']}", expanded=False):
            for role, message in conv["messages"]:
                if role == "user":
                    st.markdown(f"**👤 You:** {message}")
                else:
                    st.markdown(f"**🤖 AI:** {message}")
                st.markdown("")

def main():
    setup_page_config()
    render_header()
    render_sidebar()
    
    st.markdown("# 💬 AI Chat Assistant")
    st.markdown("*Powered by Mistral 7B - Coming in Phase 4*")
    
    # Initialize chat session
    initialize_chat_session()
    
    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "🚀 Features", "💭 Examples"])
    
    with tab1:
        # Coming soon banner
        st.info("🚧 **Demo Mode**: This is a preview of the chat interface. Full LLM integration with Mistral 7B will be available in Phase 4!")
        
        display_chat_interface()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_messages = st.session_state.chat_messages[:1]  # Keep welcome message
            st.rerun()
    
    with tab2:
        display_chat_features()
    
    with tab3:
        display_sample_conversations()
    
    # Integration status
    st.markdown("---")
    st.markdown("## 🔧 Integration Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Current Status:**
        - ✅ Chat UI Interface Ready
        - ✅ Message History Management
        - ✅ Mock Response System
        - 🔄 Awaiting Mistral 7B Integration
        """)
    
    with col2:
        st.markdown("""
        **Phase 4 Requirements:**
        - 🔧 Ollama Setup (localhost:11434)
        - 🤖 Mistral 7B Model Download
        - 🔗 FastAPI LLM Endpoints
        - 📝 Procurement Domain Prompts
        """)
    
    render_footer()

if __name__ == "__main__":
    main()