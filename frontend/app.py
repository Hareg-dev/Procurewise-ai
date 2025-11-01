"""
Main Streamlit application with multipage navigation.
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import pages
import Home, Predict_Single, Predict_Batch, Analysis, Recommendations, Chat, Explainability, ChatAssistant
from frontend.components.layout import setup_page_config

def main():
    """Main application with page routing."""
    setup_page_config()
    
    # Initialize current page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    # Page routing
    pages = {
        'Home': Home,
        'Predict_Single': Predict_Single,
        'Predict_Batch': Predict_Batch,
        'Analysis': Analysis,
        'Explainability': Explainability,
        'Recommendations': Recommendations,
        'Chat': Chat,
        'ChatAssistant': ChatAssistant
    }
    
    # Run the selected page
    current_page = st.session_state.current_page
    if current_page in pages:
        pages[current_page].main()
    else:
        Home.main()

if __name__ == "__main__":
    main()