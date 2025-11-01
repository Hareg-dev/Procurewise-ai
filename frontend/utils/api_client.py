"""
API client for connecting Streamlit frontend to FastAPI backend.
"""
import requests
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional
import json

class APIClient:
    """Client for interacting with the FastAPI backend."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = self._get_session_id()
    
    def _get_session_id(self) -> str:
        """Get or create session ID."""
        if 'session_id' not in st.session_state:
            try:
                response = requests.post(f"{self.base_url}/history/session", json={"user_id": "streamlit_user"})
                if response.status_code == 200:
                    st.session_state.session_id = response.json()['session_id']
                else:
                    st.session_state.session_id = "default-session"
            except:
                st.session_state.session_id = "default-session"
        return st.session_state.session_id
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return {"status": "connected", "data": response.json()}
        except requests.exceptions.ConnectionError:
            return {"status": "disconnected", "error": "Cannot connect to API server"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make single prediction request."""
        try:
            headers = {"X-Session-ID": self.session_id}
            response = requests.post(
                f"{self.base_url}/predict/single",
                json=input_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"API Error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
    
    def predict_batch(self, csv_file) -> Dict[str, Any]:
        """Make batch prediction request with CSV file."""
        try:
            headers = {"X-Session-ID": self.session_id}
            files = {"file": ("batch_data.csv", csv_file, "text/csv")}
            
            response = requests.post(
                f"{self.base_url}/predict/batch",
                files=files,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"API Error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
    
    def get_prediction_history(self, limit: int = 50) -> Dict[str, Any]:
        """Get prediction history for current session."""
        try:
            params = {"session_id": self.session_id, "limit": limit}
            response = requests.get(f"{self.base_url}/history/predictions", params=params, timeout=10)
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            params = {"session_id": self.session_id}
            response = requests.get(f"{self.base_url}/history/stats", params=params, timeout=10)
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": f"Request failed: {str(e)}"}