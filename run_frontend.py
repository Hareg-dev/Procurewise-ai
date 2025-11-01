#!/usr/bin/env python3
"""
Script to run the Streamlit frontend application.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit frontend."""
    frontend_path = Path(__file__).parent / "frontend" / "app.py"
    
    print("🚀 Starting Procurement Risk Prediction Frontend...")
    print(f"📁 Frontend path: {frontend_path}")
    print("🌐 Access the app at: http://localhost:8501")
    print("=" * 60)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(frontend_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

if __name__ == "__main__":
    main()