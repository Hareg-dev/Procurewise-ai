#!/usr/bin/env python3
"""
Setup script for Ollama and Mistral 7B model installation.
"""
import subprocess
import sys
import requests
import time

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_ollama():
    """Install Ollama."""
    print("📦 Installing Ollama...")
    
    try:
        # Download and install Ollama
        subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'], check=True)
        print("✅ Ollama installation script downloaded")
        
        # Run installation
        subprocess.run(['sh', '-c', 'curl -fsSL https://ollama.ai/install.sh | sh'], check=True)
        print("✅ Ollama installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Ollama: {e}")
        return False
    
    return True

def start_ollama_service():
    """Start Ollama service."""
    print("🚀 Starting Ollama service...")
    
    try:
        # Start Ollama in background
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        time.sleep(5)
        
        # Check if service is running
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service started successfully")
            return True
        else:
            print("❌ Ollama service not responding")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start Ollama service: {e}")
        return False

def pull_mistral_model():
    """Pull Mistral 7B model."""
    print("📥 Downloading Mistral 7B model (this may take a while)...")
    
    try:
        result = subprocess.run(['ollama', 'pull', 'mistral:7b'], check=True, capture_output=True, text=True)
        print("✅ Mistral 7B model downloaded successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download Mistral model: {e}")
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        return False

def test_mistral_model():
    """Test Mistral model."""
    print("🧪 Testing Mistral model...")
    
    try:
        # Test API call
        payload = {
            "model": "mistral:7b",
            "prompt": "Hello, are you working?",
            "stream": False
        }
        
        response = requests.post('http://localhost:11434/api/generate', json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Mistral model is working!")
            print(f"Test response: {result.get('response', 'No response')[:100]}...")
            return True
        else:
            print(f"❌ Mistral model test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test Mistral model: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("OLLAMA + MISTRAL 7B SETUP")
    print("=" * 60)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("Ollama not found. Installing...")
        if not install_ollama():
            print("❌ Setup failed: Could not install Ollama")
            sys.exit(1)
    else:
        print("✅ Ollama is already installed")
    
    # Start Ollama service
    if not start_ollama_service():
        print("❌ Setup failed: Could not start Ollama service")
        sys.exit(1)
    
    # Pull Mistral model
    if not pull_mistral_model():
        print("❌ Setup failed: Could not download Mistral model")
        sys.exit(1)
    
    # Test model
    if not test_mistral_model():
        print("❌ Setup failed: Mistral model not working")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Ollama is running on: http://localhost:11434")
    print("Mistral 7B model is ready for use")
    print("\nYou can now run the backend with LLM support:")
    print("python3 backend/main.py")

if __name__ == "__main__":
    main()