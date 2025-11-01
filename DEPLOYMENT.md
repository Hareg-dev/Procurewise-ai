# Deployment Guide

## GitHub Deployment

The project is ready for deployment to GitHub. Follow these steps:

### 1. Authentication Setup

```bash
# Configure Git with your credentials
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# For HTTPS (recommended)
git remote set-url origin https://github.com/Bekalu1997et/Procurewise-ai.git

# Push to GitHub
git push -u origin main
```

### 2. Alternative: SSH Setup

```bash
# Generate SSH key (if not exists)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add SSH key to GitHub account
cat ~/.ssh/id_ed25519.pub

# Set SSH remote
git remote set-url origin git@github.com:Bekalu1997et/Procurewise-ai.git
git push -u origin main
```

### 3. Repository Structure

```
Procurewise-ai/
├── backend/           # FastAPI backend
├── frontend/          # Streamlit frontend  
├── src/              # Core ML modules
├── data/             # Training data
├── tests/            # Testing framework
├── requirements.txt  # Dependencies
├── README.md         # Documentation
└── .gitignore       # Git ignore rules
```

## Local Development

### Prerequisites
- Python 3.8+
- Ollama (for LLM features)
- Git

### Quick Start

```bash
# Clone repository
git clone https://github.com/Bekalu1997et/Procurewise-ai.git
cd Procurewise-ai

# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Start backend
cd backend && python main.py &

# Start frontend
streamlit run frontend/streamlit_app.py
```

## Production Deployment

### Docker Deployment (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Manual Deployment

1. **Backend**: Deploy FastAPI on cloud service (AWS, GCP, Azure)
2. **Frontend**: Deploy Streamlit on Streamlit Cloud or similar
3. **Database**: Use managed SQLite or PostgreSQL
4. **LLM**: Deploy Ollama on dedicated server

## Environment Variables

Create `.env` file:

```
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

## Features

✅ **Complete ML Pipeline**
- XGBoost risk prediction model
- SHAP explainability
- MLflow experiment tracking

✅ **Production Backend**
- FastAPI with async support
- SQLite database integration
- Comprehensive API endpoints

✅ **Interactive Frontend**
- Streamlit multipage application
- Real-time predictions
- SHAP visualizations

✅ **AI Integration**
- Mistral 7B LLM via Ollama
- Contextual recommendations
- Interactive chat interface

✅ **Testing Framework**
- Unit tests for all components
- Integration testing
- API endpoint validation

## Support

For issues and questions, please create an issue on GitHub:
https://github.com/Bekalu1997et/Procurewise-ai/issues