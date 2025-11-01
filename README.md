# Procurement Risk Prediction Platform

A scalable machine learning platform for predicting procurement risks using XGBoost, with explainable AI insights and LLM-powered recommendations.

## 🏗️ Architecture

- **Frontend:** Streamlit (multi-page)
- **Backend:** FastAPI (multi-endpoint API)
- **ML Stack:** XGBoost, SHAP, MLflow
- **Database:** SQLite (via SQLAlchemy)
- **LLM:** Mistral 7B (via Ollama)
- **Containerization:** Docker

## 📁 Project Structure

```
procu-risk-prediction/
├── data/                          # Training data
│   ├── merged_training.csv
│   ├── suppliers.csv
│   ├── transactions.csv
│   └── contracts.csv
├── src/                           # Core ML modules
│   ├── __init__.py
│   ├── preprocessing.py           # Data preprocessing & feature engineering
│   └── model_training.py          # XGBoost training & evaluation
├── models/                        # Saved model artifacts (auto-created)
├── logs/                          # Application logs (auto-created)
├── config.py                      # Configuration settings
├── train_model.py                 # Main training script
├── requirements.txt               # Python dependencies
└── README.md
```

## 🚀 Phase 1: Core ML Pipeline (Current)

### Features Implemented:
- ✅ Data preprocessing with feature engineering
- ✅ XGBoost model training with MLflow tracking
- ✅ Model evaluation and artifact saving
- ✅ Single and batch prediction functions
- ✅ Modular, scalable architecture

### Quick Start:

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model:**
```bash
python train_model.py
```

3. **Use the trained model:**
```python
from src.model_training import predict_risk

# Single prediction
input_data = {
    'annual_revenue': 1000000,
    'annual_spend': 500000,
    'region': 'North America',
    'industry': 'IT Services',
    # ... other features
}

result = predict_risk(input_data)
print(f"Risk Level: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## 📊 Model Features

### Input Features:
- **Financial:** annual_revenue, annual_spend, contract_value, credit_score
- **Performance:** delivery_score, avg_payment_delay_days, dispute_rate
- **Relationship:** relationship_years, financial_stability_index
- **Contract:** contract_duration_months, contract_criticality, clause_risk_score
- **Categorical:** region, industry

### Engineered Features:
- Revenue-to-spend ratio
- Contract-to-revenue ratio  
- Payment risk score
- Delivery risk score
- Relationship strength index

### Target:
- **Risk Labels:** low, medium, high

## 🔮 Future Phases

### Phase 2: API Backend
- FastAPI endpoints for predictions
- Database integration with SQLAlchemy
- User session management

### Phase 3: Frontend Dashboard
- Streamlit multi-page application
- SHAP explainability visualizations
- Interactive risk assessment

### Phase 4: LLM Integration
- Ollama-hosted Mistral 7B integration
- Risk recommendation engine
- Chat-based analysis interface

### Phase 5: Production Deployment
- Docker containerization
- CI/CD pipeline
- Monitoring and logging

## 🛠️ Configuration

All settings are centralized in `config.py`:
- File paths and model parameters
- MLflow experiment tracking
- Feature definitions
- API and frontend settings

## 📈 MLflow Tracking

The platform uses MLflow for experiment tracking:
- Model parameters and metrics
- Artifact storage
- Model versioning
- Performance comparison

Access MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 🔧 Development

The codebase is designed for:
- **Modularity:** Each component is independently testable
- **Scalability:** Easy to extend with new features
- **Production-readiness:** Proper error handling and logging
- **Integration:** Ready for FastAPI backend integration