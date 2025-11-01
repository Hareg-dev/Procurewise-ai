"""
Configuration settings for the Procurement Risk Prediction Platform
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data files
MERGED_TRAINING_DATA = DATA_DIR / "merged_training.csv"
SUPPLIERS_DATA = DATA_DIR / "suppliers.csv"
TRANSACTIONS_DATA = DATA_DIR / "transactions.csv"
CONTRACTS_DATA = DATA_DIR / "contracts.csv"

# Model settings
MODEL_NAME = "xgboost_risk_predictor"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.pkl"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# MLflow settings
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "procurement_risk_prediction"

# Model hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Feature engineering settings
NUMERICAL_FEATURES = [
    'annual_revenue', 'annual_spend', 'avg_payment_delay_days',
    'contract_value', 'contract_duration_months', 'past_disputes',
    'delivery_score', 'financial_stability_index', 'relationship_years',
    'credit_score', 'txn_count', 'avg_txn_amount', 'avg_delay',
    'late_ratio', 'dispute_rate', 'avg_delivery_quality', 'clause_risk_score'
]

CATEGORICAL_FEATURES = [
    'region', 'industry', 'contract_criticality'
]

TARGET_COLUMN = 'risk_label'

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Streamlit settings
STREAMLIT_PORT = 8501

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"