"""
Model training module for procurement risk prediction.
Handles XGBoost model training, evaluation, and MLflow experiment tracking.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import mlflow
import mlflow.xgboost
import pickle
import json
from pathlib import Path
import config
from src.preprocessing import preprocess_data


def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


def train_xgboost_model(X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray) -> xgb.XGBClassifier:
    """Train XGBoost classifier with the given data."""
    
    # Initialize model
    model = xgb.XGBClassifier(**config.XGBOOST_PARAMS)
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model performance and return metrics."""
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': y_pred.tolist(),
        'prediction_probabilities': y_pred_proba.tolist()
    }
    
    return metrics


def save_model(model: xgb.XGBClassifier, feature_names: list):
    """Save trained model and metadata."""
    
    # Save model
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names
    feature_names_path = config.MODELS_DIR / "feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    
    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Feature names saved to {feature_names_path}")


def load_model() -> tuple:
    """Load trained model and feature names."""
    
    # Load model
    with open(config.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # Load feature names
    feature_names_path = config.MODELS_DIR / "feature_names.json"
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    return model, feature_names


def train_and_evaluate_model(data_path: str = None) -> dict:
    """
    Complete training pipeline with MLflow tracking.
    Returns trained model and evaluation metrics.
    """
    
    # Setup MLflow
    setup_mlflow()
    
    with mlflow.start_run():
        
        # Preprocess data
        print("Preprocessing data...")
        data = preprocess_data(data_path)
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        feature_names = data['feature_names']
        
        # Log data info
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Log model parameters
        for param, value in config.XGBOOST_PARAMS.items():
            mlflow.log_param(param, value)
        
        # Train model
        print("Training XGBoost model...")
        model = train_xgboost_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Save model locally
        save_model(model, feature_names)
        
        print(f"Training completed!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_names': feature_names
        }


def predict_risk(input_data: dict) -> dict:
    """
    Make risk prediction for single input.
    Returns prediction and probability scores.
    """
    
    # Load model and target encoder
    model, feature_names = load_model()
    
    with open(config.ENCODER_PATH, 'rb') as f:
        target_encoder = pickle.load(f)
    
    # Preprocess input
    from src.preprocessing import preprocess_single_input
    X_processed = preprocess_single_input(input_data)
    
    # Make prediction
    prediction_encoded = model.predict(X_processed)[0]
    probabilities = model.predict_proba(X_processed)[0]
    
    # Decode prediction
    prediction = target_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get class labels
    classes = target_encoder.inverse_transform(model.classes_)
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    
    return {
        'prediction': prediction,
        'probabilities': prob_dict,
        'confidence': float(max(probabilities))
    }


def batch_predict_risk(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make risk predictions for batch input.
    Returns DataFrame with predictions and probabilities.
    """
    
    # Load model
    model, feature_names = load_model()
    
    results = []
    
    for _, row in input_df.iterrows():
        try:
            # Convert row to dict
            input_data = row.to_dict()
            
            # Make prediction
            result = predict_risk(input_data)
            result['supplier_id'] = input_data.get('supplier_id', 'Unknown')
            results.append(result)
            
        except Exception as e:
            # Handle prediction errors
            results.append({
                'supplier_id': input_data.get('supplier_id', 'Unknown'),
                'prediction': 'Error',
                'probabilities': {},
                'confidence': 0.0,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Train model if run directly
    result = train_and_evaluate_model()
    print("Training pipeline completed successfully!")