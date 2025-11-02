"""
Prediction service for handling ML model inference and SHAP explanations.
"""
import time
import uuid
import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from backend.models.db_models import PredictionHistory, BatchPredictionSummary
from backend.models.api_models import PredictionInput, PredictionResponse, BatchPredictionResponse
from src.model_training import predict_risk, load_model, batch_predict_risk
import mlflow
import config

class PredictionService:
    """Service for handling predictions and SHAP explanations."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.explainer = None
        self._load_model()
    
    def _load_model(self):
        """Load trained model and initialize SHAP explainer."""
        try:
            self.model, self.feature_names = load_model()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _get_shap_values(self, input_data: Dict) -> Dict[str, float]:
        """Calculate SHAP values for feature importance."""
        try:
            from src.preprocessing import preprocess_single_input
            X_processed = preprocess_single_input(input_data)
            
            # Initialize explainer if not exists
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(self.model)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_processed)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):  # Multi-class output
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]  # Use medium risk class
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Get first sample
            
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(shap_values):
                    feature_importance[feature] = float(shap_values[i])
            
            return feature_importance
            
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return {}
    
    def predict_single(self, input_data: PredictionInput, session_id: str, db: Session) -> PredictionResponse:
        """Make single prediction with SHAP explanation."""
        start_time = time.time()
        
        # Convert to dict
        input_dict = input_data.dict()
        
        # Make prediction
        result = predict_risk(input_dict)
        
        # Calculate SHAP values (with error handling)
        try:
            shap_values = self._get_shap_values(input_dict)
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            shap_values = {}
        
        processing_time = time.time() - start_time
        
        # Save to database (only if db is provided)
        if db is not None:
            try:
                prediction_record = PredictionHistory(
                    session_id=session_id,
                    prediction_type="single",
                    input_data=input_dict,
                    prediction_result=result,
                    model_version="xgboost_v1",
                    confidence_score=result['confidence'],
                    risk_level=result['prediction'],
                    shap_values=shap_values,
                    processing_time=processing_time
                )
                db.add(prediction_record)
                db.commit()
            except Exception as e:
                print(f"Database save failed: {e}")
        
        # Log to MLflow (with error handling)
        try:
            with mlflow.start_run():
                mlflow.log_param("prediction_type", "single")
                mlflow.log_metric("confidence", result['confidence'])
                mlflow.log_metric("processing_time", processing_time)
        except Exception as e:
            print(f"MLflow logging failed: {e}")
        
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            shap_values=shap_values,
            model_version="xgboost_v1",
            processing_time=processing_time
        )
    
    def predict_batch(self, df: pd.DataFrame, session_id: str, file_name: str, db: Session) -> BatchPredictionResponse:
        """Make batch predictions and return summary."""
        start_time = time.time()
        batch_id = str(uuid.uuid4())
        
        # Make batch predictions
        results_df = batch_predict_risk(df)
        
        # Calculate statistics
        total_records = len(results_df)
        successful_predictions = len(results_df[results_df['prediction'] != 'Error'])
        failed_predictions = total_records - successful_predictions
        
        # Risk distribution
        risk_counts = results_df['prediction'].value_counts().to_dict()
        risk_distribution = {
            'high': risk_counts.get('high', 0),
            'medium': risk_counts.get('medium', 0),
            'low': risk_counts.get('low', 0)
        }
        
        # Average confidence
        valid_confidences = results_df[results_df['prediction'] != 'Error']['confidence']
        avg_confidence = float(valid_confidences.mean()) if len(valid_confidences) > 0 else 0.0
        
        processing_time = time.time() - start_time
        
        # Save to database (only if db is provided)
        if db is not None:
            try:
                # Save batch summary
                batch_summary = BatchPredictionSummary(
                    session_id=session_id,
                    batch_id=batch_id,
                    total_records=total_records,
                    high_risk_count=risk_distribution['high'],
                    medium_risk_count=risk_distribution['medium'],
                    low_risk_count=risk_distribution['low'],
                    avg_confidence=avg_confidence,
                    processing_time=processing_time,
                    file_name=file_name
                )
                db.add(batch_summary)
                
                # Save individual predictions (limit to avoid performance issues)
                for i, (_, row) in enumerate(results_df.iterrows()):
                    if i < 1000:  # Limit to first 1000 records
                        prediction_record = PredictionHistory(
                            session_id=session_id,
                            prediction_type="batch",
                            input_data={"supplier_id": row.get('supplier_id', f'Unknown_{i}')},
                            prediction_result=row.to_dict(),
                            model_version="xgboost_v1",
                            confidence_score=row.get('confidence', 0.0),
                            risk_level=row.get('prediction', 'Error'),
                            processing_time=processing_time / total_records
                        )
                        db.add(prediction_record)
                
                db.commit()
            except Exception as e:
                print(f"Database save failed: {e}")
        
        # Log to MLflow (with error handling)
        try:
            with mlflow.start_run():
                mlflow.log_param("prediction_type", "batch")
                mlflow.log_param("total_records", total_records)
                mlflow.log_metric("avg_confidence", avg_confidence)
                mlflow.log_metric("processing_time", processing_time)
        except Exception as e:
            print(f"MLflow logging failed: {e}")
        
        return BatchPredictionResponse(
            batch_id=batch_id,
            total_records=total_records,
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            risk_distribution=risk_distribution,
            avg_confidence=avg_confidence,
            processing_time=processing_time,
            results=results_df.to_dict('records')
        )