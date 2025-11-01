"""
Prediction endpoints for single and batch risk assessment.
"""
import io
import uuid
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header
from sqlalchemy.orm import Session
from backend.utils.db import get_db
from backend.models.api_models import PredictionInput, PredictionResponse, BatchPredictionResponse
from backend.services.prediction_service import PredictionService

router = APIRouter(prefix="/predict", tags=["predictions"])
prediction_service = PredictionService()

def get_session_id(x_session_id: str = Header(None)) -> str:
    """Extract session ID from headers or generate new one."""
    if x_session_id:
        return x_session_id
    return str(uuid.uuid4())

@router.post("/single", response_model=PredictionResponse)
async def predict_single(
    input_data: PredictionInput,
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id)
):
    """
    Make single risk prediction with SHAP explanation.
    
    Returns risk level, confidence score, probabilities, and feature importance.
    """
    try:
        result = prediction_service.predict_single(input_data, session_id, db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    session_id: str = Depends(get_session_id)
):
    """
    Make batch risk predictions from CSV file upload.
    
    Returns summary statistics and individual prediction results.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Make batch predictions
        result = prediction_service.predict_batch(df, session_id, file.filename, db)
        return result
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint for prediction service."""
    try:
        # Verify model is loaded
        if prediction_service.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "feature_count": len(prediction_service.feature_names) if prediction_service.feature_names else 0
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")