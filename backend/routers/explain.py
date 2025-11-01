"""
Explainability endpoints for SHAP analysis and model interpretation.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.utils.db import get_db
from backend.models.api_models import PredictionInput
from backend.services.shap_explainer import SHAPExplainer
from backend.services.prediction_service import PredictionService
from typing import Dict, Any

router = APIRouter(prefix="/explain", tags=["explainability"])

# Initialize services
try:
    shap_explainer = SHAPExplainer()
except Exception as e:
    print(f"Warning: SHAP explainer initialization failed: {e}")
    shap_explainer = None

prediction_service = PredictionService()

@router.get("/global-importance")
async def get_global_feature_importance(sample_size: int = 100):
    """
    Get global feature importance using SHAP analysis.
    
    Returns the most important features across all predictions.
    """
    try:
        if shap_explainer is None:
            raise HTTPException(status_code=503, detail="SHAP explainer not available")
            
        result = shap_explainer.get_global_feature_importance(sample_size)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate global importance: {str(e)}")

@router.post("/prediction")
async def explain_prediction(input_data: PredictionInput):
    """
    Generate SHAP explanation for a specific prediction.
    
    Returns feature impacts and human-readable explanations.
    """
    try:
        # Convert to dict
        input_dict = input_data.dict()
        
        # Generate SHAP explanation
        shap_result = shap_explainer.explain_prediction(input_dict)
        
        if 'error' in shap_result:
            raise HTTPException(status_code=500, detail=shap_result['error'])
        
        # Generate summary
        summary = shap_explainer.get_explanation_summary(shap_result)
        
        return {
            "success": True,
            "data": {
                "shap_explanation": shap_result,
                "summary": summary
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to explain prediction: {str(e)}")

@router.post("/visualize")
async def generate_shap_visualizations(input_data: PredictionInput):
    """
    Generate SHAP visualization plots for a prediction.
    
    Returns base64-encoded plot images.
    """
    try:
        input_dict = input_data.dict()
        
        # Generate plots
        force_plot = shap_explainer.generate_force_plot(input_dict)
        summary_plot = shap_explainer.generate_summary_plot()
        
        return {
            "success": True,
            "data": {
                "force_plot": force_plot,
                "summary_plot": summary_plot
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate visualizations: {str(e)}")

@router.get("/summary-plot")
async def get_summary_plot(sample_size: int = 50):
    """
    Generate global SHAP summary plot.
    
    Returns base64-encoded summary plot showing feature importance distribution.
    """
    try:
        plot_data = shap_explainer.generate_summary_plot(sample_size)
        
        if plot_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate summary plot")
        
        return {
            "success": True,
            "data": {
                "plot": plot_data,
                "sample_size": sample_size
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary plot: {str(e)}")

@router.post("/compare")
async def compare_predictions(
    prediction1: PredictionInput,
    prediction2: PredictionInput
):
    """
    Compare SHAP explanations between two predictions.
    
    Useful for what-if analysis and scenario comparison.
    """
    try:
        # Generate explanations for both predictions
        shap1 = shap_explainer.explain_prediction(prediction1.dict())
        shap2 = shap_explainer.explain_prediction(prediction2.dict())
        
        if 'error' in shap1 or 'error' in shap2:
            raise HTTPException(status_code=500, detail="Failed to generate one or both explanations")
        
        # Calculate differences
        impact1 = shap1.get('feature_impact', {})
        impact2 = shap2.get('feature_impact', {})
        
        differences = {}
        for feature in impact1.keys():
            if feature in impact2:
                differences[feature] = impact2[feature] - impact1[feature]
        
        # Sort by absolute difference
        sorted_diffs = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            "success": True,
            "data": {
                "prediction1_shap": shap1,
                "prediction2_shap": shap2,
                "differences": dict(sorted_diffs),
                "top_changes": dict(sorted_diffs[:5])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare predictions: {str(e)}")