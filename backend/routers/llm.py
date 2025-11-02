"""
LLM endpoints for AI-powered recommendations and chat interface.
"""
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Any, Optional
from backend.models.api_models import PredictionInput
from backend.services.llm_agent import ProcurementLLMAgent

router = APIRouter(prefix="/llm", tags=["llm"])

# Initialize LLM agent
llm_agent = ProcurementLLMAgent()

class ChatMessage(BaseModel):
    """Chat message model."""
    message: str
    session_id: Optional[str] = None
    include_context: bool = True

class RecommendationRequest(BaseModel):
    """Recommendation request model."""
    input_data: Dict[str, Any]
    prediction_result: Dict[str, Any]
    shap_explanation: Dict[str, Any]

class WhatIfRequest(BaseModel):
    """What-if analysis request model."""
    original_data: Dict[str, Any]
    modified_data: Dict[str, Any]
    original_prediction: Dict[str, Any]

@router.post("/explain-prediction")
async def explain_prediction_with_llm(input_data: PredictionInput):
    """
    Generate natural language explanation of prediction using Phi3 model.
    
    Combines model prediction with SHAP analysis for comprehensive explanation.
    """
    try:
        from src.model_training import predict_risk
        from backend.services.shap_explainer import SHAPExplainer
        
        input_dict = input_data.dict()
        
        # Get prediction
        prediction_result = predict_risk(input_dict)
        
        # Get SHAP explanation
        try:
            shap_explainer = SHAPExplainer()
            shap_explanation = shap_explainer.explain_prediction(input_dict)
        except Exception as e:
            shap_explanation = {"error": f"SHAP analysis failed: {str(e)}"}
        
        # Generate Phi3 explanation
        llm_explanation = llm_agent.generate_prediction_explanation(
            prediction_result,
            shap_explanation,
            input_dict
        )
        
        return {
            "success": True,
            "data": {
                "prediction": prediction_result,
                "shap_explanation": shap_explanation,
                "llm_explanation": llm_explanation,
                "model": llm_agent.model_name
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Phi3 explanation unavailable: {str(e)}")

@router.post("/recommendations")
async def generate_recommendations(request: RecommendationRequest):
    """
    Generate Phi3-powered recommendations based on prediction and SHAP analysis.
    
    Returns structured recommendations and action items.
    """
    try:
        recommendations = llm_agent.generate_recommendations(
            request.prediction_result,
            request.shap_explanation,
            request.input_data
        )
        
        return {
            "success": True,
            "data": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Phi3 recommendations unavailable: {str(e)}")

@router.post("/chat")
async def chat_with_llm(chat_request: ChatMessage):
    """
    Interactive chat with Phi3 model about procurement risks and predictions.
    
    Maintains conversation context and provides contextual responses.
    """
    try:
        session_id = chat_request.session_id or "default_session"
        
        # Get context if requested
        context = None
        if chat_request.include_context:
            # Get recent predictions from database for this session
            try:
                from backend.utils.db import get_db
                from backend.models.db_models import PredictionHistory
                from sqlalchemy.orm import Session
                from sqlalchemy import desc
                
                db = next(get_db())
                recent_prediction = db.query(PredictionHistory).filter(
                    PredictionHistory.session_id == session_id
                ).order_by(desc(PredictionHistory.created_at)).first()
                
                if recent_prediction:
                    context = {
                        "prediction": {
                            "prediction": recent_prediction.risk_level,
                            "confidence": recent_prediction.confidence_score
                        },
                        "input_data": recent_prediction.input_data,
                        "shap_explanation": recent_prediction.shap_values or {}
                    }
                db.close()
            except Exception as e:
                print(f"Error getting context: {e}")
                context = {}
        
        response = llm_agent.chat_with_context(
            chat_request.message,
            session_id,
            context
        )
        
        return {
            "success": True,
            "data": {
                "response": response,
                "session_id": session_id,
                "model": llm_agent.model_name,
                "has_context": context is not None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Phi3 chat unavailable: {str(e)}")

@router.post("/what-if")
async def analyze_what_if_scenario(request: WhatIfRequest):
    """
    Analyze what-if scenarios using Phi3 model.
    
    Compares original vs modified data and explains potential impact.
    """
    try:
        analysis = llm_agent.analyze_what_if_scenario(
            request.original_data,
            request.modified_data,
            request.original_prediction
        )
        
        return {
            "success": True,
            "data": {
                "analysis": analysis,
                "changes_detected": len([k for k, v in request.modified_data.items() 
                                       if request.original_data.get(k) != v]),
                "model": llm_agent.model_name
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Phi3 what-if analysis unavailable: {str(e)}")

@router.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    """
    Get conversation history for a session.
    
    Returns previous chat exchanges for context.
    """
    try:
        history = llm_agent.get_conversation_history(session_id)
        
        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "history": history,
                "message_count": len(history)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@router.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    """
    Clear conversation history for a session.
    
    Useful for starting fresh conversations.
    """
    try:
        llm_agent.clear_conversation_history(session_id)
        
        return {
            "success": True,
            "data": {
                "message": f"Chat history cleared for session {session_id}",
                "model": llm_agent.model_name
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")

@router.get("/health")
async def llm_health_check():
    """
    Check Phi3 model health and Ollama connectivity.
    
    Returns status of Ollama connection and Phi3 model availability.
    """
    try:
        # Quick Ollama connectivity check without model inference
        import requests
        response = requests.get(f"{llm_agent.ollama_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            phi3_available = any(llm_agent.model_name in model.get('name', '') for model in models)
            
            if phi3_available:
                return {
                    "success": True,
                    "status": "healthy",
                    "ollama_url": llm_agent.ollama_url,
                    "model": llm_agent.model_name,
                    "note": "Model available but may have memory constraints"
                }
            else:
                return {
                    "success": False,
                    "status": "model_not_found",
                    "error": f"{llm_agent.model_name} model not found",
                    "available_models": [m.get('name') for m in models]
                }
        else:
            raise Exception(f"Ollama API returned status {response.status_code}")
        
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "error": f"Ollama connection failed: {str(e)}",
            "model": llm_agent.model_name,
            "recommendation": "Please ensure Ollama is running and accessible"
        }