"""
LLM endpoints for AI-powered recommendations and chat interface.
"""
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Any, Optional
from backend.models.api_models import PredictionInput
from backend.services.llm_agent import ProcurementLLMAgent
from backend.services.shap_explainer import SHAPExplainer
from backend.services.prediction_service import PredictionService

router = APIRouter(prefix="/llm", tags=["llm"])

# Initialize services
llm_agent = ProcurementLLMAgent()
try:
    shap_explainer = SHAPExplainer()
except Exception as e:
    print(f"Warning: SHAP explainer initialization failed: {e}")
    shap_explainer = None

prediction_service = PredictionService()

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
    Generate natural language explanation of prediction using LLM.
    
    Combines model prediction with SHAP analysis for comprehensive explanation.
    """
    try:
        input_dict = input_data.dict()
        
        # Get prediction
        prediction_result = prediction_service.predict_single(input_data, "llm_session", None)
        
        # Get SHAP explanation
        shap_explanation = shap_explainer.explain_prediction(input_dict)
        
        if 'error' in shap_explanation:
            raise HTTPException(status_code=500, detail=shap_explanation['error'])
        
        # Generate LLM explanation
        llm_explanation = llm_agent.generate_prediction_explanation(
            prediction_result.__dict__ if hasattr(prediction_result, '__dict__') else prediction_result,
            shap_explanation,
            input_dict
        )
        
        return {
            "success": True,
            "data": {
                "prediction": prediction_result,
                "shap_explanation": shap_explanation,
                "llm_explanation": llm_explanation
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate LLM explanation: {str(e)}")

@router.post("/recommendations")
async def generate_recommendations(request: RecommendationRequest):
    """
    Generate AI-powered recommendations based on prediction and SHAP analysis.
    
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
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@router.post("/chat")
async def chat_with_llm(chat_request: ChatMessage):
    """
    Interactive chat with LLM about procurement risks and predictions.
    
    Maintains conversation context and provides contextual responses.
    """
    try:
        session_id = chat_request.session_id or "default_session"
        
        # Get context if requested
        context = None
        if chat_request.include_context:
            # This would typically include recent predictions, but for now we'll use None
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
                "session_id": session_id
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@router.post("/what-if")
async def analyze_what_if_scenario(request: WhatIfRequest):
    """
    Analyze what-if scenarios using LLM.
    
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
                                       if request.original_data.get(k) != v])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-if analysis failed: {str(e)}")

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
                "message": f"Chat history cleared for session {session_id}"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")

@router.get("/health")
async def llm_health_check():
    """
    Check LLM service health and Ollama connectivity.
    
    Returns status of Ollama connection and model availability.
    """
    try:
        # Test Ollama connection with a simple prompt
        test_response = llm_agent._call_ollama("Hello, are you working?")
        
        if "Error" in test_response:
            return {
                "success": False,
                "status": "unhealthy",
                "error": test_response
            }
        
        return {
            "success": True,
            "status": "healthy",
            "ollama_url": llm_agent.ollama_url,
            "model": llm_agent.model_name,
            "test_response": test_response[:100] + "..." if len(test_response) > 100 else test_response
        }
        
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e)
        }