"""
History endpoints for retrieving prediction history and session management.
"""
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from backend.utils.db import get_db
from backend.models.db_models import UserSession, PredictionHistory, BatchPredictionSummary
from backend.models.api_models import HistoryResponse, SessionRequest, SessionResponse

router = APIRouter(prefix="/history", tags=["history"])

@router.post("/session", response_model=SessionResponse)
async def create_session(
    request: SessionRequest,
    db: Session = Depends(get_db)
):
    """Create new user session for tracking predictions."""
    try:
        session_id = str(uuid.uuid4())
        
        # Create session record
        session = UserSession(
            session_id=session_id,
            user_id=request.user_id or "anonymous"
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return SessionResponse(
            session_id=session_id,
            created_at=session.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@router.get("/predictions", response_model=List[HistoryResponse])
async def get_prediction_history(
    session_id: str = Query(..., description="Session ID to retrieve history for"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    prediction_type: Optional[str] = Query(None, description="Filter by prediction type (single/batch)"),
    db: Session = Depends(get_db)
):
    """
    Retrieve prediction history for a session.
    
    Returns paginated list of previous predictions with metadata.
    """
    try:
        query = db.query(PredictionHistory).filter(
            PredictionHistory.session_id == session_id
        )
        
        if prediction_type:
            query = query.filter(PredictionHistory.prediction_type == prediction_type)
        
        predictions = query.order_by(desc(PredictionHistory.created_at)).offset(offset).limit(limit).all()
        
        return [
            HistoryResponse(
                id=pred.id,
                prediction_type=pred.prediction_type,
                risk_level=pred.risk_level,
                confidence_score=pred.confidence_score,
                created_at=pred.created_at,
                processing_time=pred.processing_time
            )
            for pred in predictions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@router.get("/batch-summaries")
async def get_batch_summaries(
    session_id: str = Query(..., description="Session ID to retrieve summaries for"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of summaries to return"),
    db: Session = Depends(get_db)
):
    """
    Retrieve batch prediction summaries for a session.
    
    Returns summary statistics for batch predictions.
    """
    try:
        summaries = db.query(BatchPredictionSummary).filter(
            BatchPredictionSummary.session_id == session_id
        ).order_by(desc(BatchPredictionSummary.created_at)).limit(limit).all()
        
        return [
            {
                "batch_id": summary.batch_id,
                "total_records": summary.total_records,
                "risk_distribution": {
                    "high": summary.high_risk_count,
                    "medium": summary.medium_risk_count,
                    "low": summary.low_risk_count
                },
                "avg_confidence": summary.avg_confidence,
                "processing_time": summary.processing_time,
                "file_name": summary.file_name,
                "created_at": summary.created_at
            }
            for summary in summaries
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve batch summaries: {str(e)}")

@router.get("/stats")
async def get_session_stats(
    session_id: str = Query(..., description="Session ID to get statistics for"),
    db: Session = Depends(get_db)
):
    """
    Get aggregated statistics for a session.
    
    Returns total predictions, risk distribution, and average confidence.
    """
    try:
        # Total predictions
        total_predictions = db.query(PredictionHistory).filter(
            PredictionHistory.session_id == session_id
        ).count()
        
        # Risk distribution
        risk_counts = db.query(
            PredictionHistory.risk_level,
            db.func.count(PredictionHistory.id)
        ).filter(
            PredictionHistory.session_id == session_id
        ).group_by(PredictionHistory.risk_level).all()
        
        risk_distribution = {risk: count for risk, count in risk_counts}
        
        # Average confidence
        avg_confidence = db.query(
            db.func.avg(PredictionHistory.confidence_score)
        ).filter(
            PredictionHistory.session_id == session_id
        ).scalar() or 0.0
        
        return {
            "session_id": session_id,
            "total_predictions": total_predictions,
            "risk_distribution": risk_distribution,
            "avg_confidence": float(avg_confidence),
            "prediction_types": {
                "single": db.query(PredictionHistory).filter(
                    PredictionHistory.session_id == session_id,
                    PredictionHistory.prediction_type == "single"
                ).count(),
                "batch": db.query(PredictionHistory).filter(
                    PredictionHistory.session_id == session_id,
                    PredictionHistory.prediction_type == "batch"
                ).count()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session stats: {str(e)}")