"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class PredictionInput(BaseModel):
    """Single prediction input model."""
    annual_revenue: float = Field(..., gt=0, description="Annual revenue in USD")
    annual_spend: float = Field(..., gt=0, description="Annual spend in USD")
    avg_payment_delay_days: float = Field(..., ge=0, description="Average payment delay in days")
    contract_value: float = Field(..., gt=0, description="Contract value in USD")
    contract_duration_months: int = Field(..., gt=0, le=120, description="Contract duration in months")
    past_disputes: int = Field(..., ge=0, description="Number of past disputes")
    delivery_score: float = Field(..., ge=0, le=100, description="Delivery performance score")
    financial_stability_index: float = Field(..., ge=0, le=1, description="Financial stability index")
    relationship_years: int = Field(..., ge=0, description="Years of relationship")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    txn_count: float = Field(..., ge=0, description="Transaction count")
    avg_txn_amount: float = Field(..., ge=0, description="Average transaction amount")
    avg_delay: float = Field(..., ge=0, description="Average delay")
    late_ratio: float = Field(..., ge=0, le=1, description="Late payment ratio")
    dispute_rate: float = Field(..., ge=0, le=1, description="Dispute rate")
    avg_delivery_quality: float = Field(..., ge=0, le=100, description="Average delivery quality")
    clause_risk_score: float = Field(..., ge=0, le=100, description="Contract clause risk score")
    region: str = Field(..., description="Geographic region")
    industry: str = Field(..., description="Industry sector")
    contract_criticality: str = Field(..., description="Contract criticality level")

class PredictionResponse(BaseModel):
    """Single prediction response model."""
    prediction: str = Field(..., description="Risk level prediction")
    confidence: float = Field(..., description="Prediction confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    shap_values: Optional[Dict[str, float]] = Field(None, description="SHAP feature importance")
    model_version: str = Field(..., description="Model version used")
    processing_time: float = Field(..., description="Processing time in seconds")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    batch_id: str = Field(..., description="Unique batch identifier")
    total_records: int = Field(..., description="Total number of records processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    risk_distribution: Dict[str, int] = Field(..., description="Distribution of risk levels")
    avg_confidence: float = Field(..., description="Average confidence score")
    processing_time: float = Field(..., description="Total processing time")
    results: List[Dict[str, Any]] = Field(..., description="Individual prediction results")

class HistoryResponse(BaseModel):
    """Prediction history response model."""
    id: int
    prediction_type: str
    risk_level: str
    confidence_score: float
    created_at: datetime
    processing_time: float

class SessionRequest(BaseModel):
    """Session creation request."""
    user_id: Optional[str] = Field(None, description="Optional user identifier")

class SessionResponse(BaseModel):
    """Session creation response."""
    session_id: str = Field(..., description="Generated session identifier")
    created_at: datetime = Field(..., description="Session creation timestamp")