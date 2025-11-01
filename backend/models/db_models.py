"""
SQLAlchemy ORM models for procurement risk prediction platform.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class UserSession(Base):
    """User session tracking for prediction history."""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True)
    user_id = Column(String(255), index=True)
    created_at = Column(DateTime, default=func.now())
    last_activity = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<UserSession(session_id='{self.session_id}', user_id='{self.user_id}')>"


class PredictionHistory(Base):
    """Store prediction results and metadata."""
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    prediction_type = Column(String(50))  # 'single' or 'batch'
    input_data = Column(JSON)
    prediction_result = Column(JSON)
    model_version = Column(String(100))
    confidence_score = Column(Float)
    risk_level = Column(String(50))
    shap_values = Column(JSON, nullable=True)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<PredictionHistory(id={self.id}, risk_level='{self.risk_level}')>"


class BatchPredictionSummary(Base):
    """Summary statistics for batch predictions."""
    __tablename__ = "batch_prediction_summary"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    batch_id = Column(String(255), unique=True, index=True)
    total_records = Column(Integer)
    high_risk_count = Column(Integer)
    medium_risk_count = Column(Integer)
    low_risk_count = Column(Integer)
    avg_confidence = Column(Float)
    processing_time = Column(Float)
    file_name = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<BatchPredictionSummary(batch_id='{self.batch_id}', total_records={self.total_records})>"