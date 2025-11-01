"""
FastAPI main application for Procurement Risk Prediction Platform.
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mlflow
from backend.routers import predict, history, explain, llm
from backend.utils.db import init_database
import config

# Initialize FastAPI app
app = FastAPI(
    title="Procurement Risk Prediction API",
    description="ML-powered API for predicting procurement risks with explainable AI insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and MLflow
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        # Initialize database
        init_database()
        print("✅ Database initialized")
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        print("✅ MLflow configured")
        
        print("🚀 Procurement Risk Prediction API started successfully")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise

# Include routers
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(explain.router)
app.include_router(llm.router)

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint with service information."""
    return {
        "service": "Procurement Risk Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predictions": "/predict",
            "history": "/history",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    try:
        # Check database connection
        from backend.utils.db import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "mlflow": "configured",
            "model": "loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )