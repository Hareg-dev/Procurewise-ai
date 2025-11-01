"""
Database connection and session management utilities.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from backend.models.db_models import Base
import config

# Database setup
DATABASE_URL = f"sqlite:///{config.PROJECT_ROOT}/backend.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)

def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with tables."""
    create_tables()
    print("Database initialized successfully")