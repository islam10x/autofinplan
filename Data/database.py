
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env if available

# Use Supabase PostgreSQL URL (replace with your actual URL or set in .env)
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Please set it in the environment or .env file.")

# Optional fix if Supabase gives you a 'postgres://' URI (SQLAlchemy wants 'postgresql://')
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    income = Column(Float, nullable=True)
    expenses = Column(Float, nullable=True)
    debt = Column(Float, nullable=True)
    age = Column(Integer)
    assets = Column(Float, nullable=True)
    risk_tolerance = Column(String, default="moderate")
    financial_goals = Column(Text, nullable=True)
    plaid_access_token = Column(String, nullable=True)
    bank_account_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class InvestmentAlert(Base):
    __tablename__ = "investment_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    alert_data = Column(Text)  # JSON string
    sent_at = Column(DateTime, default=datetime.utcnow)

# Create tables in Supabase
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
