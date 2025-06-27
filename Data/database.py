# database.py - Add this file for production database support

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./financial_advisor.db")

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

# Create tables
Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()