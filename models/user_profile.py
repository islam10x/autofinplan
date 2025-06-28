from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.accounts_get_request import AccountsGetRequest


# Enhanced User Profile
class UserProfile(BaseModel):
    email: str = Field(..., description="User email for notifications")
    income: Optional[float] = Field(None, description="Monthly income (auto-extracted if None)")
    expenses: Optional[float] = Field(None, description="Monthly expenses (auto-extracted if None)")
    debt: Optional[float] = Field(None, description="Total debt (auto-extracted if None)")
    age: int = Field(..., description="Current age")
    assets: Optional[float] = Field(None, description="Current assets (auto-extracted if None)")
    risk_tolerance: str = Field(default="moderate", description="Risk tolerance level")
    financial_goals: Optional[str] = Field(None, description="Financial goals")
    plaid_access_token: Optional[str] = Field(None, description="Plaid access token for bank data")
    bank_account_id: Optional[str] = Field(None, description="Bank account ID")

