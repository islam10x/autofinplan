from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Import your existing models and services
from models.user_profile import UserProfile
from services.rl_training_service import RLTrainingService
from Agents.Hybrid_RL_LLM import HybridLLMRLAgent
from ai.LLM_client import LLMClient

# Import the classes from your original financial advisor code
from pydantic import BaseModel, Field
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import requests
import pandas as pd
import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.accounts_get_request import AccountsGetRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced RL + LLM Financial Planning Service",
    description="Advanced AI-powered financial advisor combining RL agents, LLM reasoning, and real-time market data",
    version="2.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from services.trending_stocks_service import TrendingStocksMixin


# Bank Data Extractor Class 
class BankDataExtractor:
    """Extract financial data from bank accounts using Plaid API"""
    
    def __init__(self):
        self.client_id = os.getenv("PLAID_CLIENT_ID")
        self.secret = os.getenv("PLAID_SECRET")
        self.environment = os.getenv("PLAID_ENV", "sandbox")
        
        if self.client_id and self.secret:
            configuration = plaid.Configuration(
                host=getattr(plaid.Environment, self.environment.capitalize()),
                api_key={
                    'clientId': self.client_id,
                    'secret': self.secret
                }
            )
            api_client = plaid.ApiClient(configuration)
            self.client = plaid_api.PlaidApi(api_client)
        else:
            self.client = None
            logger.warning("Plaid credentials not configured")
    
    async def extract_financial_data(self, access_token: str) -> dict:
        """Extract comprehensive financial data from user's bank account"""
        if not self.client:
            return {"error": "Plaid not configured"}
            
        try:
            # Get accounts
            accounts_request = AccountsGetRequest(access_token=access_token)
            accounts_response = self.client.accounts_get(accounts_request)
            accounts = accounts_response['accounts']
            
            # Calculate total assets
            total_assets = sum(acc['balances']['current'] for acc in accounts if acc['balances']['current'])
            
            # Get recent transactions (last 90 days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)
            
            transactions_request = TransactionsGetRequest(
                access_token=access_token,
                start_date=start_date,
                end_date=end_date
            )
            transactions_response = self.client.transactions_get(transactions_request)
            transactions = transactions_response['transactions']
            
            # Analyze transactions
            monthly_expenses = self._calculate_monthly_expenses(transactions)
            monthly_income = self._calculate_monthly_income(transactions)
            debt_payments = self._calculate_debt_payments(transactions)
            
            return {
                "assets": total_assets,
                "monthly_income": monthly_income,
                "monthly_expenses": monthly_expenses,
                "estimated_debt": debt_payments * 12,
                "accounts_summary": [dict(name=acc['name'], type=acc['type'], balance=acc['balances']['current']) for acc in accounts]

            }
            
        except Exception as e:
            logger.error(f"Bank data extraction failed: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_monthly_expenses(self, transactions) -> float:
        expenses = [t['amount'] for t in transactions if t['amount'] > 0]
        return sum(expenses) / 3 if expenses else 0
    
    def _calculate_monthly_income(self, transactions) -> float:
        income = [abs(t['amount']) for t in transactions if t['amount'] < 0 and t['amount'] < -1000]
        return sum(income) / 3 if income else 0
    
    def _calculate_debt_payments(self, transactions) -> float:
        debt_keywords = ['loan', 'credit', 'mortgage', 'card', 'payment']
        debt_payments = [
            t['amount'] for t in transactions 
            if t['amount'] > 0 and any(keyword in (t.get('merchant_name') or '').lower() for keyword in debt_keywords)
        ]
        return sum(debt_payments) / 3 if debt_payments else 0

# Trending Investment Analyzer Class
class TrendingInvestmentAnalyzer(TrendingStocksMixin):
    def __init__(self):
        super().__init__()
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")

    async def get_trending_investments(self, risk_tolerance: str = "moderate") -> dict:
        try:
            trending_data = {}
            trending_stocks = await self._get_trending_stocks()
            trending_data["stocks"] = trending_stocks
            crypto_trends = await self._get_crypto_trends()
            trending_data["crypto"] = crypto_trends
            sector_analysis = await self._get_sector_analysis()
            trending_data["sectors"] = sector_analysis
            filtered_recommendations = self._filter_by_risk_tolerance(trending_data, risk_tolerance)
            return {
                "recommendations": filtered_recommendations,
                "analysis_timestamp": datetime.now().isoformat(),
                "risk_level": risk_tolerance
            }
        except Exception as e:
            logger.error(f"Trending investment analysis failed: {str(e)}")
            return {"error": str(e)}
    
    async def _get_crypto_trends(self) -> list:
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "percent_change_24h_desc",
                "per_page": 10,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h,7d"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        "symbol": coin["symbol"].upper(),
                        "name": coin["name"],
                        "current_price": coin["current_price"],
                        "24h_change_percent": coin["price_change_percentage_24h"],
                        "7d_change_percent": coin.get("price_change_percentage_7d_in_currency", 0),
                        "market_cap": coin["market_cap"],
                        "volume": coin["total_volume"]
                    }
                    for coin in data[:5]
                ]
            return []
            
        except Exception as e:
            logger.error(f"Crypto trend analysis failed: {str(e)}")
            return []
    
    async def _get_sector_analysis(self) -> list:
        try:
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financial": "XLF",
                "Energy": "XLE",
                "Consumer": "XLY"
            }
            
            sector_performance = []
            for sector, etf in sector_etfs.items():
                ticker = yf.Ticker(etf)
                hist = ticker.history(period="30d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    month_change = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    
                    sector_performance.append({
                        "sector": sector,
                        "etf_symbol": etf,
                        "30day_performance": float(month_change),
                        "current_price": float(current_price)
                    })
            
            sector_performance.sort(key=lambda x: x["30day_performance"], reverse=True)
            return sector_performance
            
        except Exception as e:
            logger.error(f"Sector analysis failed: {str(e)}")
            return []
    
    def _filter_by_risk_tolerance(self, trending_data: dict, risk_tolerance: str) -> dict:
        if risk_tolerance == "conservative":
            stocks = [s for s in trending_data.get("stocks", []) if s.get("market_cap", 0) > 100_000_000_000]
            crypto = []
            sectors = [s for s in trending_data.get("sectors", []) if s["sector"] in ["Healthcare", "Consumer"]]
        elif risk_tolerance == "aggressive":
            stocks = trending_data.get("stocks", [])
            crypto = trending_data.get("crypto", [])
            sectors = trending_data.get("sectors", [])
        else:  # moderate
            stocks = trending_data.get("stocks", [])[:3]
            crypto = trending_data.get("crypto", [])[:2]
            sectors = trending_data.get("sectors", [])
        
        return {
            "stocks": stocks,
            "crypto": crypto,
            "sectors": sectors
        }

# Email Notification Service
class EmailNotificationService:
    """Send email notifications to users"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("EMAIL_USER")
        self.email_password = os.getenv("EMAIL_PASSWORD")
    
    async def send_investment_alert(self, user_email: str, investment_data: dict):
        if not self.email_user or not self.email_password:
            return {"status": "error", "message": "Email not configured"}
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = user_email
            msg['Subject'] = "🚀 AI Financial Advisor - Investment Alert"
            
            html_body = self._create_investment_email_html(investment_data)
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            return {"status": "success", "message": "Investment alert sent successfully"}
            
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def send_financial_plan_email(self, user_email: str, financial_plan: dict):
        if not self.email_user or not self.email_password:
            return {"status": "error", "message": "Email not configured"}
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = user_email
            msg['Subject'] = "📊 Your AI-Generated Financial Plan"
            
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2>Your Personalized Financial Plan</h2>
                    <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
{str(financial_plan)}
                    </pre>
                    <p><em>Generated by your AI Financial Advisor</em></p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            return {"status": "success", "message": "Financial plan sent successfully"}
            
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _create_investment_email_html(self, investment_data: dict) -> str:
        recommendations = investment_data.get("recommendations", {})
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c3e50;">🚀 Trending Investment Opportunities</h2>
                <p>AI-powered investment recommendations based on real-time market analysis:</p>
        """
        
        # Add stocks section
        if recommendations.get("stocks"):
            html += """
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="color: #e74c3c;">📈 Top Performing Stocks</h3>
                    <ul>
            """
            for stock in recommendations.get("stocks", [])[:3]:
                html += f"""
                        <li><strong>{stock['symbol']}</strong> - {stock['name']}: 
                        ${stock['current_price']:.2f} ({stock['5day_change_percent']:.2f}% 5-day change)</li>
                """
            html += "</ul></div>"
        
        # Add crypto section
        if recommendations.get("crypto"):
            html += """
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="color: #856404;">₿ Trending Cryptocurrencies</h3>
                    <ul>
            """
            for crypto in recommendations.get("crypto", [])[:2]:
                html += f"""
                        <li><strong>{crypto['symbol']}</strong> - {crypto['name']}: 
                        ${crypto['current_price']:.4f} ({crypto['24h_change_percent']:.2f}% 24h change)</li>
                """
            html += "</ul></div>"
        
        html += f"""
                <p style="font-size: 12px; color: #666; margin-top: 30px;">
                    <em>Analysis generated at {investment_data.get('analysis_timestamp', 'N/A')}</em>
                </p>
                <p style="font-size: 12px; color: #666;">
                    <strong>Disclaimer:</strong> This is not financial advice. Consult with a financial advisor.
                </p>
            </div>
        </body>
        </html>
        """
        
        return html

# Initialize services
try:
    rl_service = RLTrainingService()
    llm_client = LLMClient(os.getenv("OPENAI_API_KEY", ""))
    hybrid_agent = HybridLLMRLAgent(llm_client, rl_service.agents) if hasattr(rl_service, 'agents') else None
except Exception as e:
    logger.error(f"Failed to initialize RL/LLM services: {str(e)}")
    rl_service = None
    llm_client = None
    hybrid_agent = None

# Initialize other services
bank_extractor = BankDataExtractor()
investment_analyzer = TrendingInvestmentAnalyzer()
email_service = EmailNotificationService()

# Scheduler for periodic tasks
scheduler = AsyncIOScheduler()

# In-memory storage (replace with database in production)
user_profiles = {}

@app.on_event("startup")
async def startup_event():
    scheduler.start()
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    logger.info("Application shutdown")

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "rl_service": "operational" if rl_service else "not_available",
            "llm_client": "operational" if llm_client else "not_available",
            "bank_data": "operational" if bank_extractor.client else "not_configured",
            "email": "operational" if email_service.email_user else "not_configured",
            "scheduler": "running" if scheduler.running else "stopped"
        }
    }

# RL + LLM Endpoints
@app.post("/hybrid-plan/")
async def generate_hybrid_plan(profile: UserProfile):
    """Generate plan using both RL agents and LLM reasoning"""
    if not hybrid_agent:
        raise HTTPException(status_code=503, detail="Hybrid agent not available")
        
    try:
        user_dict = profile.dict()
        hybrid_plan = await hybrid_agent.generate_hybrid_recommendation(user_dict)
        
        return {
            "status": "success",
            "financial_plan": hybrid_plan,
            "methodology": "reinforcement_learning + large_language_model"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid planning failed: {str(e)}")

@app.post("/train-rl-agents/")
async def train_rl_agents(background_tasks: BackgroundTasks, timesteps: int = 50000):
    """Trigger RL agent training (runs in background)"""
    if not rl_service:
        raise HTTPException(status_code=503, detail="RL service not available")
        
    background_tasks.add_task(rl_service.train_all_agents, timesteps)
    return {"message": f"RL training started with {timesteps} timesteps"}

@app.get("/rl-status/")
async def rl_status():
    """Check RL agent training status"""
    if not rl_service:
        return {"error": "RL service not available"}
        
    return {
        "agents": list(rl_service.agents.keys()) if hasattr(rl_service, 'agents') else [],
        "trained": [name for name, agent in rl_service.agents.items() if hasattr(agent, 'model') and agent.model is not None] if hasattr(rl_service, 'agents') else [],
        "algorithms": {
            "portfolio": "PPO",
            "debt": "TD3"
        }
    }

@app.post("/rl-only-plan/")
async def generate_rl_only_plan(profile: UserProfile):
    """Generate plan using only RL agents (no LLM)"""
    if not rl_service:
        raise HTTPException(status_code=503, detail="RL service not available")
        
    try:
        user_dict = profile.dict()
        
        rl_recommendations = {}
        if hasattr(rl_service, 'agents') and 'portfolio' in rl_service.agents:
            rl_recommendations['portfolio'] = rl_service.agents['portfolio'].analyze_portfolio(user_dict)
        if hasattr(rl_service, 'agents') and 'debt' in rl_service.agents:
            rl_recommendations['debt'] = rl_service.agents['debt'].analyze_debt_strategy(user_dict)
        
        return {
            "status": "success",
            "financial_plan": rl_recommendations,
            "methodology": "pure_reinforcement_learning"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL planning failed: {str(e)}")

# Enhanced endpoints with auto-extraction and trending analysis
@app.post("/auto-extract-profile/")
async def auto_extract_user_profile(
    email: str,
    age: int,
    plaid_access_token: str,
    risk_tolerance: str = "moderate",
    financial_goals: Optional[str] = None
):
    """Automatically extract user financial data from bank accounts"""
    try:
        # Extract bank data
        bank_data = await bank_extractor.extract_financial_data(plaid_access_token)
        
        if "error" in bank_data:
            raise HTTPException(status_code=400, detail=f"Bank data extraction failed: {bank_data['error']}")
        
        # Create enhanced user profile
        enhanced_profile = UserProfile(
            email=email,
            income=bank_data.get("monthly_income"),
            expenses=bank_data.get("monthly_expenses"),
            debt=bank_data.get("estimated_debt"),
            age=age,
            assets=bank_data.get("assets"),
            risk_tolerance=risk_tolerance,
            financial_goals=financial_goals,
            plaid_access_token=plaid_access_token
        )
        
        # Store user profile
        user_profiles[email] = enhanced_profile.dict()
        
        return {
            "status": "success",
            "message": "Financial data extracted successfully",
            "user_profile": enhanced_profile.dict(),
            "bank_summary": bank_data.get("accounts_summary", []),
            "extraction_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-extraction failed: {str(e)}")

@app.get("/trending-investments/")
async def get_trending_investments(risk_tolerance: str = "moderate"):
    """Get real-time trending investment recommendations"""
    try:
        trending_data = await investment_analyzer.get_trending_investments(risk_tolerance)
        return {
            "status": "success",
            "trending_investments": trending_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investment analysis failed: {str(e)}")

@app.post("/send-investment-alert/")
async def send_investment_alert(user_email: str, risk_tolerance: str = "moderate"):
    """Send trending investment alert via email"""
    try:
        # Get trending investments
        investment_data = await investment_analyzer.get_trending_investments(risk_tolerance)
        
        # Send email
        email_result = await email_service.send_investment_alert(user_email, investment_data)
        
        return {
            "status": "success",
            "message": "Investment alert sent successfully",
            "email_status": email_result,
            "investment_data": investment_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert sending failed: {str(e)}")

@app.post("/setup-daily-alerts/")
async def setup_daily_investment_alerts(user_email: str, risk_tolerance: str = "moderate", hour: int = 9):
    """Setup daily investment alerts for a user"""
    try:
        # Schedule daily job
        scheduler.add_job(
            send_daily_investment_alert,
            'cron',
            hour=hour,
            args=[user_email, risk_tolerance],
            id=f"daily_alert_{user_email}",
            replace_existing=True
        )
        
        return {
            "status": "success",
            "message": f"Daily investment alerts scheduled for {user_email} at {hour}:00",
            "next_alert": f"Tomorrow at {hour}:00"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert setup failed: {str(e)}")

async def send_daily_investment_alert(user_email: str, risk_tolerance: str):
    """Background task for sending daily investment alerts"""
    try:
        investment_data = await investment_analyzer.get_trending_investments(risk_tolerance)
        await email_service.send_investment_alert(user_email, investment_data)
        logger.info(f"Daily investment alert sent to {user_email}")
    except Exception as e:
        logger.error(f"Daily alert failed for {user_email}: {str(e)}")

@app.post("/enhanced-hybrid-plan/")
async def generate_enhanced_hybrid_plan(profile: UserProfile):
    """Generate enhanced financial plan with auto-extracted data and trending investments"""
    try:
        user_dict = profile.dict()
        
        # If bank token provided, extract fresh data
        if profile.plaid_access_token:
            bank_data = await bank_extractor.extract_financial_data(profile.plaid_access_token)
            if "error" not in bank_data:
                # Update profile with fresh bank data
                for key, value in bank_data.items():
                    if key in user_dict and value is not None:
                        user_dict[key] = value
        
        # Get trending investments for the user's risk tolerance
        trending_investments = await investment_analyzer.get_trending_investments(profile.risk_tolerance)
        
        # Generate hybrid plan if available
        hybrid_plan = None
        if hybrid_agent:
            try:
                hybrid_plan = await hybrid_agent.generate_hybrid_recommendation(user_dict)
            except Exception as e:
                logger.warning(f"Hybrid agent failed, using fallback: {str(e)}")
        
        # Create enhanced plan
        enhanced_plan = {
            "user_profile": user_dict,
            "trending_opportunities": trending_investments,
            "hybrid_ai_analysis": hybrid_plan,
            "recommendations": {
                "immediate_actions": [
                    "Review and optimize current portfolio allocation",
                    "Consider trending investment opportunities based on risk tolerance",
                    "Automate savings and investment contributions"
                ],
                "trending_investments": trending_investments.get("recommendations", {}),
                "risk_assessment": f"Based on {profile.risk_tolerance} risk tolerance"
            },
            "next_review_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "methodology": "hybrid_rl_llm_trending_analysis" if hybrid_plan else "trending_analysis_only"
        }
        
        # Send email notification
        email_sent = False
        if profile.email:
            email_result = await email_service.send_financial_plan_email(profile.email, enhanced_plan)
            email_sent = email_result.get("status") == "success"
        
        return {
            "status": "success",
            "enhanced_financial_plan": enhanced_plan,
            "email_sent": email_sent
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced planning failed: {str(e)}")

# User management endpoints
@app.post("/users/register/")
async def register_user(profile: UserProfile):
    """Register a new user"""
    user_profiles[profile.email] = profile.dict()
    return {
        "status": "success",
        "message": "User registered successfully",
        "user_email": profile.email
    }

@app.get("/users/{email}/profile/")
async def get_user_profile(email: str):
    """Get user profile"""
    if email not in user_profiles:
        raise HTTPException(status_code=404, detail="User not found")
    return user_profiles[email]

@app.put("/users/{email}/profile/")
async def update_user_profile(email: str, profile: UserProfile):
    """Update user profile"""
    if email not in user_profiles:
        raise HTTPException(status_code=404, detail="User not found")
    user_profiles[email] = profile.dict()
    return {
        "status": "success",
        "message": "Profile updated successfully"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)