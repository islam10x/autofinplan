import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from datetime import datetime, timedelta
from typing import Any, Literal, Optional, List, Dict
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import time
from sqlalchemy.orm import Session
from Data.database import get_db, User
# Import your existing models and services
from models.user_profile import UserProfile
from services.rl_training_service import RLTrainingService
from Agents.Hybrid_RL_LLM import HybridLLMRLAgent, create_hybrid_agent_with_pretrained_models
from ai.LLM_client import LLMClient
import pprint
# Import the classes from your original financial advisor code
from pydantic import BaseModel, EmailStr, Field
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
from pydantic import BaseModel
from typing import Optional
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    "portfolio_model_path": os.getenv("PORTFOLIO_MODEL_PATH", "./trained_models/portfolio_model.zip"),
    "debt_model_path": os.getenv("DEBT_MODEL_PATH", "./trained_models/debt_model.zip"),
    "portfolio_algorithm": os.getenv("PORTFOLIO_ALGORITHM", "PPO")  # PPO or SAC
}
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
class InvestmentAlertRequest(BaseModel):
    user_email: EmailStr
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
class AutoExtractRequest(BaseModel):
    email: str
    age: int
    plaid_access_token: str
    risk_tolerance: Optional[str] = "moderate"
    financial_goals: Optional[str] = None
def safe_get(data: Dict, key: str, default=None):
    """Safely extract and serialize data from nested objects"""
    if not isinstance(data, dict):
        return default
        
    value = data.get(key, default)
    return sanitize_value(value, default)

def sanitize_value(value, default=None):
    """Recursively sanitize values to ensure JSON serializability"""
    try:
        if value is None:
            return default
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, dict):
            return {k: sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [sanitize_value(item) for item in value]
        elif hasattr(value, "to_dict"):
            return sanitize_value(value.to_dict())
        elif hasattr(value, "__dict__"):
            # Handle objects with __dict__ but skip problematic attributes
            obj_dict = {}
            for k, v in value.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    obj_dict[k] = sanitize_value(v)
            return obj_dict
        else:
            # For any other type, convert to string as last resort
            return str(value)
    except Exception as e:
        logger.error(f"Error sanitizing value {type(value)}: {e}")
        return default 
def user_to_dict(user: User) -> Dict[str, Any]:
    """Convert User SQLAlchemy model to dictionary"""
    try:
        user_dict = {}
        for column in user.__table__.columns:
            value = getattr(user, column.name)
            user_dict[column.name] = sanitize_value(value)
        return user_dict
    except Exception as e:
        logger.error(f"Error converting user to dict: {e}")
        return {}
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
            
            # Run all analyses concurrently
            tasks = [
                self._get_trending_stocks(),
                self._get_crypto_trends(),
                self._get_sector_analysis()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            trending_data["stocks"] = results[0] if not isinstance(results[0], Exception) else []
            trending_data["crypto"] = results[1] if not isinstance(results[1], Exception) else []
            trending_data["sectors"] = results[2] if not isinstance(results[2], Exception) else []
            
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
        """Optimized crypto trends with aiohttp"""
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
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            {
                                "symbol": coin["symbol"].upper(),
                                "name": coin["name"],
                                "current_price": float(coin["current_price"]) if coin["current_price"] else 0.0,
                                "24h_change_percent": float(coin["price_change_percentage_24h"]) if coin["price_change_percentage_24h"] else 0.0,
                                "7d_change_percent": float(coin.get("price_change_percentage_7d_in_currency", 0)),
                                "market_cap": int(coin["market_cap"]) if coin["market_cap"] else 0,
                                "volume": int(coin["total_volume"]) if coin["total_volume"] else 0
                            }
                            for coin in data[:5] if coin.get("current_price") is not None
                        ]
            return []
            
        except Exception as e:
            logger.error(f"Crypto trend analysis failed: {str(e)}")
            return []
    
    async def _get_sector_analysis(self) -> list:
        """Optimized sector analysis with concurrent processing"""
        try:
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financial": "XLF",
                "Energy": "XLE",
                "Consumer": "XLY"
            }
            
            # Process sectors concurrently
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=5) as executor:
                tasks = [
                    loop.run_in_executor(executor, self._get_sector_data, sector, etf)
                    for sector, etf in sector_etfs.items()
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                sector_performance = []
                for result in results:
                    if isinstance(result, dict) and result is not None:
                        sector_performance.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Sector analysis error: {str(result)}")
            
            sector_performance.sort(key=lambda x: x["30day_performance"], reverse=True)
            return sector_performance
            
        except Exception as e:
            logger.error(f"Sector analysis failed: {str(e)}")
            return []

    def _get_sector_data(self, sector: str, etf: str) -> Dict:
        """Get data for a single sector ETF"""
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(period="30d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                month_change = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                
                return {
                    "sector": sector,
                    "etf_symbol": etf,
                    "30day_performance": float(month_change),
                    "current_price": float(current_price)
                }
            return None
        except Exception as e:
            logger.warning(f"Failed to get data for {sector} ({etf}): {str(e)}")
            return None
    
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
    
    async def send_investment_alert(self, user_email: str, html_body: str):
        if not self.email_user or not self.email_password:
            return {"status": "error", "message": "Email not configured"}
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = user_email
            msg['Subject'] = "🚀 AI Financial Advisor - Investment Alert"

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

        # ✅ Stocks section
        stocks = recommendations.get("stocks", [])
        if stocks:
            html += """
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="color: #e74c3c;">📈 Top Performing Stocks</h3>
                    <ul>
            """
            for stock in stocks[:3]:
                try:
                    symbol = str(stock.get("symbol", "N/A"))
                    name = str(stock.get("name", "Unknown"))
                    price = float(stock.get("current_price") or 0)
                    change = float(stock.get("5day_change_percent") or 0)

                    html += f"""
                        <li><strong>{symbol}</strong> - {name}: 
                        ${price:.2f} ({change:.2f}% 5-day change)</li>
                    """
                except (ValueError, TypeError):
                    continue
            html += "</ul></div>"

        # ✅ Crypto section
        crypto = recommendations.get("crypto", [])
        if crypto:
            html += """
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="color: #856404;">₿ Trending Cryptocurrencies</h3>
                    <ul>
            """
            for coin in crypto[:2]:
                try:
                    symbol = str(coin.get("symbol", "N/A"))
                    name = str(coin.get("name", "Unknown"))
                    price = float(coin.get("current_price") or 0)
                    change = float(coin.get("24h_change_percent") or 0)

                    html += f"""
                        <li><strong>{symbol}</strong> - {name}: 
                        ${price:.4f} ({change:.2f}% 24h change)</li>
                    """
                except (ValueError, TypeError):
                    continue
            html += "</ul></div>"

        # ✅ Footer
        timestamp = str(investment_data.get('analysis_timestamp', 'N/A'))
        html += f"""
                <p style="font-size: 12px; color: #666; margin-top: 30px;">
                    <em>Analysis generated at {timestamp}</em>
                </p>
                <p style="font-size: 12px; color: #666;">
                    <strong>Disclaimer:</strong> This is not financial advice. Consult with a financial advisor.
                </p>
            </div>
        </body>
        </html>
        """

        return html



llm_client = LLMClient(os.getenv("OPENAI_API_KEY", ""))
# Initialize hybrid agent with pre-trained models
def initialize_hybrid_agent() -> Optional[object]:
    """Initialize hybrid agent with pre-trained models"""
    try:
        # Check if model files exist
        portfolio_model_exists = os.path.exists(MODEL_CONFIG["portfolio_model_path"])
        debt_model_exists = os.path.exists(MODEL_CONFIG["debt_model_path"])
        
        if not portfolio_model_exists and not debt_model_exists:
            logging.warning("No pre-trained models found. Hybrid agent will be disabled.")
            return None
        
        # Create hybrid agent with available models
        hybrid_agent = create_hybrid_agent_with_pretrained_models(
            llm_client=llm_client,
            portfolio_model_path=MODEL_CONFIG["portfolio_model_path"] if portfolio_model_exists else None,
            debt_model_path=MODEL_CONFIG["debt_model_path"] if debt_model_exists else None,
            portfolio_algorithm=MODEL_CONFIG["portfolio_algorithm"]
        )
        
        logging.info(f"Hybrid agent initialized successfully")
        logging.info(f"Portfolio model: {'loaded' if portfolio_model_exists else 'not available'}")
        logging.info(f"Debt model: {'loaded' if debt_model_exists else 'not available'}")
        
        return hybrid_agent
        
    except Exception as e:
        logging.error(f"Failed to initialize hybrid agent: {e}")
        return None
# Initialize services
try:
    rl_service = RLTrainingService()
    hybrid_agent = initialize_hybrid_agent()
except Exception as e:
    logger.error(f"Failed to initialize RL/LLM services: {str(e)}")
    rl_service = None
    hybrid_agent = None

# Initialize other services
bank_extractor = BankDataExtractor()
investment_analyzer = TrendingInvestmentAnalyzer()
email_service = EmailNotificationService()

# Scheduler for periodic tasks
scheduler = AsyncIOScheduler()

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
    
    # Check if already training
    current_state = rl_service.get_training_state()
    if current_state["is_training"]:
        raise HTTPException(
            status_code=409, 
            detail=f"Training already in progress (ID: {current_state['training_id']})"
        )
    
    # Generate unique training ID
    training_id = f"train_{int(time.time())}"
    
    # Start training in background
    background_tasks.add_task(rl_service.train_all_agents, timesteps, training_id)
    
    return {
        "message": f"RL training started with {timesteps} timesteps",
        "training_id": training_id,
        "agents": list(rl_service.agents.keys()),
        "status_endpoint": f"/training-status/{training_id}"
    }

@app.post("/stop-training/")
async def stop_training():
    """Stop the currently running training"""
    success, message = rl_service.stop_training()
    
    if not success:
        raise HTTPException(status_code=404, detail=message)
    
    current_state = rl_service.get_training_state()
    return {
        "message": message,
        "training_id": current_state["training_id"],
        "current_agent": current_state["current_agent"],
        "progress": current_state["progress"]
    }

@app.get("/training-status/")
async def get_training_status():
    """Get current training status"""
    try:
        status = rl_service.get_training_state()
        
        # Calculate elapsed time if training
        if status["start_time"]:
            elapsed_time = time.time() - status["start_time"]
            status["elapsed_time_seconds"] = round(elapsed_time, 2)
            
            # Estimate remaining time based on progress
            if status["progress"] > 0:
                estimated_total_time = elapsed_time / (status["progress"] / 100)
                remaining_time = estimated_total_time - elapsed_time
                status["estimated_remaining_seconds"] = max(0, round(remaining_time, 2))
        
        # Add helpful error context
        if status["status"] == "error" and "error_message" in status:
            if "tqdm and rich" in status["error_message"]:
                status["error_solution"] = "Run: pip install stable-baselines3[extra] or pip install tqdm rich"
            elif "callback" in status["error_message"]:
                status["error_solution"] = "Your agent doesn't support progress callbacks. Training will use fallback method."
        
        return status
    except Exception as e:
        return {
            "error": "Failed to get training status",
            "error_message": str(e),
            "server_time": time.time()
        }

@app.get("/training-status/{training_id}")
async def get_training_status_by_id(training_id: str):
    """Get training status for a specific training ID"""
    status = rl_service.get_training_state()
    
    if status["training_id"] != training_id:
        raise HTTPException(status_code=404, detail="Training ID not found")
    
    return status

@app.delete("/reset-training-state/")
async def reset_training_state():
    """Reset training state (use when training is stuck)"""
    success, message = rl_service.reset_training_state()
    
    if not success:
        raise HTTPException(status_code=409, detail=message)
    
    return {"message": message}

@app.post("/save-models/")
async def save_models(base_path: str = "./trained_models/"):
    """Save all trained models"""
    try:
        saved_models = rl_service.save_models(base_path)
        return {
            "message": f"Saved {len(saved_models)} models",
            "saved_models": saved_models,
            "base_path": base_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save models: {str(e)}")

@app.post("/load-models/")
async def load_models(base_path: str = "./trained_models/"):
    """Load pre-trained models"""
    try:
        loaded_models = rl_service.load_models(base_path)
        return {
            "message": f"Loaded {len(loaded_models)} models",
            "loaded_models": loaded_models,
            "base_path": base_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.post("/retry-training/")
async def retry_training(background_tasks: BackgroundTasks, timesteps: int = 50000):
    """Retry training after fixing errors"""
    current_state = rl_service.get_training_state()
    
    # Only allow retry if previous training had an error
    if current_state["status"] != "error":
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot retry. Current status: {current_state['status']}"
        )
    
    # Reset the state first
    success, message = rl_service.reset_training_state()
    if not success:
        raise HTTPException(status_code=500, detail=message)
    
    # Start new training
    training_id = f"retry_{int(time.time())}"
    background_tasks.add_task(rl_service.train_all_agents, timesteps, training_id)
    
    return {
        "message": f"Retrying RL training with {timesteps} timesteps",
        "training_id": training_id,
        "previous_error": current_state.get("error_message", "Unknown error")
    }

@app.get("/debug-info/")
async def get_debug_info():
    """Get debug information about the service state"""
    try:
        current_state = rl_service.get_training_state()
        return {
            "service_status": "running",
            "training_state": current_state,
            "agents_available": list(rl_service.agents.keys()),
            "server_time": time.time()
        }
    except Exception as e:
        return {
            "service_status": "error",
            "error": str(e),
            "server_time": time.time()
        }
async def health_check():
    """Health check endpoint"""
    current_state = rl_service.get_training_state()
    return {
        "status": "healthy",
        "rl_service_available": rl_service is not None,
        "current_training_status": current_state["status"],
        "agents_available": list(rl_service.agents.keys())
    }
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
async def auto_extract_user_profile(request: AutoExtractRequest , db: Session=Depends(get_db)):
    """Automatically extract user financial data from bank accounts"""
    
    def safe_get(data: Dict, key: str, default=None):
        """Safely extract and serialize data from nested objects"""
        if not isinstance(data, dict):
            return default
            
        value = data.get(key, default)
        return sanitize_value(value, default)
    
    def sanitize_value(value, default=None):
        """Recursively sanitize values to ensure JSON serializability"""
        try:
            if value is None:
                return default
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [sanitize_value(item) for item in value]
            elif hasattr(value, "to_dict"):
                return sanitize_value(value.to_dict())
            elif hasattr(value, "__dict__"):
                # Handle objects with __dict__ but skip problematic attributes
                obj_dict = {}
                for k, v in value.__dict__.items():
                    if not k.startswith('_') and not callable(v):
                        obj_dict[k] = sanitize_value(v)
                return obj_dict
            else:
                # For any other type, convert to string as last resort
                return str(value)
        except Exception as e:
            print(f"Error sanitizing value {type(value)}: {e}")
            return default

    try:
        # Validate required fields
        if not request.plaid_access_token:
            raise HTTPException(
                status_code=400, 
                detail="plaid_access_token is required"
            )
        
        if not request.email:
            raise HTTPException(
                status_code=400, 
                detail="email is required"
            )

        # Extract bank data
        bank_data = await bank_extractor.extract_financial_data(request.plaid_access_token)
        
        # Check if bank_data is valid
        if not isinstance(bank_data, dict):
            raise HTTPException(
                status_code=500, 
                detail="Invalid response from bank data extractor"
            )
                
        if "error" in bank_data:
            raise HTTPException(
                status_code=400, 
                detail=f"Bank data extraction failed: {bank_data['error']}"
            )

        # Extract financial data with safe_get and additional sanitization
        income = safe_get(bank_data, "monthly_income", 0.0)
        expenses = safe_get(bank_data, "monthly_expenses", 0.0)
        debt = safe_get(bank_data, "estimated_debt", 0.0)
        assets = safe_get(bank_data, "assets", 0.0)
        summary = safe_get(bank_data, "accounts_summary", [])

        # Ensure all values are JSON serializable
        income = sanitize_value(income, 0.0)
        expenses = sanitize_value(expenses, 0.0)
        debt = sanitize_value(debt, 0.0)
        assets = sanitize_value(assets, 0.0)
        summary = sanitize_value(summary, [])

        # Create enhanced profile
        enhanced_profile = UserProfile(
            email=request.email,
            income=income,
            expenses=expenses,
            debt=debt,
            age=request.age,
            assets=assets,
            risk_tolerance=request.risk_tolerance,
            financial_goals=request.financial_goals,
            plaid_access_token=request.plaid_access_token
        )

        # Store profile with sanitized data
        profile_dict = enhanced_profile.dict()
        sanitized_profile = sanitize_value(profile_dict)
        existing_user = db.query(User).filter(User.email == request.email).first()

        if existing_user:
            for key, value in sanitized_profile.items():
                setattr(existing_user, key, value)
            db.commit()
            db.refresh(existing_user)
        else:
            new_user = User(**sanitized_profile)
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
        
        # Debug output
        print("\n\nDEBUG - ENHANCED PROFILE:")
        pprint.pprint(sanitized_profile)

        return {
            "status": "success",
            "message": "Financial data extracted successfully",
            "user_profile": sanitized_profile,
            "bank_summary": summary,
            "extraction_timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Unexpected error in auto_extract_user_profile: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Auto-extraction failed: {str(e)}"
        )
@app.get("/trending-investments/")
async def get_trending_investments(risk_tolerance: str = "moderate"):
    """Get real-time trending investment recommendations"""
    
    # Validate risk_tolerance
    valid_risk_levels = ["conservative", "moderate", "aggressive"]
    if risk_tolerance not in valid_risk_levels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk_tolerance. Must be one of: {valid_risk_levels}"
        )
    
    try:
        trending_data = await investment_analyzer.get_trending_investments(risk_tolerance)
        
        if not trending_data:
            return {
                "status": "success",
                "trending_investments": [],
                "message": "No trending investments found"
            }
        
        return {
            "status": "success",
            "trending_investments": trending_data,
            "risk_tolerance": risk_tolerance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in get_trending_investments: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Investment analysis failed: {str(e)}"
        )

@app.post("/send-investment-alert/")
async def send_investment_alert(alert_request: InvestmentAlertRequest):
    """Send trending investment alert via email"""

    try:
        # Extract validated data
        user_email = alert_request.user_email
        risk_tolerance = alert_request.risk_tolerance

        # Get trending investments
        investment_data = await investment_analyzer.get_trending_investments(risk_tolerance)

        if not investment_data:
            raise HTTPException(
                status_code=404,
                detail="No trending investments found for the specified risk tolerance"
            )

        # Generate email content
        email_service_instance = EmailNotificationService()
        html_body = email_service_instance._create_investment_email_html(investment_data)

        # Send the email
        result = await email_service_instance.send_investment_alert(user_email, html_body)

        return {
            "status": "success",
            "message": "Investment alert sent successfully",
            "email_status": result,
            "user_email": user_email,
            "risk_tolerance": risk_tolerance,
            "investment_count": len(investment_data.get("recommendations", {}).get("stocks", [])) +
                                len(investment_data.get("recommendations", {}).get("crypto", [])),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise  # Re-raise specific FastAPI errors
    except Exception as e:
        print(f"Error in send_investment_alert: {e}")
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

@app.post("/enhanced-hybrid-plan/{email}")
async def generate_enhanced_hybrid_plan(email: str, db: Session = Depends(get_db)):
    """Generate enhanced financial plan with auto-extracted data and trending investments"""
    try:
        # Get user from database
        existing_user = db.query(User).filter(User.email == email).first()
        if not existing_user:
            raise HTTPException(status_code=404, detail="User doesn't exist")
        
        # Convert User object to dictionary properly
        user_dict = user_to_dict(existing_user)
        
        # If bank token provided, extract fresh data
        if user_dict.get('plaid_access_token'):
            try:
                bank_data = await bank_extractor.extract_financial_data(user_dict['plaid_access_token'])
                if bank_data and "error" not in bank_data:
                    # Update user with fresh bank data
                    update_fields = ['assets', 'monthly_income', 'monthly_expenses', 'estimated_debt']
                    for field in update_fields:
                        if field in bank_data and bank_data[field] is not None:
                            # Map bank data fields to user fields
                            user_field = 'income' if field == 'monthly_income' else \
                                        'expenses' if field == 'monthly_expenses' else \
                                        'debt' if field == 'estimated_debt' else field
                            setattr(existing_user, user_field, bank_data[field])
                            user_dict[user_field] = bank_data[field]
                    
                    db.commit()
                    db.refresh(existing_user)
            except Exception as e:
                logger.warning(f"Failed to update bank data: {str(e)}")
        
        # Get trending investments for the user's risk tolerance
        risk_tolerance = user_dict.get('risk_tolerance', 'moderate')
        trending_investments = {}
        try:
            trending_investments = await investment_analyzer.get_trending_investments(risk_tolerance)
        except Exception as e:
            logger.warning(f"Failed to get trending investments: {str(e)}")
            trending_investments = {"error": str(e)}
        
        # Generate hybrid plan if available
        hybrid_plan = None
        
        
        if hybrid_agent:
            try:
                hybrid_plan = await hybrid_agent.generate_hybrid_recommendation(user_dict)
            except Exception as e:
                logger.warning(f"Hybrid agent failed, using fallback: {str(e)}")
        else:
            logger.info("Hybrid agent not available, skipping AI analysis")
        
        # Create enhanced plan with safe data handling
        enhanced_plan = {
            "user_profile": {
                "email": user_dict.get('email'),
                "income": user_dict.get('income', 0),
                "expenses": user_dict.get('expenses', 0),
                "debt": user_dict.get('debt', 0),
                "assets": user_dict.get('assets', 0),
                "age": user_dict.get('age'),
                "risk_tolerance": user_dict.get('risk_tolerance', 'moderate'),
                "financial_goals": user_dict.get('financial_goals')
            },
            "trending_opportunities": trending_investments,
            "hybrid_ai_analysis": hybrid_plan,
            "ai_status": {
                "hybrid_agent_available": hybrid_agent is not None,
                "agent_status": hybrid_agent._get_agent_status() if hybrid_agent else {}
            },
            "recommendations": {
                "immediate_actions": [
                    "Review and optimize current portfolio allocation",
                    "Consider trending investment opportunities based on risk tolerance",
                    "Automate savings and investment contributions"
                ],
                "trending_investments": trending_investments.get("recommendations", {}) if isinstance(trending_investments, dict) else {},
                "risk_assessment": f"Based on {risk_tolerance} risk tolerance"
            },
            "next_review_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "methodology": "hybrid_rl_llm_trending_analysis" if hybrid_plan else "trending_analysis_only"
        }
        
        # Send email notification
        email_sent = False
        if user_dict.get('email'):
            try:
                email_result = await email_service.send_financial_plan_email(user_dict['email'], enhanced_plan)
                email_sent = email_result.get("status") == "success"
            except Exception as e:
                logger.warning(f"Failed to send email: {str(e)}")
        
        return {
            "status": "success",
            "enhanced_financial_plan": enhanced_plan,
            "email_sent": email_sent,
            "user_email": user_dict.get('email')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced planning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced planning failed: {str(e)}")
# User management endpoints
@app.post("/users/register/")
async def register_user(profile: UserProfile, db: Session = Depends(get_db)):
    """Register a new user"""

    # Check if user already exists
    existing_user = db.query(User).filter(User.email == profile.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    # Create new user object
    new_user = User(**profile.dict())

    # Add to database
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "status": "success",
        "message": "User registered successfully",
        "user_email": new_user.email
    }
@app.get("/users/{email}/profile/")
async def get_user_profile(email: str, db : Session= Depends(get_db) ):
    existing_user = db.query(User).filter(User.email == email).first()
    if (not existing_user):
        raise HTTPException(status_code=400, detail="User doesnt exist")
    return {"status": "success", 
            "User Details":existing_user
            }

@app.put("/users/{email}/profile/")
async def update_user_profile(email: str, profile: UserProfile, db: Session = Depends(get_db)):
    """Update user profile"""
    existing_user = db.query(User).filter(User.email == email).first()

    if not existing_user:
        raise HTTPException(status_code=400, detail="User doesn't exist")

    # Update fields
    profile_data = profile.dict(exclude_unset=True)  # Only fields that were provided
    for key, value in profile_data.items():
        setattr(existing_user, key, value)

    db.commit()
    db.refresh(existing_user)

    return {
        "status": "success",
        "message": "Profile updated successfully",
        "user_email": existing_user.email
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)