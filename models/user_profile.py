from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.accounts_get_request import AccountsGetRequest
import logging

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

class BankDataExtractor:
    """Extract financial data from bank accounts using Plaid API"""
    
    def __init__(self):
        self.client_id = os.getenv("PLAID_CLIENT_ID")
        self.secret = os.getenv("PLAID_SECRET")
        self.environment = os.getenv("PLAID_ENV", "sandbox")  # sandbox, development, production
        
        configuration = plaid.Configuration(
            host=getattr(plaid.Environment, self.environment),
            api_key={
                'clientId': self.client_id,
                'secret': self.secret
            }
        )
        api_client = plaid.ApiClient(configuration)
        self.client = plaid_api.PlaidApi(api_client)
    
    async def extract_financial_data(self, access_token: str) -> Dict:
        """Extract comprehensive financial data from user's bank account"""
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
                "estimated_debt": debt_payments * 12,  # Rough estimate
                "accounts_summary": [
                    {
                        "name": acc['name'],
                        "type": acc['type'],
                        "balance": acc['balances']['current']
                    } for acc in accounts
                ]
            }
            
        except Exception as e:
            logging.error(f"Bank data extraction failed: {str(e)}")
            return {}
    
    def _calculate_monthly_expenses(self, transactions) -> float:
        """Calculate average monthly expenses from transactions"""
        expenses = [t['amount'] for t in transactions if t['amount'] > 0]  # Positive amounts are expenses
        return sum(expenses) / 3 if expenses else 0  # Average over 3 months
    
    def _calculate_monthly_income(self, transactions) -> float:
        """Calculate average monthly income from transactions"""
        income = [abs(t['amount']) for t in transactions if t['amount'] < 0 and t['amount'] < -1000]  # Large negative amounts likely income
        return sum(income) / 3 if income else 0  # Average over 3 months
    
    def _calculate_debt_payments(self, transactions) -> float:
        """Calculate monthly debt payments"""
        debt_keywords = ['loan', 'credit', 'mortgage', 'card', 'payment']
        debt_payments = [
            t['amount'] for t in transactions 
            if t['amount'] > 0 and any(keyword in t['merchant_name'].lower() for keyword in debt_keywords if t.get('merchant_name'))
        ]
        return sum(debt_payments) / 3 if debt_payments else 0

class TrendingInvestmentAnalyzer:
    """Analyze trending investments and market opportunities"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
    
    async def get_trending_investments(self, risk_tolerance: str = "moderate") -> Dict:
        """Get trending investment recommendations based on real-time data"""
        try:
            trending_data = {}
            
            # Get trending stocks
            trending_stocks = await self._get_trending_stocks()
            trending_data["stocks"] = trending_stocks
            
            # Get crypto trends
            crypto_trends = await self._get_crypto_trends()
            trending_data["crypto"] = crypto_trends
            
            # Get sector analysis
            sector_analysis = await self._get_sector_analysis()
            trending_data["sectors"] = sector_analysis
            
            # Filter by risk tolerance
            filtered_recommendations = self._filter_by_risk_tolerance(trending_data, risk_tolerance)
            
            # Get market news
            market_news = await self._get_market_news()
            
            return {
                "recommendations": filtered_recommendations,
                "market_insights": market_news,
                "analysis_timestamp": datetime.now().isoformat(),
                "risk_level": risk_tolerance
            }
            
        except Exception as e:
            logging.error(f"Trending investment analysis failed: {str(e)}")
            return {"error": str(e)}
    
    async def _get_trending_stocks(self) -> List[Dict]:
        """Get trending stocks from various sources"""
        try:
            # Popular stocks to monitor
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
            trending_stocks = []
            
            for symbol in symbols:
                stock = yf.Ticker(symbol)
                info = stock.info
                hist = stock.history(period="5d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    price_change = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    
                    trending_stocks.append({
                        "symbol": symbol,
                        "name": info.get("longName", symbol),
                        "current_price": float(current_price),
                        "5day_change_percent": float(price_change),
                        "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                        "market_cap": info.get("marketCap", 0),
                        "sector": info.get("sector", "Unknown")
                    })
            
            # Sort by 5-day performance
            trending_stocks.sort(key=lambda x: x["5day_change_percent"], reverse=True)
            return trending_stocks[:5]  # Top 5 performers
            
        except Exception as e:
            logging.error(f"Stock trend analysis failed: {str(e)}")
            return []
    
    async def _get_crypto_trends(self) -> List[Dict]:
        """Get trending cryptocurrencies"""
        try:
            # Using CoinGecko API (free)
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "percent_change_24h_desc",
                "per_page": 10,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h,7d"
            }
            
            response = requests.get(url, params=params)
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
                    for coin in data[:5]  # Top 5 trending crypto
                ]
            return []
            
        except Exception as e:
            logging.error(f"Crypto trend analysis failed: {str(e)}")
            return []
    
    async def _get_sector_analysis(self) -> List[Dict]:
        """Analyze sector performance"""
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
            logging.error(f"Sector analysis failed: {str(e)}")
            return []
    
    def _filter_by_risk_tolerance(self, trending_data: Dict, risk_tolerance: str) -> Dict:
        """Filter recommendations based on user's risk tolerance"""
        if risk_tolerance == "conservative":
            # Focus on stable, large-cap stocks and established sectors
            stocks = [s for s in trending_data.get("stocks", []) if s.get("market_cap", 0) > 100_000_000_000]
            crypto = []  # No crypto for conservative investors
            sectors = [s for s in trending_data.get("sectors", []) if s["sector"] in ["Healthcare", "Consumer", "Utilities"]]
        
        elif risk_tolerance == "aggressive":
            # Include all trending investments
            stocks = trending_data.get("stocks", [])
            crypto = trending_data.get("crypto", [])
            sectors = trending_data.get("sectors", [])
        
        else:  # moderate
            # Balanced approach
            stocks = trending_data.get("stocks", [])[:3]  # Top 3 stocks
            crypto = trending_data.get("crypto", [])[:2]  # Top 2 crypto
            sectors = trending_data.get("sectors", [])
        
        return {
            "stocks": stocks,
            "crypto": crypto,
            "sectors": sectors
        }
    
    async def _get_market_news(self) -> List[Dict]:
        """Get relevant market news"""
        try:
            if not self.news_api_key:
                return []
            
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": "stock market OR investment OR finance",
                "sortBy": "publishedAt",
                "pageSize": 5,
                "apiKey": self.news_api_key
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        "title": article["title"],
                        "description": article["description"],
                        "url": article["url"],
                        "published_at": article["publishedAt"]
                    }
                    for article in data.get("articles", [])
                ]
            return []
            
        except Exception as e:
            logging.error(f"Market news fetch failed: {str(e)}")
            return []

class EmailNotificationService:
    """Send email notifications to users"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.email_user = os.getenv("EMAIL_USER")
        self.email_password = os.getenv("EMAIL_PASSWORD")
    
    async def send_investment_alert(self, user_email: str, investment_data: Dict):
        """Send investment recommendation email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = user_email
            msg['Subject'] = "🚀 Trending Investment Opportunities - Your AI Financial Advisor"
            
            # Create HTML email body
            html_body = self._create_investment_email_html(investment_data)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            return {"status": "success", "message": "Investment alert sent successfully"}
            
        except Exception as e:
            logging.error(f"Email sending failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def send_financial_plan_email(self, user_email: str, financial_plan: Dict):
        """Send financial plan via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = user_email
            msg['Subject'] = "📊 Your Personalized Financial Plan - AI Financial Advisor"
            
            html_body = self._create_financial_plan_email_html(financial_plan)
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            return {"status": "success", "message": "Financial plan sent successfully"}
            
        except Exception as e:
            logging.error(f"Email sending failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _create_investment_email_html(self, investment_data: Dict) -> str:
        """Create HTML content for investment email"""
        recommendations = investment_data.get("recommendations", {})
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db;">🚀 Trending Investment Opportunities</h2>
                
                <p>Based on real-time market analysis, here are the trending investment opportunities for you:</p>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="color: #e74c3c;">📈 Top Performing Stocks</h3>
                    <ul>
        """
        
        for stock in recommendations.get("stocks", [])[:3]:
            html += f"""
                        <li><strong>{stock['symbol']}</strong> - {stock['name']}: 
                        ${stock['current_price']:.2f} ({stock['5day_change_percent']:.2f}% 5-day change)</li>
            """
        
        html += """
                    </ul>
                </div>
        """
        
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
            
            html += """
                    </ul>
                </div>
            """
        
        html += f"""
                <div style="background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="color: #0c5460;">🏭 Sector Performance</h3>
                    <ul>
        """
        
        for sector in recommendations.get("sectors", [])[:3]:
            html += f"""
                        <li><strong>{sector['sector']}</strong>: {sector['30day_performance']:.2f}% (30-day performance)</li>
            """
        
        html += f"""
                    </ul>
                </div>
                
                <p style="font-size: 12px; color: #666; margin-top: 30px;">
                    <em>This analysis was generated at {investment_data.get('analysis_timestamp', 'N/A')} 
                    based on your risk tolerance level: {investment_data.get('risk_level', 'moderate')}</em>
                </p>
                
                <p style="font-size: 12px; color: #666;">
                    <strong>Disclaimer:</strong> This is not financial advice. Please consult with a financial advisor before making investment decisions.
                </p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_financial_plan_email_html(self, financial_plan: Dict) -> str:
        """Create HTML content for financial plan email"""
        # This would create a comprehensive HTML email with the financial plan
        # Implementation would depend on the structure of your financial_plan dict
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c3e50;">📊 Your Personalized Financial Plan</h2>
                <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
{financial_plan}
                </pre>
                <p style="font-size: 12px; color: #666; margin-top: 30px;">
                    Generated by your AI Financial Advisor
                </p>
            </div>
        </body>
        </html>
        """

