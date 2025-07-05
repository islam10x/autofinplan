import yfinance as yf
import pandas as pd
import requests
import logging
import asyncio
import aiohttp
from typing import List, Dict
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

class TrendingStocksMixin:
    def __init__(self):
        # Don't create session - let yfinance handle it
        pass
        
    async def _get_trending_stocks(self) -> List[Dict]:
        try:
            # Try the faster fallback method first
            best_stocks = await self._get_top_gainers_fallback()
            if not best_stocks:
                best_stocks = await self._get_best_sp500_performers()
            return best_stocks[:5]
        except Exception as e:
            logger.error(f"Best stocks analysis failed: {str(e)}")
            return []

    async def _get_best_sp500_performers(self) -> List[Dict]:
        try:
            # Reduce the number of symbols processed for speed
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(sp500_url)
            sp500_df = tables[0]
            # Only process top 50 instead of 100 for faster response
            symbols = sp500_df['Symbol'].tolist()[:50]
            
            # Use concurrent processing
            stock_data = await self._process_stocks_concurrent(symbols)
            stock_data.sort(key=lambda x: x["composite_score"], reverse=True)
            return stock_data
        except Exception as e:
            logger.error(f"S&P 500 analysis failed: {str(e)}")
            return []

    async def _process_stocks_concurrent(self, symbols: List[str]) -> List[Dict]:
        """Process stocks concurrently with timeout and error handling"""
        stock_data = []
        
        # Use asyncio with ThreadPoolExecutor for better async integration
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Create tasks for all symbols
            tasks = [
                loop.run_in_executor(executor, self._get_single_stock_data, symbol)
                for symbol in symbols
            ]
            
            # Wait for all tasks with timeout
            try:
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=30)
                
                for result in results:
                    if isinstance(result, dict) and result is not None:
                        stock_data.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Stock processing error: {str(result)}")
                        
            except asyncio.TimeoutError:
                logger.warning("Stock processing timed out after 30 seconds")
                    
        return stock_data

    def _get_single_stock_data(self, symbol: str) -> Dict:
        """Get data for a single stock with timeout and error handling"""
        try:
            # Let yfinance handle sessions automatically
            stock = yf.Ticker(symbol)
            
            # Quick timeout for info and history
            info = stock.info
            hist = stock.history(period="30d")

            if hist.empty or len(hist) < 5:
                return None

            current_price = hist['Close'].iloc[-1]
            month_ago_price = hist['Close'].iloc[0]
            week_ago_price = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]

            month_change = ((current_price - month_ago_price) / month_ago_price) * 100
            week_change = ((current_price - week_ago_price) / week_ago_price) * 100
            volatility = hist['Close'].pct_change().std() * 100
            avg_volume = hist['Volume'].mean()
            recent_volume = hist['Volume'].iloc[-5:].mean()
            volume_trend = (recent_volume / avg_volume) if avg_volume > 0 else 1
            composite_score = (month_change * 0.4 + week_change * 0.3 + 
                               (volume_trend - 1) * 20 - volatility * 0.1)

            stock_info = {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "current_price": float(current_price),
                "30day_change_percent": float(month_change),
                "7day_change_percent": float(week_change),
                "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                "market_cap": info.get("marketCap", 0),
                "sector": info.get("sector", "Unknown"),
                "pe_ratio": info.get("trailingPE", 0),
                "volatility": float(volatility),
                "composite_score": float(composite_score)
            }

            # Filter out problematic stocks early
            if (month_change > -50 and info.get("marketCap", 0) > 500_000_000 and volatility < 200):
                return stock_info
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get data for {symbol}: {str(e)}")
            return None

    async def _get_top_gainers_fallback(self) -> List[Dict]:
        """Optimized fallback with reduced symbol set"""
        try:
            # Reduced and curated list of reliable symbols
            symbols = [
                "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "NFLX", 
                "JPM", "UNH", "JNJ", "HD", "WMT", "PG", "KO", "XOM", "CVX"
            ]
            
            stock_data = await self._process_fallback_stocks_concurrent(symbols)
            stock_data.sort(key=lambda x: (x["30day_change_percent"], x["risk_adjusted_return"]), reverse=True)
            return stock_data
        except Exception as e:
            logger.error(f"Fallback analysis failed: {str(e)}")
            return []

    async def _process_fallback_stocks_concurrent(self, symbols: List[str]) -> List[Dict]:
        """Process fallback stocks concurrently"""
        stock_data = []
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=8) as executor:
            tasks = [
                loop.run_in_executor(executor, self._get_fallback_stock_data, symbol)
                for symbol in symbols
            ]
            
            try:
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=20)
                
                for result in results:
                    if isinstance(result, dict) and result is not None:
                        stock_data.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Fallback stock processing error: {str(result)}")
                        
            except asyncio.TimeoutError:
                logger.warning("Fallback stock processing timed out after 20 seconds")
                    
        return stock_data

    def _get_fallback_stock_data(self, symbol: str) -> Dict:
        """Get fallback stock data with timeout"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="30d")

            if hist.empty:
                return None

            current_price = hist['Close'].iloc[-1]
            month_ago_price = hist['Close'].iloc[0]
            week_ago_price = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]

            month_change = ((current_price - month_ago_price) / month_ago_price) * 100
            week_change = ((current_price - week_ago_price) / week_ago_price) * 100

            volatility = hist['Close'].pct_change().std() * 100
            sharpe_like_ratio = month_change / max(volatility, 1) if volatility > 0 else month_change

            return {
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "current_price": float(current_price),
                "30day_change_percent": float(month_change),
                "7day_change_percent": float(week_change),
                "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                "market_cap": info.get("marketCap", 0),
                "sector": info.get("sector", "Unknown"),
                "risk_adjusted_return": float(sharpe_like_ratio)
            }
        except Exception as e:
            logger.warning(f"Failed to get fallback data for {symbol}: {str(e)}")
            return None