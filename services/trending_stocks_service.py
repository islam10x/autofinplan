import yfinance as yf
import pandas as pd
import requests
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class TrendingStocksMixin:
    async def _get_trending_stocks(self) -> List[Dict]:
        try:
            best_stocks = await self._get_best_sp500_performers()
            if not best_stocks:
                best_stocks = await self._get_top_gainers_fallback()
            return best_stocks[:5]
        except Exception as e:
            logger.error(f"Best stocks analysis failed: {str(e)}")
            return []

    async def _get_best_sp500_performers(self) -> List[Dict]:
        try:
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(sp500_url)
            sp500_df = tables[0]
            symbols = sp500_df['Symbol'].tolist()[:100]
            stock_data = []

            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    hist = stock.history(period="30d")

                    if hist.empty or len(hist) < 5:
                        continue

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

                    if (month_change > -50 and info.get("marketCap", 0) > 500_000_000 and volatility < 200):
                        stock_data.append(stock_info)
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {str(e)}")
                    continue

            stock_data.sort(key=lambda x: x["composite_score"], reverse=True)
            return stock_data
        except Exception as e:
            logger.error(f"S&P 500 analysis failed: {str(e)}")
            return []

    async def _get_top_gainers_fallback(self) -> List[Dict]:
        try:
            symbols = [
                "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "NFLX", "ADBE", "CRM",
                "JPM", "BAC", "WFC", "GS", "MS", "C",
                "UNH", "JNJ", "PFE", "ABBV", "MRK", "TMO",
                "HD", "WMT", "PG", "KO", "PEP", "NKE",
                "BA", "CAT", "GE", "MMM", "HON",
                "XOM", "CVX", "COP", "SLB",
                "NEE", "DUK", "SO"
            ]
            stock_data = []

            for symbol in symbols:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    hist = stock.history(period="30d")

                    if hist.empty:
                        continue

                    current_price = hist['Close'].iloc[-1]
                    month_ago_price = hist['Close'].iloc[0]
                    week_ago_price = hist['Close'].iloc[-5] if len(hist) >= 5 else hist['Close'].iloc[0]

                    month_change = ((current_price - month_ago_price) / month_ago_price) * 100
                    week_change = ((current_price - week_ago_price) / week_ago_price) * 100

                    volatility = hist['Close'].pct_change().std() * 100
                    sharpe_like_ratio = month_change / max(volatility, 1) if volatility > 0 else month_change

                    stock_info = {
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
                    stock_data.append(stock_info)
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {str(e)}")
                    continue

            stock_data.sort(key=lambda x: (x["30day_change_percent"], x["risk_adjusted_return"]), reverse=True)
            return stock_data
        except Exception as e:
            logger.error(f"Fallback analysis failed: {str(e)}")
            return []

    