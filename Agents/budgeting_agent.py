from .base_agent import BaseAgent
from typing import Dict, Any

class BudgetingAgent(BaseAgent):
    def __init__(self):
        super().__init__("budgeting")
        self.rules = {
            'min_savings_rate': 0.10,
            'max_savings_rate': 0.50,
            'emergency_fund_priority': 0.15
        }
    
    def analyze(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        income = user_profile.get("income", 0)
        expenses = user_profile.get("expenses", 0)
        emergency_months = user_profile.get("emergency_fund_months", 0)
        
        if income <= 0:
            return {"save_pct": self.rules['min_savings_rate'], "priority": "increase_income"}
        
        # Calculate available for savings
        available = income - expenses
        base_save_rate = available / income if available > 0 else 0
        
        # Adjust for emergency fund needs
        if emergency_months < 3:
            emergency_boost = self.rules['emergency_fund_priority']
            save_rate = min(base_save_rate + emergency_boost, self.rules['max_savings_rate'])
            priority = "emergency_fund"
        else:
            save_rate = max(self.rules['min_savings_rate'], 
                          min(base_save_rate, self.rules['max_savings_rate']))
            priority = "long_term_savings"
        
        recommendation = {
            "save_pct": round(save_rate, 3),
            "priority": priority,
            "available_monthly": round(available, 2),
            "confidence": min(1.0, available / (income * 0.2))
        }
        
        self.log_recommendation(user_profile, recommendation)
        return recommendation