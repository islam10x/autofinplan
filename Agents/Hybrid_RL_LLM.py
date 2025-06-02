import json
import logging
from typing import Dict, Any
from .base_agent import RLAgentBase

class HybridLLMRLAgent:
    """
    Advanced agent that combines RL decision-making with LLM reasoning
    """
    
    def __init__(self, llm_client, rl_agents: Dict[str, RLAgentBase]):
        self.llm_client = llm_client
        self.rl_agents = rl_agents
        self.logger = logging.getLogger("hybrid_agent")
        
        self.integration_prompt = """
You are a senior financial advisor working with AI assistants. You have access to recommendations from specialized RL algorithms that have been trained on thousands of financial scenarios.

RL Agent Recommendations:
{rl_recommendations}

User Profile:
{user_profile}

Your tasks:
1. Interpret and validate the RL recommendations
2. Provide human-readable explanations for the algorithmic decisions
3. Identify any potential issues or risks the RL agents might miss
4. Suggest modifications based on qualitative factors RL can't capture
5. Create a narrative that explains WHY these recommendations make sense

Consider:
- RL agents optimize for mathematical reward functions
- Real humans have emotional and psychological factors
- Market conditions and economic context
- Life events and personal circumstances
- Risk tolerance vs. risk capacity differences

Provide your analysis in JSON format:
{{
    "rl_validation": {{
        "portfolio_recommendation": "approved/modified/rejected",
        "debt_recommendation": "approved/modified/rejected",
        "modifications_suggested": ["change1", "change2", ...]
    }},
    "human_readable_explanation": {{
        "why_this_portfolio": "detailed explanation",
        "debt_strategy_rationale": "detailed explanation",
        "risk_considerations": "detailed explanation"
    }},
    "qualitative_adjustments": {{
        "emotional_factors": ["factor1", "factor2", ...],
        "life_stage_considerations": ["consideration1", "consideration2", ...],
        "market_timing_thoughts": "current market analysis"
    }},
    "final_recommendations": {{
        "portfolio_allocation": {{"stocks": 0.XX, "bonds": 0.XX, "cash": 0.XX}},
        "savings_rate": 0.XX,
        "debt_payment_rate": 0.XX,
        "implementation_notes": ["note1", "note2", ...]
    }},
    "confidence_reasoning": "explanation of confidence level"
}}
"""
    
    async def generate_hybrid_recommendation(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine RL agent recommendations with LLM reasoning
        """
        try:
            # Get RL recommendations
            rl_recommendations = {}
            
            if 'portfolio' in self.rl_agents:
                rl_recommendations['portfolio'] = self.rl_agents['portfolio'].analyze_portfolio(user_profile)
            
            if 'debt' in self.rl_agents:
                rl_recommendations['debt'] = self.rl_agents['debt'].analyze_debt_strategy(user_profile)
            
            # Get LLM analysis of RL recommendations
            prompt = self.integration_prompt.format(
                rl_recommendations=json.dumps(rl_recommendations, indent=2),
                user_profile=json.dumps(user_profile, indent=2)
            )
            
            llm_response = await self.llm_client.generate_response(prompt, max_tokens=1200)
            
            try:
                llm_analysis = json.loads(llm_response)
            except json.JSONDecodeError:
                return self._fallback_integration(rl_recommendations, user_profile)
            
            # Combine RL precision with LLM wisdom
            hybrid_recommendation = {
                "hybrid_approach": {
                    "rl_base_recommendations": rl_recommendations,
                    "llm_validation": llm_analysis.get("rl_validation", {}),
                    "llm_explanations": llm_analysis.get("human_readable_explanation", {}),
                    "qualitative_adjustments": llm_analysis.get("qualitative_adjustments", {}),
                    "final_recommendations": llm_analysis.get("final_recommendations", {})
                },
                "method": "hybrid_rl_llm",
                "confidence": 0.92,  # High confidence from combining both approaches
                "reasoning": llm_analysis.get("confidence_reasoning", "")
            }
            
            return hybrid_recommendation
            
        except Exception as e:
            self.logger.error(f"Hybrid recommendation failed: {e}")
            return self._fallback_integration(rl_recommendations if 'rl_recommendations' in locals() else {}, user_profile)
    
    def _fallback_integration(self, rl_recs: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback when hybrid integration fails"""
        return {
            "hybrid_approach": {
                "rl_base_recommendations": rl_recs,
                "method": "rl_only_fallback"
            },
            "confidence": 0.7
        }
