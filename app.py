from fastapi import FastAPI, HTTPException, BackgroundTasks
import os
from models.user_profile import UserProfile
from services.rl_training_service import RLTrainingService
from Agents.Hybrid_RL_LLM import HybridLLMRLAgent
from ai.LLM_client import LLMClient

app = FastAPI(title="RL + LLM Financial Planning Service")

# Initialize services
rl_service = RLTrainingService()
llm_client = LLMClient(os.getenv("OPENAI_API_KEY", ""))

# Initialize hybrid agent
hybrid_agent = HybridLLMRLAgent(llm_client, rl_service.agents)

@app.post("/hybrid-plan/")
async def generate_hybrid_plan(profile: UserProfile):
    """Generate plan using both RL agents and LLM reasoning"""
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
    background_tasks.add_task(rl_service.train_all_agents, timesteps)
    return {"message": f"RL training started with {timesteps} timesteps"}

@app.get("/rl-status/")
async def rl_status():
    """Check RL agent training status"""
    return {
        "agents": list(rl_service.agents.keys()),
        "trained": [name for name, agent in rl_service.agents.items() if agent.model is not None],
        "algorithms": {
            "portfolio": "PPO",
            "debt": "TD3"
        }
    }

@app.post("/rl-only-plan/")
async def generate_rl_only_plan(profile: UserProfile):
    """Generate plan using only RL agents (no LLM)"""
    try:
        user_dict = profile.dict()
        
        rl_recommendations = {}
        if 'portfolio' in rl_service.agents:
            rl_recommendations['portfolio'] = rl_service.agents['portfolio'].analyze_portfolio(user_dict)
        if 'debt' in rl_service.agents:
            rl_recommendations['debt'] = rl_service.agents['debt'].analyze_debt_strategy(user_dict)
        
        return {
            "status": "success",
            "financial_plan": rl_recommendations,
            "methodology": "pure_reinforcement_learning"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RL planning failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)