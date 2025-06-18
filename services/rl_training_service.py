from Agents.Prortfolio_agent import RLPortfolioAgent
from Agents.dept_agent import RLDebtAgent

class RLTrainingService:
    """
    Service for training and managing RL agents
    """
    
    def __init__(self):
        self.agents = {
            'portfolio': RLPortfolioAgent("PPO"),
            'debt': RLDebtAgent()
        }
        
    async def train_all_agents(self, timesteps: int = 100000):
        """Train all RL agents"""
        for name, agent in self.agents.items():
            print(f"Training {name} agent...")
            agent.train(total_timesteps=timesteps)
            print(f"{name} agent training completed!")
    
    def save_models(self, base_path: str = "./models/"):
        """Save all trained models"""
        for name, agent in self.agents.items():
            model_path = f"{base_path}{name}_model"
            agent.model.save(model_path)
            print(f"Saved {name} model to {model_path}")
    
    def load_models(self, base_path: str = "./models/"):
        """Load pre-trained models"""
        for name, agent in self.agents.items():
            model_path = f"{base_path}{name}_model"
            try:
                agent.load_model(model_path)
                print(f"Loaded {name} model from {model_path}")
            except Exception as e:
                print(f"Failed to load {name} model: {e}")