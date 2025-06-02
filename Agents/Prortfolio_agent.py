import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from typing import Dict, Any
from .base_agent import RLAgentBase

class RLPortfolioAgent(RLAgentBase):
    """
    RL Agent specialized for portfolio allocation decisions
    """
    
    def __init__(self, algorithm: str = "PPO"):
        super().__init__("portfolio_agent")
        self.algorithm = algorithm
        self.setup_model()
    
    def setup_model(self):
        """Initialize the RL model"""
        # Create training environment
        env_config = {
            'max_steps': 120,
            'focus': 'portfolio_optimization'
        }
        self.training_env = make_vec_env(
            lambda: self.create_environment(env_config),
            n_envs=4  # Parallel environments for faster training
        )
        
        # Initialize model based on algorithm choice
        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                self.training_env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log=f"./tensorboard/{self.name}/"
            )
        elif self.algorithm == "SAC":
            self.model = SAC(
                "MlpPolicy",
                self.training_env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                tensorboard_log=f"./tensorboard/{self.name}/"
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def analyze_portfolio(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Use trained RL agent to recommend portfolio allocation"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Convert user profile to environment state
        state = self._profile_to_state(user_profile)
        
        # Get RL recommendation
        action, _states = self.predict(state)
        
        # Convert action to portfolio allocation
        savings_rate, debt_payment_rate, stock_alloc, bond_alloc = action
        
        # Normalize allocations
        total_alloc = stock_alloc + bond_alloc
        if total_alloc > 0:
            stock_alloc /= total_alloc
            bond_alloc /= total_alloc
        
        recommendation = {
            "rl_allocation": {
                "stocks": round(float(stock_alloc), 3),
                "bonds": round(float(bond_alloc), 3),
                "cash": round(1 - float(stock_alloc) - float(bond_alloc), 3)
            },
            "rl_savings_rate": round(float(savings_rate), 3),
            "rl_debt_payment_rate": round(float(debt_payment_rate), 3),
            "confidence": 0.85,
            "method": f"rl_{self.algorithm.lower()}",
            "agent_type": "reinforcement_learning"
        }
        
        return recommendation
    
    def _profile_to_state(self, user_profile: Dict[str, Any]) -> np.ndarray:
        """Convert user profile to RL state representation"""
        return np.array([
            user_profile.get('income', 5000),
            user_profile.get('expenses', 3000),
            user_profile.get('debt', 0),
            user_profile.get('age', 35),
            user_profile.get('assets', 10000),
            0.0,  # market_return (current)
            0.0   # goal_progress (will be calculated)
        ], dtype=np.float32)