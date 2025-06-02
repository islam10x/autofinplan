import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from typing import Dict, Any
from .base_agent import RLAgentBase

class RLDebtAgent(RLAgentBase):
    """
    RL Agent specialized for debt management strategies
    """
    
    def __init__(self):
        super().__init__("debt_agent")
        self.setup_model()
    
    def setup_model(self):
        """Initialize specialized debt management RL model"""
        env_config = {
            'max_steps': 60,  # 5 years for debt payoff
            'focus': 'debt_optimization'
        }
        self.training_env = make_vec_env(
            lambda: self.create_environment(env_config),
            n_envs=4
        )
        
        # Use TD3 for debt management (good for continuous control)
        self.model = TD3(
            "MlpPolicy",
            self.training_env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            tensorboard_log=f"./tensorboard/{self.name}/"
        )
    
    def analyze_debt_strategy(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """RL-based debt management recommendations"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        state = self._profile_to_state(user_profile)
        action, _states = self.predict(state)
        
        # Extract debt-specific recommendations
        _, debt_payment_rate, _, _ = action
        
        debt = user_profile.get('debt', 0)
        income = user_profile.get('income', 1)
        
        # Calculate strategy based on RL recommendation
        monthly_payment = income * float(debt_payment_rate)
        
        if debt <= 0:
            strategy = "none"
            priority = "low"
        elif debt_payment_rate > 0.25:
            strategy = "aggressive_payoff"
            priority = "critical"
        elif debt_payment_rate > 0.15:
            strategy = "balanced_approach"
            priority = "high"
        else:
            strategy = "minimum_plus"
            priority = "medium"
        
        recommendation = {
            "rl_debt_payment_rate": round(float(debt_payment_rate), 3),
            "rl_monthly_payment": round(monthly_payment, 2),
            "rl_strategy": strategy,
            "priority": priority,
            "method": "rl_td3",
            "agent_type": "reinforcement_learning"
        }
        
        return recommendation
