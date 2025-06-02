import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from typing import Dict, Any, List, Tuple, Optional
import logging
from ..Environment.financial_env import FinancialPlanningEnv
class RLAgentBase:
    def __init__(self, name: str, model_path: Optional[str] = None):
        self.name = name
        self.model = None
        self.model_path = model_path
        self.training_env = None
        self.logger = logging.getLogger(f"rl_agent.{name}")
        
    def create_environment(self, env_config: Dict[str, Any]) -> gym.Env:
        """Create and configure the training environment"""
        return FinancialPlanningEnv(env_config)
    
    def train(self, total_timesteps: int = 100000, eval_freq: int = 10000):
        """Train the RL agent"""
        if self.training_env is None:
            raise ValueError("Training environment not set")
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            self.training_env,
            best_model_save_path=f"./models/{self.name}_best/",
            log_path=f"./logs/{self.name}/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save the trained model
        if self.model_path:
            self.model.save(self.model_path)
        
        self.logger.info(f"Training completed for {self.name}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        try:
            self.model = self.model.load(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(state, deterministic=deterministic)