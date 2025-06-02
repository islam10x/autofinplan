import gymnasium as gym
import numpy as np
from typing import Dict, Any

# autofinplan/environments/financial_env.py
class FinancialPlanningEnv(gym.Env):
    """
    Custom RL Environment for Financial Planning
    
    State Space: [income, expenses, debt, age, risk_tolerance, market_conditions, goal_progress]
    Action Space: [savings_rate, debt_payment_rate, investment_allocation, risk_adjustment]
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        # Environment configuration
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 120)  # 10 years monthly
        self.current_step = 0
        
        # State space: [income, expenses, debt, age, assets, market_return, goal_progress]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 18, 0, -0.5, 0]),  # Min values
            high=np.array([20000, 15000, 500000, 80, 2000000, 0.5, 1.0]),  # Max values
            dtype=np.float32
        )
        
        # Action space: [savings_rate, debt_payment_rate, stock_allocation, bond_allocation]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([0.5, 0.4, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize initial conditions for training diversity
        self.state = np.array([
            np.random.uniform(3000, 12000),  # income
            np.random.uniform(2000, 8000),   # expenses
            np.random.uniform(0, 50000),     # debt
            np.random.uniform(25, 45),       # age
            np.random.uniform(0, 25000),     # assets
            0.0,                             # market_return (will be updated)
            0.0                              # goal_progress
        ], dtype=np.float32)
        
        self.current_step = 0
        self.total_reward = 0
        self.financial_goal = np.random.uniform(100000, 500000)  # Target savings
        
        return self.state, {}
    
    def step(self, action):
        savings_rate, debt_payment_rate, stock_alloc, bond_alloc = action
        
        # Normalize allocations
        total_alloc = stock_alloc + bond_alloc
        if total_alloc > 0:
            stock_alloc /= total_alloc
            bond_alloc /= total_alloc
        
        # Extract current state
        income, expenses, debt, age, assets, _, goal_progress = self.state
        
        # Simulate market conditions
        market_return = np.random.normal(0.07, 0.15)  # 7% mean, 15% volatility
        
        # Calculate monthly changes
        monthly_savings = income * savings_rate
        monthly_debt_payment = min(debt, income * debt_payment_rate)
        
        # Update debt
        new_debt = max(0, debt - monthly_debt_payment)
        
        # Investment returns
        investment_value = assets * (stock_alloc * market_return + bond_alloc * 0.03) / 12
        
        # Update assets
        new_assets = assets + monthly_savings + investment_value
        
        # Update goal progress
        new_goal_progress = min(1.0, new_assets / self.financial_goal)
        
        # Calculate reward
        reward = self._calculate_reward(
            savings_rate, debt_payment_rate, stock_alloc, bond_alloc,
            new_assets, new_debt, new_goal_progress, market_return
        )
        
        # Update state
        self.state = np.array([
            income, expenses, new_debt, age + 1/12, new_assets, 
            market_return, new_goal_progress
        ], dtype=np.float32)
        
        self.current_step += 1
        self.total_reward += reward
        
        # Episode termination conditions
        terminated = (
            self.current_step >= self.max_steps or  # Time limit
            new_goal_progress >= 1.0 or            # Goal achieved
            new_assets < 0                         # Bankruptcy
        )
        
        truncated = False
        info = {
            'total_reward': self.total_reward,
            'goal_progress': new_goal_progress,
            'assets': new_assets,
            'debt': new_debt
        }
        
        return self.state, reward, terminated, truncated, info
    
    def _calculate_reward(self, savings_rate, debt_payment_rate, stock_alloc, bond_alloc,
                         assets, debt, goal_progress, market_return):
        """
        Complex reward function balancing multiple financial objectives
        """
        reward = 0
        
        # Goal progress reward (primary objective)
        reward += goal_progress * 100
        
        # Debt reduction reward
        if debt > 0:
            reward += debt_payment_rate * 50
        
        # Diversification reward
        diversification_score = 1 - abs(stock_alloc - 0.6)  # Penalize extreme allocations
        reward += diversification_score * 10
        
        # Savings discipline reward
        if savings_rate > 0.1:
            reward += 20
        elif savings_rate < 0.05:
            reward -= 10
        
        # Risk-adjusted return reward
        if assets > 0:
            reward += (market_return * stock_alloc + 0.03 * bond_alloc) * 10
        
        # Penalty for unrealistic allocations
        if savings_rate + debt_payment_rate > 0.6:
            reward -= 50  # Can't allocate more than 60% of income
        
        return reward