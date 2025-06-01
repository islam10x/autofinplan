# autofinplan/agents/rl_agent.py

import gym
import numpy as np
from stable_baselines3 import PPO

class FinancialEnv(gym.Env):
    """
    Custom Gym environment to simulate a user's financial scenario.
    State: [income, expenses, assets, debt]
    Action: [save_pct, invest_pct, spend_pct] of income
    """
    def __init__(self):
        super(FinancialEnv, self).__init__()

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1_000_000, shape=(4,), dtype=np.float32)

        # Initialize internal state
        self.initial_state = np.array([5000, 2000, 10000, 2000], dtype=np.float32)
        self.state = self.initial_state.copy()

    def reset(self):
        self.state = self.initial_state.copy()
        return self.state

    def step(self, action):
        income, _, assets, debt = self.state
        save_pct, invest_pct, spend_pct = action

        # Normalize action if it doesn't sum to 1
        total_pct = save_pct + invest_pct + spend_pct
        if total_pct > 1:
            save_pct /= total_pct
            invest_pct /= total_pct
            spend_pct /= total_pct

        expenses = spend_pct * income
        returns = invest_pct * income * 1.05  # 5% market return

        assets += save_pct * income + returns
        debt *= 1.01  # compound debt growth

        reward = (assets - debt) / (expenses + 1e-5)  # add small number to avoid divide-by-zero

        self.state = np.array([income, expenses, assets, debt], dtype=np.float32)
        done = False
        return self.state, reward, done, {}

def train_agent():
    env = FinancialEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save("financial_agent")
    print("Model training complete and saved as 'financial_agent'.")

if __name__ == "__main__":
    train_agent()
