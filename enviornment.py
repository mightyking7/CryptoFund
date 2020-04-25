import gym
import numpy as np
from gym.spaces import Discrete, Box

class TradingEnv(gym.Env):
    """
    :author Isaac Buitrago

    Environment used to train an agent to manage a fund of crypto assets

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * num_rl: maximum number of controllable vehicles in the network

    States
        The observation consists of the predicted prices for 6 coins
        and the current yield of the fund.
        - Bitcoin
        - Etherum
        - Litecoin
        - Dash
        - Monero
        - Ripple

    Actions
        The action space is discrete and consists of a vector of six
        trading decisions Buy, Sell, and Hold with the amount being traded.

    Rewards
        The reward function is the yield of the fund from the previous day.
        This encourages the agent to create a fund that maximizes ROI.

    Termination
        A rollout is terminated after a single day of trading.
    """
    def __init__(self, env_config):
        self.action_space = Discrete(2)
        self.observation_space = Box(low = 0, high = float('inf'),
                                     shape = (7, ), dtype = np.float32)

    def reset(self):
        return < obs >

    def step(self, action):
        return < obs >, < reward: float >, < done: bool >, < info: dict >