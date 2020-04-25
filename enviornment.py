import gym
import numpy as np
from gym.spaces import Box
from datetime import datetime, timedelta


class TradingEnv(gym.Env):
    """
    :author Isaac Buitrago

    Environment used to train an agent to manage a fund of crypto assets

    Required from env_params:

    # This is the value of the fund
    value = w1 * Xbtc + w2 * Xeth + w3 * Xdash + w4 * Xltc + w5 * Xxmr + w6 * Xxrp
    Where w is the ammount of each asset that is owned.

    States
        The observation consists of the predicted prices for 6 coins
        and the current yield of the fund. The 6 coins being traded are
        - Bitcoin
        - Etherum
        - Litecoin
        - Dash
        - Monero
        - Ripple

    Actions
        The action space consists of a vector of six floats that represent trading decisions.

        Positive values indicate the number of coins to Buy.
        Negative values indicate the number of coins to Sell.
        Zero indicates to hold the current position.

    Rewards
        The reward function is the yield of the fund from the previous step.
        This encourages the agent to create a fund that maximizes ROI.

    Termination
        A rollout is terminated after 14 days of trading.
    """
    def __init__(self, env_config):

        self.rollout_len = 14
        self.start_date = datetime.now()
        self.end_date = self.start_date + timedelta(days=self.rollout_len)

        self.action_space = Box(low = - float('inf'), high = float('inf'),
                                     shape = (6, ), dtype = np.float32)

        self.observation_space = Box(low = 0, high = float('inf'),
                                     shape = (7, ), dtype = np.float32)

        self.fund_value = 1 * env_config["btc_price"] + 0 * env_config["eth_price"] + 0 * env_config["dash_price"]\
                          + 0 * env_config["ltc_price"] + 0 * env_config["xmr_price"] + 0 * env_config["xrp_price"]


    def reset(self):
        """
        Resets the environment
        :return:
        """
        self.start_date = datetime.now()
        self.end_date = self.start_date + timedelta(days=self.rollout_len)

        # number of coins to trade
        trades = np.array([1, 0, 0, 0, 0, 0])

        return trades

    def step(self, action):
        return < obs >, < reward: float >, < done: bool >, < info: dict >