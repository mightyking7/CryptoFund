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
        The observation consists of the number of coins currently
        held in the fund. The 6 coins being traded are
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
        Zero indicates to Hold the current position.

    Rewards
        The reward function is the yield of the fund from the previous day.
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
                                     shape = (6, ), dtype = np.float32)

        # asset prices
        self.btc_price = env_config["btc_price"]
        self.eth_price = env_config["eth_price"]
        self.ltc_price = env_config["ltc_price"]
        self.dash_price = env_config["dash_price"]
        self.xmr_price = env_config["xmr_price"]
        self.xrp_price = env_config["xrp_price"]

        # current holdings in number of coins
        self.fund = {"btc": 1, "eth": 0, "ltc":0, "dash": 0, "xmr": 0, "xrp": 0}

        self.done = False


    def reset(self):
        """
        Resets the environment
        :return:
        """

        # reset dates
        self.start_date = datetime.now()
        self.end_date = self.start_date + timedelta(days=self.rollout_len)

        # default holdings
        self.fund =  {"btc": 1, "eth": 0, "ltc":0, "dash": 0, "xmr": 0, "xrp": 0}
        self.done = False

        return self.fund

    def step(self, action):
        """
        Updates the fund holdings and returns new holdings
        and current yield from previous trading day.

        The agent uses an LSTM model to predict
        the price of each asset.
        
        :param action: vector with amount of each coin to buy/sell/hold

        :return:
        Observation
            vector with amount of each coin in the fund
        Reward
            fund yield from previous day
        Done
            whether training rollout has ended
        Info
            additional info for debugging
        """

        # update positions
        for i, k in enumerate(self.fund):
            self.fund[k] += action[i]

        # compute reward
        roi = (self.current_fund_value - self.previous_fund_value) / self.previous_fund_value

        fund = np.array(self.fund.values())

        if action[0] == self.end_date:
            self.done = True

        return fund, roi, self.done, {}


    def _compute_fund_value(self):
        """
        Multiplys holdings by current prices for each asset.
        :return: value of fund as a float
        """
        self.fund_value = self.fund["btc"] * self.btc_price + \
                          self.fund["eth"] * self.eth_price + \
                          self.fund["ltc"] * self.ltc_price + \
                          self.fund["dash"] * self.dash_price + \
                          self.fund["xmr"] * self.xmr_price + \
                          self.fund["xrp"] * self.xrp_price

        return self.fund_value
