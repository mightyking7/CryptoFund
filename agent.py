# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:01:56 2020

@author: techm
"""

import pandas as pd
import numpy as np
#import ray
#try:
#    from ray.rllib.agents.agent import get_agent_class
#except ImportError:
#    from ray.rllib.agents.registry import get_agent_class
#from ray.tune import run_experiments
#from ray.tune.registry import register_env

def load_extract(cryptocurrency):
    """
    Loads data used for training/trading
    :param cryptocurrency: crypto to trade
    :return: dataframe of data
    """
    df = pd.read_csv(f'features/{cryptocurrency}.csv')
    df = df['Close'].copy()
    df = df[0:10].copy().transpose()
    return df


# load data
df_btc = load_extract('bitcoin')
df_eth = load_extract('ethereum')
df_dash = load_extract('dash')
df_ltc = load_extract('litecoin')
df_xmr = load_extract('monero')
df_xrp = load_extract('ripple')


