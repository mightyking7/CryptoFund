# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:01:56 2020

@author: techm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    df = df.drop(columns=['30 mavg', '30 std', '26 ema', '12 ema', 'MACD', 'Signal'], axis=1)
    df = df['Close'].copy()
    df = df[0:10].copy()
    return df


# load data
df_btc = load_extract('bitcoin')
df_eth = load_extract('ethereum')
df_dash = load_extract('dash')
df_ltc = load_extract('litecoin')
df_xmr = load_extract('monero')
df_xrp = load_extract('ripple')

trent_output = pd.DataFrame(data=[df_btc, df_eth, df_dash, df_ltc, df_xmr, df_xrp])

trent_output.index = ['bitcoin', 'ethereum', 'dash', 'litecoin', 'monero', 'ripple'] 
sns.distplot(df_btc, bins=20, kde=False, rug=True)

trent_output = trent_output.T
isaac_input = pd.DataFrame(columns=trent_output.columns)
isaac_input.loc[0] = np.empty(6)

for coin in list(trent_output.columns):
    if trent_output[coin].mode()[0] < 10:
        isaac_input[coin] = trent_output[coin].median()
    else:
        isaac_input[coin] = trent_output[coin].mode()
