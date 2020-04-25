# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:01:56 2020

@author: techm
"""

import pandas as pd
import numpy as np


def load_extract(cryptocurrency):
    df = pd.read_csv(f'features/{cryptocurrency}.csv')
    df = df['Close'].copy()
    return df

df_btc = load_extract('bitcoin')
df_eth = load_extract('ethereum')
df_dash = load_extract('dash')
df_ltc = load_extract('litecoin')
df_xmr = load_extract('monero')
df_xrp = load_extract('ripple')

