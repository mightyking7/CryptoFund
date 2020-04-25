# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:01:56 2020

@author: techm
"""

import pandas as pd
import numpy as np

df_btc = pd.read_csv('features/bitcoin.csv')
df_eth = pd.read_csv('features/ethereum.csv')
df_dash = pd.read_csv('features/dash.csv')
df_ltc = pd.read_csv('features/litecoin.csv')
df_xmr = pd.read_csv('features/monero.csv')
df_xrp = pd.read_csv('features/ripple.csv')