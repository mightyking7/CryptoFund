# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:01:56 2020

@author: techm
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

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
    df = pd.read_csv(f'input_12mo/{cryptocurrency}.csv')
    df = df['Close'].copy()
    df = df[-183:].copy()
    return df

def load_predict(cryptocurrency):
    """
    Loads data used for training/trading
    :param cryptocurrency: crypto to trade
    :return: dataframe of data
    """
    df = pd.read_csv(f'output_12mo/{cryptocurrency}.csv')
    #df = df.drop(0, axis=0).copy()
    #df = df['Close'].copy()
    #df = df[:-25].copy()
    return df


# load actual data
df_btc = load_extract('bitcoin')
df_eth = load_extract('ethereum')
df_dash = load_extract('dash')
df_ltc = load_extract('litecoin')
df_xmr = load_extract('monero')
df_xrp = load_extract('ripple')

# load predict data
df_btc_pred = load_predict('bitcoin')
df_eth_pred = load_predict('ethereum')
df_dash_pred = load_predict('dash')
df_ltc_pred = load_predict('litecoin')
df_xmr_pred = load_predict('monero')
df_xrp_pred = load_predict('ripple')

# use predicted prices to execute trades for the month of April

trade_days = len(df_btc)

# fund value through out April
april_value = np.zeros((trade_days, 1))

for j in range(trade_days):

    trent_output = pd.DataFrame(data=[df_btc_pred[df_btc_pred.columns[j]],
                                      df_eth_pred[df_eth_pred.columns[j]],
                                      df_dash_pred[df_dash_pred.columns[j]],
                                      df_ltc_pred[df_ltc_pred.columns[j]],
                                      df_xmr_pred[df_xmr_pred.columns[j]],
                                      df_xrp_pred[df_xrp_pred.columns[j]]])
    trent_output.index = ['bitcoin', 'ethereum', 'dash', 'litecoin', 'monero', 'ripple']
    trent_output = trent_output.T

    today_price = np.array([df_btc.to_list()[j],
                            df_eth.to_list()[j],
                            df_dash.to_list()[j],
                            df_ltc.to_list()[j],
                            df_xmr.to_list()[j],
                            df_xrp.to_list()[j]])

    if j == 0:
        #trent_output.index = ['bitcoin', 'ethereum', 'dash', 'litecoin', 'monero', 'ripple']
        #sns.distplot(df_btc, bins=20, kde=False, rug=True)
        #trent_output = trent_output.T
        isaac_input = pd.DataFrame(columns=trent_output.columns)
        isaac_input.loc[0] = np.empty(6)

    # predicted price for each coin
    for coin in list(trent_output.columns):
        if len(trent_output[coin].mode()) < 10:
            modes = list(trent_output[coin].mode())
            mean = trent_output[coin].mean()
            d = defaultdict(float)
            for mode in modes:
                distance = abs(mode - mean)
                d[mode] = distance
            result = min(d.items(), key=lambda x: x[1])
            if j == 0:
                isaac_input[coin] = trent_output[coin].mode()[0]
            else:
                isaac_input.at[coin, 'Pred_Price'] = trent_output[coin].mode()[0]
        else:
            if j == 0:
                isaac_input[coin] = trent_output[coin].median()
            else:
                isaac_input.at[coin, 'Pred_Price'] = trent_output[coin].median()

    # confidence intervals
    c_values = np.zeros(6)

    coin_names = trent_output.columns.to_list()

    gains = np.zeros(6)

    weights = np.array([1, 0, 0, 0, 0, 0])

    # compute confidence for each coin
    for i, coin in enumerate(coin_names):

        if j == 0:
            # one precent range
            one_high, one_low = 1.01 * isaac_input[coin].values[0], .99 * isaac_input[coin].values[0]

            # ten percent range
            ten_high, ten_low = 1.10 * isaac_input[coin].values[0], .90 * isaac_input[coin].values[0]
        else:
            # one precent range
            one_high, one_low = 1.01 * isaac_input.at[coin, 'Pred_Price'], .99 * isaac_input.at[coin, 'Pred_Price']

            # ten percent range
            ten_high, ten_low = 1.10 * isaac_input.at[coin, 'Pred_Price'], .90 * isaac_input.at[coin, 'Pred_Price']

        obs = trent_output[coin].values

        # find observations within one percent and ten percent of the predicted price
        num = np.count_nonzero(np.logical_and(obs >= one_low, obs <= one_high))
        den = np.count_nonzero(np.logical_and(obs >= ten_low, obs <= ten_high))

        c_values[i] = num / den

    # compute percent increase
    if j == 0:
        gains[0] = (isaac_input["bitcoin"].values[0] / df_btc.iloc[j]) - 1
        gains[1] = (isaac_input["ethereum"].values[0] / df_eth.iloc[j]) - 1
        gains[2] = (isaac_input["dash"].values[0] / df_dash.iloc[j]) - 1
        gains[3] = (isaac_input["litecoin"].values[0] / df_ltc.iloc[j]) - 1
        gains[4] = (isaac_input["monero"].values[0] / df_xmr.iloc[j]) - 1
        gains[5] = (isaac_input["ripple"].values[0] / df_xrp.iloc[j]) - 1
    else:
        gains[0] = (isaac_input.at['bitcoin', 'Pred_Price'] / df_btc.iloc[j]) - 1
        gains[1] = (isaac_input.at['ethereum', 'Pred_Price'] / df_eth.iloc[j]) - 1
        gains[2] = (isaac_input.at['dash', 'Pred_Price'] / df_dash.iloc[j]) - 1
        gains[3] = (isaac_input.at['litecoin', 'Pred_Price'] / df_ltc.iloc[j]) - 1
        gains[4] = (isaac_input.at['monero', 'Pred_Price'] / df_xmr.iloc[j]) - 1
        gains[5] = (isaac_input.at['ripple', 'Pred_Price'] / df_xrp.iloc[j]) - 1

    if j == 0:
        isaac_input = isaac_input.T
    isaac_input = isaac_input.rename(columns={0: "Pred_Price"})
    isaac_input['CurrentPrice'] = today_price
    isaac_input['Gain'] = gains
    isaac_input['Weights'] = weights
    isaac_input["C_value"] = c_values

    max_gain = isaac_input.index[isaac_input['Gain'] == isaac_input['Gain'].max()].to_list()[0]

    losses = sorted(isaac_input["Gain"].to_list())

    # trade from currency with least gain to currency with most gain

    for loss in losses:

        index = isaac_input.index[isaac_input['Gain'] == loss].to_list()

        if isaac_input.at[index[0], "Weights"] != 0 and isaac_input.loc[index[0], "Gain"] < 0:
            min_gain = index[0]
            break
        else:
            min_gain = 'None'

    if min_gain != 'None' and min_gain != max_gain:
        allocate = isaac_input.loc[min_gain, "Weights"] * isaac_input.loc[min_gain, "C_value"]

        update = (1 - isaac_input.loc[min_gain, "C_value"]) * isaac_input.loc[min_gain, "Weights"]

        isaac_input.loc[min_gain, "Weights"] = update

        new_currency = (allocate / isaac_input.loc[max_gain, "CurrentPrice"]) * isaac_input.loc[min_gain, "CurrentPrice"]

        isaac_input.loc[max_gain, "Weights"] = new_currency

    fund_value = isaac_input["Weights"] * isaac_input["Pred_Price"]
    #if fund_value.sum() < 4500 and j > 125:
    # If curious in fixing icicle at j = 133rd day
    if j == 133:
        print(max_gain)
        print(min_gain)
        print(update, new_currency)
        print(isaac_input)

    april_value[j] = fund_value.sum()


# plot performance of fund versus BTC
days = np.arange(trade_days)

#plt.title("JKIT performance v.s. BTC April 2020")
#plt.plot(days, april_value, color="blue", label="JKIT Fund")
#plt.plot(days, df_btc.values, color="yellow", label="Bitcoin")
#plt.legend()
#plt.savefig("./images/jkit_performance.png")

sns.set()
f, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=np.array(df_btc), color='b', label="BTC");
sns.lineplot(data=np.array(april_value)[:,0], color='r', label="JKIT");
ax.set(xlabel='Time [days]', ylabel='Close Price [$]')
plt.title("JKIT performance v.s. BTC Last 6 months", fontsize=18)

overall_gain = ((april_value[len(april_value)-1] / np.array(df_btc)[len(df_btc)-1]) - 1) * 100
jkit_gain = ((np.array(april_value)[-1] - np.array(april_value)[0]) / np.array(april_value)[0]) * 100
btc_gain = ((np.array(df_btc)[-1] - np.array(df_btc)[0]) / np.array(df_btc)[0]) * 100
