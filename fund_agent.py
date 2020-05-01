#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:44:40 2020

@author: trentc
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def get_data():
    df = pd.read_csv("newOutputData/bitcoin.csv")
    N = df.shape[1]
    true_price = np.zeros((N+1,6))
    mu = np.zeros((N,6))
    sigma = np.zeros((N,6))
    n = 0
    for name in ["bitcoin","dash","ethereum","litecoin","monero","ripple"]:
        df = pd.read_csv("testLong/"+name+".csv")
        true_price[:,n] = df["Close"][-N-1:]
        
        df = pd.read_csv("newOutputData/"+name+".csv")
        mu[:,n] = np.mean(df, axis=0)
        sigma[:,n] = np.std(df, axis=0)
        n+=1
    
    true_chg = true_price[1:] / true_price[:-1,:] - 1
    est_chg = mu / true_price[:-1,:] - 1
    est_sig = sigma / true_price[:-1,:]
    return true_price, true_chg, est_chg, est_sig
########################################################

def calc_dist(u, s):
    
    mu_delta = u[1:] - u[0]
    sigma_delta = np.sqrt(s[1:]**2 + s[0]**2)
    
    P_best = np.zeros(u.shape) # Pr{largest return}
    P_best[1:] = stats.norm(0,1).cdf(mu_delta/sigma_delta)
    P_best[0] = 1-max(P_best)
    
    weighted_P_largest = P_best * u
    W = weighted_P_largest / sum(weighted_P_largest)
    return W
######################################################

true_price, true_chg, est_chg, est_sig = get_data()

P_lose = stats.norm(est_chg, est_sig).cdf(0)

risk_thold = 0.25 # if P_lose > this, sell it
gain_thold = 0.005 # if est gain is < this, sell it
acct_balance = true_price[0,0] + np.zeros(len(est_chg))

for day in range(len(est_chg)):
    W = np.zeros((6,1)) # distribution of investments
    idx = (P_lose[day,:] < risk_thold) & (est_chg[day,:] > gain_thold)
    if any(idx):
        W[idx,0] = calc_dist(est_chg[day,idx], est_sig[day,idx])
        day_chg = np.matmul(W.T, true_chg[day,:]) + 1
        acct_balance[day] = acct_balance[max(0,day-1)] * day_chg
    else:
        # sell all investments and hold for 0% change
        acct_balance[day] = acct_balance[max(0,day-1)]
    

plt.plot(true_price[:,0],'.-',label="BTC");
plt.plot(acct_balance,'.-',label="JKIT"); plt.grid()
plt.legend()

"""
x = np.linspace(0,0.1,500)
for k in range(6):
    plt.plot(x*100 , stats.norm(est_chg[day,k],est_sig[day,k]).pdf(x))
"""


    